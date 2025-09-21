from __future__ import annotations
"""
TTS Controller - FastAPI endpoints for text-to-speech synthesis

Provides the following endpoints:
1. POST /v1/tts/synthesize - Synchronous TTS synthesis
2. GET /v1/tts/stream - HTTP chunked streaming synthesis
3. GET /v1/voices - List available voices
4. GET /healthz - Health check

Uses the TTSEngine port for actual synthesis operations.
"""
from typing import List, Optional, Iterator
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
import logging

from  components.ports import SynthesisRequest, AudioFormat, TTSEngine
from services.tts_service import get_tts_service

logger = logging.getLogger(__name__)

# Router with proper prefix and tags
router = APIRouter(prefix="/v1/tts", tags=["tts"])

# ------------------- Request/Response Models -------------------

class SynthesizeRequest(BaseModel):
    """Request model for synchronous TTS synthesis"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_id: str = Field("en_female_1", description="Voice identifier")
    sample_rate: int = Field(22050, ge=8000, le=48000, description="Audio sample rate")
    speed: float = Field(1.0, ge=0.1, le=3.0, description="Speech speed multiplier")
    pitch_semitones: int = Field(0, ge=-12, le=12, description="Pitch adjustment in semitones")
    format: AudioFormat = Field(AudioFormat.wav, description="Output audio format")
    engine_id: str = Field("neural", description="TTS engine to use")
    vocoder_id: str = Field("hifigan", description="Vocoder for neural engine")
    normalize_loudness: bool = Field(True, description="Apply loudness normalization")

class VoiceInfo(BaseModel):
    """Information about an available voice"""
    voice_id: str
    name: str
    language: str
    gender: str
    sample_rate: int
    description: Optional[str] = None

class VoicesResponse(BaseModel):
    """Response containing list of available voices"""
    voices: List[VoiceInfo]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
    engines: List[str]

# ------------------- Endpoints -------------------

@router.post("/synthesize", response_class=Response)
async def synthesize_speech(
    req: SynthesizeRequest,
    tts_service: TTSEngine = Depends(get_tts_service)
) -> Response:
    """
    Synchronous text-to-speech synthesis

    Returns audio data directly as response body with appropriate Content-Type.
    """
    try:
        # Convert to internal request format
        synthesis_req = SynthesisRequest(
            text=req.text,
            voice_id=req.voice_id,
            sample_rate=req.sample_rate,
            speed=req.speed,
            pitch_semitones=req.pitch_semitones,
            fmt=req.format,
            engine_id=req.engine_id,
            vocoder_id=req.vocoder_id,
            normalize_loudness=req.normalize_loudness,
        )

        # Perform synthesis
        result = tts_service.synthesize_sync(synthesis_req)

        # Return audio response
        headers = {
            "X-Model-Header": result.model_header,
            "X-Cache-Status": result.cache,
        }
        if result.latency_ms:
            headers["X-Latency-Ms"] = str(result.latency_ms)

        return Response(
            content=result.audio,
            media_type=result.media_type,
            headers=headers
        )

    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@router.get("/stream")
async def stream_speech(
    text: str = Query(..., min_length=1, max_length=5000, description="Text to synthesize"),
    voice_id: str = Query("en_female_1", description="Voice identifier"),
    sample_rate: int = Query(22050, ge=8000, le=48000, description="Audio sample rate"),
    speed: float = Query(1.0, ge=0.1, le=3.0, description="Speech speed multiplier"),
    pitch_semitones: int = Query(0, ge=-12, le=12, description="Pitch adjustment"),
    format: AudioFormat = Query(AudioFormat.wav, description="Output audio format"),
    engine_id: str = Query("neural", description="TTS engine to use"),
    vocoder_id: str = Query("hifigan", description="Vocoder for neural engine"),
    normalize_loudness: bool = Query(True, description="Apply loudness normalization"),
    tts_service: TTSEngine = Depends(get_tts_service)
) -> StreamingResponse:
    """
    Streaming text-to-speech synthesis

    Returns audio data as HTTP chunked response for progressive playback.
    """
    try:
        # Convert to internal request format
        synthesis_req = SynthesisRequest(
            text=text,
            voice_id=voice_id,
            sample_rate=sample_rate,
            speed=speed,
            pitch_semitones=pitch_semitones,
            fmt=format,
            engine_id=engine_id,
            vocoder_id=vocoder_id,
            normalize_loudness=normalize_loudness,
        )

        def audio_generator() -> Iterator[bytes]:
            """Generate audio chunks for streaming"""
            try:
                # Check if service supports streaming
                if hasattr(tts_service, 'synthesize_stream_sync'):
                    for chunk in tts_service.synthesize_stream_sync(synthesis_req):
                        yield chunk
                else:
                    # Fallback: synthesize fully then stream in chunks
                    result = tts_service.synthesize_sync(synthesis_req)
                    chunk_size = 8192
                    audio_data = result.audio
                    for i in range(0, len(audio_data), chunk_size):
                        yield audio_data[i:i + chunk_size]
            except Exception as e:
                logger.error(f"Streaming synthesis failed: {e}")
                # Note: Can't raise HTTPException from generator, client will see connection close
                return

        # Determine media type
        media_type = f"audio/{format.value}"

        return StreamingResponse(
            audio_generator(),
            media_type=media_type,
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        logger.error(f"Streaming setup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

@router.get("/voices", response_model=VoicesResponse)
async def list_voices() -> VoicesResponse:
    """
    List available voices

    Returns information about all available TTS voices.
    """
    try:
        # TODO: Make this dynamic based on loaded models
        voices = [
            VoiceInfo(
                voice_id="en_female_1",
                name="English Female 1",
                language="en-US",
                gender="female",
                sample_rate=22050,
                description="Neural English female voice"
            ),
            VoiceInfo(
                voice_id="en_male_1",
                name="English Male 1",
                language="en-US",
                gender="male",
                sample_rate=22050,
                description="Neural English male voice"
            ),
        ]

        return VoicesResponse(voices=voices)

    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve voice list")

@router.get("/healthz", response_model=HealthResponse)
async def health_check(
    tts_service: TTSEngine = Depends(get_tts_service)
) -> HealthResponse:
    """
    Health check endpoint

    Returns service health status and available engines.
    """
    try:
        # Test that the service is responsive
        available_engines = []

        # Check if neural engine is available
        if tts_service and hasattr(tts_service, 'id'):
            available_engines.append(tts_service.id)

        return HealthResponse(
            status="healthy",
            engines=available_engines
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            engines=[]
        )

# ------------------- Utility Functions -------------------

def validate_synthesis_request(req: SynthesisRequest) -> None:
    """Validate synthesis request parameters"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(req.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")

    if req.speed < 0.1 or req.speed > 3.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.1 and 3.0")