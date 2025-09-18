"""ASR (Automatic Speech Recognition) controller using Whisper."""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from services.asr_service import asr_service

router = APIRouter(prefix="/asr", tags=["asr"])


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""

    text: str
    language: str
    segments: List[Dict[str, Any]]
    model_used: str
    filename: str


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    available_models: List[str]
    default_model: str
    loaded_models: List[str]
    device: str
    model_descriptions: Dict[str, str]


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form("base", description="Whisper model to use"),
    language: Optional[str] = Form(None, description="Language code (optional, auto-detect if not provided)")
):
    """
    Transcribe an audio file using OpenAI Whisper.

    Supported audio formats: mp3, wav, m4a, flac, ogg, and more.
    """
    # Validate file
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check file size (limit to 25MB)
    max_size = 25 * 1024 * 1024  # 25MB
    audio_content = await audio_file.read()
    if len(audio_content) > max_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 25MB")

    # Validate model
    if model not in asr_service.available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Available models: {asr_service.available_models}"
        )

    try:
        result = await asr_service.transcribe_audio(
            audio_file=audio_content,
            filename=audio_file.filename,
            model_name=model,
            language=language
        )

        return TranscriptionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about available Whisper models."""
    try:
        info = asr_service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def asr_health_check():
    """Health check for ASR service."""
    try:
        info = asr_service.get_model_info()
        return {
            "status": "healthy",
            "service": "ASR (Whisper)",
            "device": info["device"],
            "models_loaded": len(info["loaded_models"])
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "ASR (Whisper)",
            "error": str(e)
        }