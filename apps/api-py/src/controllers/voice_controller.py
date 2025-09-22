"""Voice controller for handling voice-to-voice conversations using LangGraph."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from services.langgraph_service import get_voice_agent
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/")
async def voice_conversation(
    file: UploadFile = File(..., description="Audio file for voice conversation")
):
    """
    Process voice conversation through intelligent LangGraph agent:
    Audio → LangGraph → [ASR → LLM → TTS] → Audio

    The LangGraph agent intelligently decides which tools to use and when.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check file size (limit to 25MB)
    max_size = 25 * 1024 * 1024  # 25MB
    audio_content = await file.read()
    if len(audio_content) > max_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 25MB")

    try:
        logger.info(f"Processing voice conversation: {file.filename}")

        # Direct sequential approach: ASR -> LLM -> TTS
        # Step 1: Transcribe audio
        from services.asr_service import asr_service
        transcription_result = await asr_service.transcribe_audio(
            audio_file=audio_content,
            filename=file.filename,
            model_name="base"
        )
        user_text = transcription_result["text"]
        logger.info(f"Transcribed: {user_text}")

        # Step 2: Process with LLM (placeholder for now)
        ai_response = f"I heard you say: '{user_text}'. This is a test response from the AI assistant."
        logger.info(f"AI Response: {ai_response}")

        # Step 3: Generate speech with TTS
        from services.tts_service import get_tts_service
        from components.ports import SynthesisRequest
        import tempfile

        tts_engine = get_tts_service()
        synthesis_request = SynthesisRequest(
            text=ai_response,
            voice_id="en_female_1",
            sample_rate=22050
        )

        tts_result = tts_engine.synthesize_sync(synthesis_request)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with open(temp_file.name, "wb") as f:
            f.write(tts_result.audio)

        audio_response_path = temp_file.name

        if audio_response_path:
            # Return the generated audio file
            return FileResponse(
                audio_response_path,
                media_type="audio/wav",
                filename="response.wav"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate audio response")

    except Exception as e:
        logger.error(f"Error in voice conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent")
async def voice_conversation_agent(
    file: UploadFile = File(..., description="Audio file for voice conversation via LangGraph")
):
    """
    Process voice conversation through LangGraph agent:
    Audio → LangGraph Agent → [ASR → LLM → TTS] → Audio

    The LangGraph agent intelligently orchestrates the ASR, LLM, and TTS tools.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check file size (limit to 25MB)
    max_size = 25 * 1024 * 1024  # 25MB
    audio_content = await file.read()
    if len(audio_content) > max_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 25MB")

    try:
        logger.info(f"Processing voice conversation via LangGraph: {file.filename}")

        # Use LangGraph agent for processing
        voice_agent = get_voice_agent()
        audio_response_path = voice_agent.process_voice_conversation(
            audio_data=audio_content,
            filename=file.filename
        )

        if audio_response_path:
            # Return the generated audio file
            return FileResponse(
                audio_response_path,
                media_type="audio/wav",
                filename="agent_response.wav"
            )
        else:
            raise HTTPException(status_code=500, detail="LangGraph agent failed to generate audio response")

    except Exception as e:
        logger.error(f"Error in LangGraph voice conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def voice_health_check():
    """Health check for voice service."""
    try:
        # Get voice agent and check its status
        voice_agent = get_voice_agent()

        return {
            "status": "healthy",
            "service": "Voice (Intelligent LangGraph Agent)",
            "langgraph_status": "initialized",
            "tools": ["transcribe_audio", "process_with_llm", "synthesize_speech"]
        }
    except Exception as e:
        logger.error(f"Voice health check failed: {e}")
        return {
            "status": "error",
            "service": "Voice (Intelligent LangGraph Agent)",
            "error": str(e)
        }