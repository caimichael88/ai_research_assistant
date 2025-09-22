from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env file from the api-py directory (parent of src)
env_path = Path(__file__).parent.parent / '.env'
loaded = load_dotenv(env_path)
print(f"Environment file loaded: {loaded}, path: {env_path}")
print(f"TTS_ENGINE={os.getenv('TTS_ENGINE')}, FAKE_TTS={os.getenv('FAKE_TTS')}")

# Import controllers with proper module path handling
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controllers import ai_controller, health_controller, asr_controller, tts_controller, voice_controller

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # STARTUP - runs when server starts
    print("🚀 Starting AI Research Assistant API...")

    # Preload Whisper ASR model
    try:
        from services.asr_service import asr_service
        print("📥 Preloading ASR models...")
        asr_service.preload_default_model()
    except Exception as e:
        print(f"⚠️ ASR model preloading failed: {e}")

    # Preload TTS model (warmup)
    try:
        from services.tts_service import get_tts_service
        print("🎤 Preloading TTS models...")
        tts_engine = get_tts_service()
        if hasattr(tts_engine, 'warmup'):
            tts_engine.warmup()
            print("✅ TTS model preloaded successfully")
        else:
            print("ℹ️ TTS engine doesn't support warmup")
    except Exception as e:
        print(f"⚠️ TTS model preloading failed: {e}")

    # Preload llm model
    try:
        from services.llm_service import llm_service
        print("Preloading LLM service")
        model, tokenizer, device = llm_service.load_trained_model()
    except Exception as e:
         print(f"⚠️ LLM model preloading failed: {e}")
    
   
    print("✅ AI Research Assistant API ready!")

    yield  # Server runs here, handling requests

    # SHUTDOWN - runs when server stops
    print("🛑 Shutting down AI Research Assistant API...")

app = FastAPI(
    title="AI Research Assistant API",
    description="A voice agent to provide ",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(ai_controller.router)
app.include_router(asr_controller.router)
app.include_router(health_controller.router)
app.include_router(tts_controller.router)
app.include_router(voice_controller.router)

@app.get("/")
async def root():
    """root end point"""
    return {"message": "AI Research Voice Agent application", "version": "1.0.0"}
