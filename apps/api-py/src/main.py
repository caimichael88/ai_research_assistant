from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import controllers with proper module path handling
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controllers import ai_controller, health_controller, asr_controller, tts_controller

app = FastAPI(
    title="AI Research Assistant API",
    description="A voice agent to provide ",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(ai_controller.router)
app.include_router(asr_controller.router)
app.include_router(health_controller.router)
app.include_router(tts_controller.router)

@app.get("/")
async def root():
    """root end point"""
    return {"message": "AI Research Voice Agent application", "version": "1.0.0"}
