"""FastAPI application for AI Research Assistant."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controllers import health_controller, ai_controller, asr_controller

app = FastAPI(
    title="AI Research Assistant API",
    description="Python API for AI components of the research assistant",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_controller.router)
app.include_router(ai_controller.router)
app.include_router(asr_controller.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI Research Assistant Python API", "version": "0.1.0"}