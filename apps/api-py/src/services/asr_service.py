"""Automatic Speech Recognition service using OpenAI Whisper."""

import os
import tempfile
from typing import Optional, Dict, Any
import whisper
import torch
from pathlib import Path


class ASRService:
    """Service for automatic speech recognition using Whisper."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self.available_models = ["tiny", "base", "small", "medium", "large"]
        self.default_model = "base"

    def _get_model(self, model_name: str = "base"):
        """Load and cache Whisper model."""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Choose from: {self.available_models}")

        if model_name not in self._models:
            print(f"Loading Whisper model: {model_name}")
            self._models[model_name] = whisper.load_model(model_name)

        return self._models[model_name]

    async def transcribe_audio(
        self,
        audio_file: bytes,
        filename: str,
        model_name: str = "base",
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_file: Audio file content as bytes
            filename: Original filename for format detection
            model_name: Whisper model to use
            language: Optional language code (e.g., 'en', 'es', 'fr')

        Returns:
            Dictionary with transcription results
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                tmp_file.write(audio_file)
                tmp_file_path = tmp_file.name

            try:
                # Load model
                model = self._get_model(model_name)

                # Transcribe
                options = {}
                if language:
                    options["language"] = language

                result = model.transcribe(tmp_file_path, **options)

                return {
                    "text": result["text"].strip(),
                    "language": result.get("language", "unknown"),
                    "segments": [
                        {
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"].strip()
                        }
                        for segment in result.get("segments", [])
                    ],
                    "model_used": model_name,
                    "filename": filename
                }

            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")

    def preload_default_model(self) -> None:
        """Preload the default model at startup for faster first request."""
        try:
            print(f"Preloading Whisper model: {self.default_model}")
            self._get_model(self.default_model)
            print(f"✅ Whisper model '{self.default_model}' preloaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to preload Whisper model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        return {
            "available_models": self.available_models,
            "default_model": self.default_model,
            "loaded_models": list(self._models.keys()),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_descriptions": {
                "tiny": "39 MB, ~32x realtime speed",
                "base": "74 MB, ~16x realtime speed",
                "small": "244 MB, ~6x realtime speed",
                "medium": "769 MB, ~2x realtime speed",
                "large": "1550 MB, ~1x realtime speed"
            }
        }


# Global service instance
asr_service = ASRService()