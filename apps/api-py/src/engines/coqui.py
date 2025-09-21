from __future__ import annotations
"""
Coqui TTS Engine Adapter

This adapter provides a complete TTS engine using Coqui TTS library.
Unlike the neural engine which separates Text2Mel and Vocoder,
Coqui TTS handles the entire pipeline internally.

Features:
- Multiple voice models support
- Built-in text normalization
- GPU/CPU acceleration
- Streaming support (via chunking)
- Voice cloning capabilities

Usage:
    engine = CoquiTTSEngine(
        model_name="tts_models/en/ljspeech/tacotron2-DDC_ph",
        device="cpu"
    )

    req = SynthesisRequest(text="Hello world", voice_id="ljspeech")
    result = engine.synthesize_sync(req)
"""
import os
import io
import logging
from typing import Dict, List, Optional, Iterator, Any
from dataclasses import dataclass
import tempfile

try:
    from ..components.ports import (
        TTSEngine,
        SynthesisRequest,
        SynthesisResult,
        AudioFormat,
        TTSSynthesisError,
        pcm16_to_wav,
    )
except ImportError:
    # Fallback for direct execution
    from components.ports import (
        TTSEngine,
        SynthesisRequest,
        SynthesisResult,
        AudioFormat,
        TTSSynthesisError,
        pcm16_to_wav,
    )

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CoquiConfig:
    """Configuration for Coqui TTS engine"""
    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC_ph"
    device: str = "cpu"
    use_gpu: bool = False
    sample_rate: int = 22050
    # Voice mapping
    voice_models: Dict[str, str] = None

    def __post_init__(self):
        if self.voice_models is None:
            # Default voice mappings
            object.__setattr__(self, 'voice_models', {
                "en_female_1": "tts_models/en/ljspeech/tacotron2-DDC_ph",
                "en_male_1": "tts_models/en/ljspeech/tacotron2-DDC_ph",  # Same model for now
                "ljspeech": "tts_models/en/ljspeech/tacotron2-DDC_ph",
                "vctk": "tts_models/en/vctk/vits",
                "jenny": "tts_models/en/jenny/jenny",
            })

class CoquiTTSEngine(TTSEngine):
    """Complete TTS engine using Coqui TTS library"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[CoquiConfig] = None
    ):
        """
        Initialize Coqui TTS engine

        Args:
            model_name: Coqui model name (e.g., "tts_models/en/ljspeech/tacotron2-DDC_ph")
            device: Device to use ("cpu", "cuda", "mps")
            config: Optional configuration object
        """
        if config is None:
            config = CoquiConfig()

        # Override config with provided parameters
        if model_name:
            config = CoquiConfig(
                model_name=model_name,
                device=device or config.device,
                use_gpu=config.use_gpu,
                sample_rate=config.sample_rate,
                voice_models=config.voice_models
            )
        elif device:
            config = CoquiConfig(
                model_name=config.model_name,
                device=device,
                use_gpu=config.use_gpu,
                sample_rate=config.sample_rate,
                voice_models=config.voice_models
            )

        self._config = config
        self._tts_instances: Dict[str, Any] = {}
        self._loaded = False

        # Environment overrides
        self._device = os.getenv("COQUI_DEVICE", self._config.device)
        self._use_gpu = os.getenv("COQUI_USE_GPU", "0") == "1"

        logger.info(f"Initializing Coqui TTS engine - device: {self._device}")

    @property
    def id(self) -> str:
        return "coqui"

    def warmup(self) -> None:
        """Load the default model for faster first synthesis"""
        logger.info("Warming up Coqui TTS engine...")
        try:
            # Load default model
            self._get_tts_instance(self._config.model_name)

            # Quick test synthesis
            self._perform_synthesis(
                "Warmup test",
                self._config.model_name,
                sample_rate=self._config.sample_rate
            )

            logger.info("Coqui TTS engine warmed up successfully")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def synthesize_sync(self, req: SynthesisRequest) -> SynthesisResult:
        """Synchronous text-to-speech synthesis"""
        import time
        start_time = time.time()

        try:
            # Map voice ID to model name
            model_name = self._get_model_for_voice(req.voice_id)

            # Perform synthesis
            audio_data = self._perform_synthesis(
                text=req.text,
                model_name=model_name,
                sample_rate=req.sample_rate,
                speed=req.speed
            )

            # Convert to requested format
            if req.fmt == AudioFormat.wav:
                # Coqui typically returns numpy array, convert to WAV
                audio_bytes = self._numpy_to_wav(audio_data, req.sample_rate)
                media_type = "audio/wav"
            else:
                # For other formats, assume raw audio for now
                audio_bytes = audio_data.tobytes() if hasattr(audio_data, 'tobytes') else audio_data
                media_type = f"audio/{req.fmt.value}"

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            return SynthesisResult(
                audio=audio_bytes,
                media_type=media_type,
                model_header=f"coqui:{model_name.split('/')[-1]}",
                cache="MISS",
                latency_ms=latency_ms
            )

        except Exception as e:
            logger.error(f"Coqui TTS synthesis failed: {e}")
            raise TTSSynthesisError(f"Synthesis failed: {str(e)}")

    def synthesize_stream_sync(self, req: SynthesisRequest) -> Iterator[bytes]:
        """Streaming synthesis (chunked output)"""
        try:
            # Coqui doesn't support true streaming, so we synthesize and chunk
            result = self.synthesize_sync(req)

            # Yield in chunks for streaming
            chunk_size = 8192
            audio_data = result.audio

            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i + chunk_size]

        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            return

    def _get_model_for_voice(self, voice_id: str) -> str:
        """Map voice ID to Coqui model name"""
        model_name = self._config.voice_models.get(voice_id)
        if model_name:
            return model_name

        # Fallback to default model
        logger.warning(f"Unknown voice_id '{voice_id}', using default model")
        return self._config.model_name

    def _get_tts_instance(self, model_name: str) -> Any:
        """Get or create TTS instance for the given model"""
        if model_name in self._tts_instances:
            return self._tts_instances[model_name]

        try:
            from TTS.api import TTS

            logger.info(f"Loading Coqui model: {model_name}")

            # Create TTS instance
            tts = TTS(
                model_name=model_name,
                progress_bar=False,
                gpu=self._use_gpu
            )

            # Move to specified device if needed
            if hasattr(tts, 'to') and self._device != "cpu":
                tts.to(self._device)

            self._tts_instances[model_name] = tts
            logger.info(f"Model loaded successfully: {model_name}")

            return tts

        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise TTSSynthesisError(f"Model loading failed: {e}")

    def _perform_synthesis(
        self,
        text: str,
        model_name: str,
        sample_rate: int = 22050,
        speed: float = 1.0
    ) -> Any:
        """Perform the actual TTS synthesis"""

        if not text.strip():
            raise TTSSynthesisError("Empty text provided")

        tts = self._get_tts_instance(model_name)

        try:
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # Synthesize to file
            logger.debug(f"Synthesizing: '{text[:50]}...' with model {model_name}")

            # Handle speed parameter (if supported by model)
            synthesis_kwargs = {}
            if hasattr(tts, 'tts_to_file'):
                # Check if model supports speed control
                try:
                    tts.tts_to_file(
                        text=text,
                        file_path=tmp_path,
                        **synthesis_kwargs
                    )
                except TypeError as e:
                    if "speed" in str(e):
                        logger.warning("Model doesn't support speed control")
                    tts.tts_to_file(text=text, file_path=tmp_path)
            else:
                # Fallback method
                audio = tts.tts(text=text)
                import soundfile as sf
                sf.write(tmp_path, audio, sample_rate)

            # Read the generated audio
            import soundfile as sf
            audio_data, sr = sf.read(tmp_path)

            # Clean up temp file
            os.unlink(tmp_path)

            # Resample if needed
            if sr != sample_rate:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)

            # Apply speed adjustment if not handled by model
            if speed != 1.0:
                import librosa
                audio_data = librosa.effects.time_stretch(audio_data, rate=speed)

            return audio_data

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise TTSSynthesisError(f"TTS synthesis error: {e}")

    def _numpy_to_wav(self, audio_data: Any, sample_rate: int) -> bytes:
        """Convert numpy audio data to WAV bytes"""
        try:
            import numpy as np
            import soundfile as sf

            # Ensure audio_data is numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)

            # Normalize to int16 range if needed
            if audio_data.dtype != np.int16:
                if audio_data.max() <= 1.0:
                    # Assume normalized float, convert to int16
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    # Already in int16 range
                    audio_data = audio_data.astype(np.int16)

            # Use the ports helper function
            return pcm16_to_wav(audio_data, sample_rate)

        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise TTSSynthesisError(f"Audio format conversion failed: {e}")

    def list_available_models(self) -> List[str]:
        """List available Coqui TTS models"""
        try:
            from TTS.api import TTS
            return TTS.list_models()
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_voice_info(self) -> List[Dict[str, Any]]:
        """Get information about available voices"""
        voices = []
        for voice_id, model_name in self._config.voice_models.items():
            voices.append({
                "voice_id": voice_id,
                "model_name": model_name,
                "language": "en" if "/en/" in model_name else "unknown",
                "description": f"Coqui TTS model: {model_name}"
            })
        return voices