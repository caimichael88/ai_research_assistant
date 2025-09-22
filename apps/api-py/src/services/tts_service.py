from __future__ import annotations
"""
TTS Service - Service layer for text-to-speech operations

Provides the TTSEngine implementation and dependency injection for controllers.
This module handles:
- Engine initialization and configuration
- Service instance management
- Fallback to fake/mock implementations for testing

The service uses the neural TTS engine by default, which combines:
- FastSpeech2 (Text → Mel spectrogram)
- HiFiGAN (Mel → Audio waveform)
- Optional loudness normalization
"""
import os
import logging
from functools import lru_cache
from typing import Optional

try:
    from ..components.ports import TTSEngine, SynthesisRequest, SynthesisResult, TTSSynthesisError
except ImportError:
    # Fallback for direct execution
    from components.ports import TTSEngine, SynthesisRequest, SynthesisResult, TTSSynthesisError

logger = logging.getLogger(__name__)

# ------------------- Service Implementation -------------------

class FakeTTSEngine(TTSEngine):
    """Fake TTS engine for testing and development without model dependencies"""

    @property
    def id(self) -> str:
        return "fake"

    def warmup(self) -> None:
        logger.info("Fake TTS engine warmup - no-op")

    def synthesize_sync(self, req: SynthesisRequest) -> SynthesisResult:
        """Generate fake audio data for testing"""
        import wave
        import io
        import struct

        # Generate simple sine wave as placeholder audio
        sample_rate = req.sample_rate
        duration = max(0.5, len(req.text) * 0.1)  # Rough duration estimate
        samples = int(sample_rate * duration)

        # Generate sine wave at 440Hz
        import math
        audio_data = []
        for i in range(samples):
            # Proper sine wave generation: sin(2π * frequency * time)
            time = i / sample_rate
            value = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * time))
            audio_data.append(value)

        # Pack into WAV format
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(struct.pack('<' + 'h' * len(audio_data), *audio_data))

        return SynthesisResult(
            audio=buf.getvalue(),
            media_type="audio/wav",
            model_header="fake-tts",
            cache="MISS",
            latency_ms=100
        )

# ------------------- Service Factory -------------------

@lru_cache(maxsize=1)
def _build_neural_tts_engine() -> TTSEngine:
    """Build neural TTS engine with FastSpeech2 + HiFiGAN"""
    try:
        # Import neural components
        try:
            from ..engines.neural import NeuralTTSEngine
            from ..adapters.fastspeech2 import FastSpeech2Adapter
            from ..postfx.loudness import SimpleRMSNormalizer, LUFSDynamicNormalizer
        except ImportError:
            # Fallback for direct execution
            from engines.neural import NeuralTTSEngine
            from adapters.fastspeech2 import FastSpeech2Adapter
            from postfx.loudness import SimpleRMSNormalizer, LUFSDynamicNormalizer

        # Get configuration from environment
        device = os.getenv("DEVICE", "cpu")
        t2m_ckpt = os.getenv("T2M_CKPT", "/models/fastspeech2.onnx")
        voc_ckpt = os.getenv("VOC_CKPT", "/models/hifigan.pt")

        # Additional configuration
        warmup_enabled = os.getenv("WARMUP_ON_STARTUP", "1") == "1"

        logger.info(f"Building neural TTS engine - device: {device}")
        logger.info(f"Text2Mel checkpoint: {t2m_ckpt}")
        logger.info(f"Vocoder checkpoint: {voc_ckpt}")

        # Build Text2Mel adapter (FastSpeech2)
        t2m = FastSpeech2Adapter(ckpt_path=t2m_ckpt, device=device)

        # Build vocoder adapter (HiFiGAN)
        logger.info(f"Building HiFiGAN vocoder - checkpoint: {voc_ckpt}")

        # Import and create HiFiGAN adapter
        try:
            from ..adapters.hifigan import create_hifigan_adapter
        except ImportError:
            from adapters.hifigan import create_hifigan_adapter

        voc = create_hifigan_adapter(ckpt_path=voc_ckpt, device=device)

        # Build loudness normalizer
        loudness: Optional[object] = None
        if os.getenv("LOUDNESS", "1") == "1":
            try:
                import pyloudnorm  # noqa: F401
                target_lufs = float(os.getenv("TARGET_LUFS", "-16"))
                loudness = LUFSDynamicNormalizer(target_lufs=target_lufs)
                logger.info(f"Using LUFS normalization (target: {target_lufs} LUFS)")
            except ImportError:
                target_dbfs = float(os.getenv("TARGET_DBFS", "-20"))
                loudness = SimpleRMSNormalizer(target_dbfs=target_dbfs)
                logger.info(f"Using RMS normalization (target: {target_dbfs} dBFS)")

        # Build neural engine
        engine = NeuralTTSEngine(t2m, voc, loudness=loudness)

        # Warmup (optional based on configuration)
        if warmup_enabled:
            logger.info("Warming up neural TTS engine...")
            engine.warmup()
            logger.info("Neural TTS engine ready")
        else:
            logger.info("Skipping warmup (WARMUP_ON_STARTUP=0)")

        return engine

    except Exception as e:
        logger.error(f"Failed to build neural TTS engine: {e}")
        logger.info("Falling back to fake TTS engine")
        return FakeTTSEngine()

@lru_cache(maxsize=1)
def _build_coqui_tts_engine() -> TTSEngine:
    """Build Coqui TTS engine"""
    try:
        # Import Coqui TTS engine
        try:
            from ..engines.coqui import CoquiTTSEngine
        except ImportError:
            # Fallback for direct execution
            from engines.coqui import CoquiTTSEngine

        # Get configuration from environment
        device = os.getenv("COQUI_DEVICE", os.getenv("DEVICE", "cpu"))
        model_name = os.getenv("COQUI_MODEL", "tts_models/en/ljspeech/tacotron2-DDC_ph")
        warmup_enabled = os.getenv("WARMUP_ON_STARTUP", "1") == "1"

        logger.info(f"Building Coqui TTS engine - device: {device}, model: {model_name}")

        # Build Coqui engine
        engine = CoquiTTSEngine(
            model_name=model_name,
            device=device
        )

        # Warmup (optional based on configuration)
        if warmup_enabled:
            logger.info("Warming up Coqui TTS engine...")
            engine.warmup()
            logger.info("Coqui TTS engine ready")
        else:
            logger.info("Skipping warmup (WARMUP_ON_STARTUP=0)")

        return engine

    except Exception as e:
        logger.error(f"Failed to build Coqui TTS engine: {e}")
        logger.info("Falling back to fake TTS engine")
        return FakeTTSEngine()

@lru_cache(maxsize=1)
def _build_tts_engine() -> TTSEngine:
    """Build TTS engine based on configuration"""

    # Debug environment variables
    fake_tts = os.getenv("FAKE_TTS", "1")  # Default to fake if not set
    engine_type = os.getenv("TTS_ENGINE", "fake")  # Default to fake if not set

    logger.info(f"TTS Engine Configuration - FAKE_TTS: {fake_tts}, TTS_ENGINE: {engine_type}")
    print(f"TTS Engine Configuration - FAKE_TTS: {fake_tts}, TTS_ENGINE: {engine_type}")

    # Check for fake mode
    if fake_tts == "1":
        logger.info("Using fake TTS engine (FAKE_TTS=1)")
        print("Using fake TTS engine (FAKE_TTS=1)")
        return FakeTTSEngine()

    # Check for Coqui TTS mode
    if engine_type == "coqui":
        logger.info("Using Coqui TTS engine")
        print("Using Coqui TTS engine")
        return _build_coqui_tts_engine()

    # Fallback to neural engine (FastSpeech2 + HiFiGAN)
    logger.info("Using neural TTS engine (FastSpeech2 + HiFiGAN)")
    print("Using neural TTS engine (FastSpeech2 + HiFiGAN)")
    return _build_neural_tts_engine()

def get_tts_service() -> TTSEngine:
    """Get TTS service instance for dependency injection"""
    return _build_tts_engine()

# ------------------- Legacy TTSService Class -------------------

class TTSService:
    """
    Legacy service class - kept for backward compatibility

    Orchestrates Tacotron2(Text2Mel) + WaveGlow/HIFI-GAN(vocoder).
    Stateless aside from caches; safe to share as a singleton with internal locks.
    """

    def __init__(self):
        self._engine = get_tts_service()

    def synthesize(self, text: str, voice_id: str = "en_female_1", **kwargs) -> bytes:
        """Legacy synthesis method"""
        req = SynthesisRequest(text=text, voice_id=voice_id, **kwargs)
        result = self._engine.synthesize_sync(req)
        return result.audio

# ------------------- Configuration -------------------

def configure_logging():
    """Configure logging for TTS service"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Environment variable documentation
"""
Configuration via environment variables:

FAKE_TTS=1              : Use fake TTS engine for testing
DEVICE=cpu|cuda|mps     : Computation device
T2M_CKPT=/path/to/model : FastSpeech2 model checkpoint
VOC_CKPT=/path/to/model : HiFiGAN vocoder checkpoint
LOUDNESS=1              : Enable loudness normalization
TARGET_LUFS=-16         : Target LUFS level (requires pyloudnorm)
TARGET_DBFS=-20         : Target dBFS level (RMS normalization)

Example usage:
export DEVICE=cpu
export T2M_CKPT=/models/fastspeech2.onnx
export VOC_CKPT=/models/hifigan.pt
export LOUDNESS=1
export TARGET_DBFS=-20
"""