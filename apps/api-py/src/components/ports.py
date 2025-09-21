"""
Ports / interfaces for the TTS pipeline — aligned with our discussion:

- **TTSEngine**: top-level service interface the router depends on (single entrypoint).
- **NeuralTextToMel**: the *Tacotron2-style* text→mel component we specifically mean
  when we say "Text2Mel" in this project.
- **NeuralVocoder**: the mel→waveform component (HiFi-GAN/WaveGlow).
- **SystemTTSEngine**: optional interface for OS/system TTS (e.g., pyttsx3) so we
  can swap engines behind the same `TTSEngine` contract.

This file deliberately avoids importing heavy libs (torch, numpy) to keep import-time
light and prevent accidental GPU initialization from routers.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

# ----------------- Basic types -----------------
MelTensor = Any        # keep loose here (torch.Tensor / np.ndarray in adapters)
AudioBytes = bytes

class AudioFormat(str, Enum):
    wav = "wav"
    mp3 = "mp3"
    ogg = "ogg"

@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    voice_id: str = "en_female_1"
    sample_rate: int = 22050
    speed: float = 1.0
    pitch_semitones: int = 0
    fmt: AudioFormat = AudioFormat.wav
    engine_id: str = "neural"         # "neural" | "system" (pyttsx3) | others
    vocoder_id: str = "hifigan"       # for neural engine
    normalize_loudness: bool = True

@dataclass(frozen=True)
class MelSpectrogram:
    data: MelTensor
    sample_rate: int
    hop_length: int
    n_mels: int
    alignments: Optional[Any] = None

@dataclass(frozen=True)
class SynthesisResult:
    audio: AudioBytes
    media_type: str            # e.g., "audio/wav"
    model_header: str          # e.g., "tacotron2+hifigan" or "system:com.apple.speech"
    cache: str = "MISS"        # "HIT" | "MISS"
    latency_ms: Optional[int] = None

# ----------------- Errors -----------------
class TTSInitError(RuntimeError):
    pass

class TTSSynthesisError(RuntimeError):
    pass

# ----------------- Interfaces (Ports) -----------------
@runtime_checkable
class TextNormalizer(Protocol):
    """Optional preprocessor before synthesis (number expansion, punctuation, SSML subset)."""
    def normalize(self, text: str) -> str: ...

@runtime_checkable
class NeuralTextToMel(Protocol):
    """**This is the Tacotron2-style Text→Mel** component we refer to as *Text2Mel*.
    Implementations: Tacotron2, FastSpeech2 (if used as text→mel), etc.
    """
    @property
    def id(self) -> str: ...                # e.g., "tacotron2"

    @property
    def voice_id(self) -> str: ...          # e.g., "en_female_1"

    def warmup(self) -> None: ...           # optional: prime kernels/caches

    def infer_mel(self, text: str, *, speed: float = 1.0) -> MelSpectrogram: ...

@runtime_checkable
class NeuralVocoder(Protocol):
    """Mel→Waveform vocoder. Implementations: HiFi-GAN, WaveGlow."""
    @property
    def id(self) -> str: ...                # e.g., "hifigan" | "waveglow"

    def warmup(self) -> None: ...

    def infer_audio(self, mel: MelSpectrogram, *, sample_rate: int = 22050) -> AudioBytes: ...

@runtime_checkable
class LoudnessNormalizer(Protocol):
    """Post-processing loudness normalization for TTS output."""
    def normalize(self, audio: AudioBytes, sample_rate: int) -> AudioBytes: ...

@runtime_checkable
class TTSEngine(Protocol):
    """Top-level TTS engine the API depends on (single entry point)."""
    @property
    def id(self) -> str: ...                # e.g., "neural" | "system"

    def warmup(self) -> None: ...

    def synthesize_sync(self, req: SynthesisRequest) -> SynthesisResult: ...

@runtime_checkable
class SystemTTSEngine(TTSEngine, Protocol):
    """Optional system/OS TTS engine (e.g., pyttsx3 wrapper)."""
    ...

# ----------------- Minimal helpers (WAV packing) -----------------
# Kept here so adapters can share them without new deps.
import io, wave, struct

def pcm16_to_wav(samples: bytes | memoryview | list[int] | "np.ndarray", sample_rate: int) -> bytes:
    """Pack PCM16 mono samples into a WAV container using stdlib only."""
    if isinstance(samples, (bytes, bytearray, memoryview)):
        pcm_bytes = bytes(samples)
    else:
        try:
            import numpy as np  # type: ignore
            if isinstance(samples, np.ndarray):
                if samples.dtype != np.int16:
                    samples = samples.astype(np.int16, copy=False)
                pcm_bytes = samples.tobytes()
            else:
                pcm_bytes = struct.pack("<" + "h" * len(samples), *samples)  # type: ignore[arg-type]
        except ModuleNotFoundError:
            pcm_bytes = struct.pack("<" + "h" * len(samples), *samples)      # type: ignore[arg-type]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()
