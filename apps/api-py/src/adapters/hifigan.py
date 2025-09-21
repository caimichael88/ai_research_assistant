from __future__ import annotations
"""
HiFiGAN Vocoder Adapter

Implements the NeuralVocoder interface for HiFiGAN models.
Converts mel spectrograms to audio waveforms using ONNX Runtime.

Features:
- ONNX Runtime inference
- GPU/CPU/MPS device support
- Configurable sample rates
- Memory-efficient processing
- Streaming support via chunking

Usage:
    vocoder = HiFiGANAdapter(
        ckpt_path="/models/hifigan.onnx",
        device="cpu"
    )

    audio = vocoder.infer_audio(mel_spectrogram, sample_rate=22050)
"""
import os
import logging
from typing import Any, List, Optional, Union
from dataclasses import dataclass
import numpy as np

try:
    from ..components.ports import NeuralVocoder, TTSSynthesisError
except ImportError:
    # Fallback for direct execution
    from components.ports import NeuralVocoder, TTSSynthesisError

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class HiFiGANConfig:
    """Configuration for HiFiGAN vocoder"""
    ckpt_path: str
    device: str = "cpu"
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    normalize: bool = True
    # ONNX Runtime settings
    intra_op_num_threads: int = 1
    inter_op_num_threads: int = 1

class HiFiGANAdapter(NeuralVocoder):
    """HiFiGAN vocoder implementation using ONNX Runtime"""

    def __init__(self, ckpt_path: str, device: str = "cpu", config: Optional[HiFiGANConfig] = None):
        """
        Initialize HiFiGAN vocoder

        Args:
            ckpt_path: Path to ONNX model file
            device: Device for inference ("cpu", "cuda", "mps")
            config: Optional configuration object
        """
        if config is None:
            config = HiFiGANConfig(ckpt_path=ckpt_path, device=device)

        self._config = config
        self._session: Optional[Any] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None

        # Validate model file exists
        if not os.path.exists(ckpt_path):
            raise TTSSynthesisError(f"HiFiGAN model not found: {ckpt_path}")

        logger.info(f"Initializing HiFiGAN vocoder - device: {device}, model: {ckpt_path}")
        self._load_model()

    @property
    def id(self) -> str:
        return "hifigan"

    def warmup(self) -> None:
        """Warmup the vocoder with dummy input"""
        logger.info("Warming up HiFiGAN vocoder...")
        try:
            # Create dummy mel spectrogram (80 mel bins x 100 time frames)
            dummy_mel = np.random.randn(1, 80, 100).astype(np.float32)

            # Run inference
            self._infer_onnx(dummy_mel)

            logger.info("HiFiGAN vocoder warmed up successfully")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def infer_audio(self, mel: Union[np.ndarray, Any], *, sample_rate: int = 22050) -> bytes:
        """
        Convert mel spectrogram to audio waveform

        Args:
            mel: Mel spectrogram array (shape: [batch, mel_bins, time])
            sample_rate: Target sample rate

        Returns:
            WAV audio bytes
        """
        try:
            # Ensure numpy array
            if not isinstance(mel, np.ndarray):
                mel = np.array(mel)

            # Ensure correct shape [batch, mel_bins, time]
            if mel.ndim == 2:
                mel = mel[np.newaxis, ...]  # Add batch dimension

            # Ensure float32 for ONNX
            if mel.dtype != np.float32:
                mel = mel.astype(np.float32)

            logger.debug(f"HiFiGAN input mel shape: {mel.shape}")

            # Run ONNX inference
            audio = self._infer_onnx(mel)

            # Post-process audio
            audio = self._postprocess_audio(audio, sample_rate)

            # Convert to WAV bytes
            return self._audio_to_wav(audio, sample_rate)

        except Exception as e:
            logger.error(f"HiFiGAN inference failed: {e}")
            raise TTSSynthesisError(f"Vocoder inference failed: {e}")

    def _load_model(self) -> None:
        """Load ONNX model with appropriate providers"""
        try:
            import onnxruntime as ort

            # Configure execution providers based on device
            providers = self._get_execution_providers()

            # Session options for performance
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self._config.intra_op_num_threads
            sess_options.inter_op_num_threads = self._config.inter_op_num_threads
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Create inference session
            self._session = ort.InferenceSession(
                self._config.ckpt_path,
                sess_options=sess_options,
                providers=providers
            )

            # Get input/output names
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name

            logger.info(f"HiFiGAN model loaded - input: {self._input_name}, output: {self._output_name}")
            logger.info(f"Using providers: {self._session.get_providers()}")

        except ImportError:
            raise TTSSynthesisError("onnxruntime not installed. Install with: pip install onnxruntime")
        except Exception as e:
            raise TTSSynthesisError(f"Failed to load HiFiGAN model: {e}")

    def _get_execution_providers(self) -> List[Union[str, tuple]]:
        """Get ONNX execution providers based on device"""
        if self._config.device == "cpu":
            return ["CPUExecutionProvider"]
        elif self._config.device == "cuda":
            return [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }),
                "CPUExecutionProvider"
            ]
        elif self._config.device == "mps":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            logger.warning(f"Unknown device '{self._config.device}', using CPU")
            return ["CPUExecutionProvider"]

    def _infer_onnx(self, mel: np.ndarray) -> np.ndarray:
        """Run ONNX inference"""
        if self._session is None:
            raise TTSSynthesisError("Model not loaded")

        try:
            # Run inference
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: mel}
            )

            audio = outputs[0]
            logger.debug(f"HiFiGAN output audio shape: {audio.shape}")

            return audio

        except Exception as e:
            raise TTSSynthesisError(f"ONNX inference failed: {e}")

    def _postprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Post-process generated audio"""
        # Remove batch dimension if present
        if audio.ndim > 1:
            audio = audio.squeeze()

        # Normalize audio if configured
        if self._config.normalize:
            # Normalize to [-1, 1] range
            audio_max = np.abs(audio).max()
            if audio_max > 0:
                audio = audio / audio_max

        # Clip to prevent clipping artifacts
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def _audio_to_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio array to WAV bytes"""
        try:
            # Import wave functions from ports
            try:
                from ..components.ports import pcm16_to_wav
            except ImportError:
                from components.ports import pcm16_to_wav

            # Convert to int16
            if audio.dtype != np.int16:
                # Scale to int16 range
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio

            return pcm16_to_wav(audio_int16, sample_rate)

        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise TTSSynthesisError(f"WAV conversion failed: {e}")

# Factory function for dependency injection
def create_hifigan_adapter(
    ckpt_path: Optional[str] = None,
    device: Optional[str] = None
) -> HiFiGANAdapter:
    """
    Create HiFiGAN adapter with environment configuration

    Args:
        ckpt_path: Override path to ONNX model
        device: Override device setting

    Returns:
        Configured HiFiGAN adapter
    """
    # Get configuration from environment
    ckpt_path = ckpt_path or os.getenv("VOC_CKPT", "/models/hifigan.onnx")
    device = device or os.getenv("DEVICE", "cpu")

    # Additional config from environment
    config = HiFiGANConfig(
        ckpt_path=ckpt_path,
        device=device,
        sample_rate=int(os.getenv("DEFAULT_SAMPLE_RATE", "22050")),
        normalize=os.getenv("VOC_NORMALIZE", "1") == "1",
        intra_op_num_threads=int(os.getenv("ONNX_INTRA_THREADS", "1")),
        inter_op_num_threads=int(os.getenv("ONNX_INTER_THREADS", "1"))
    )

    return HiFiGANAdapter(ckpt_path, device, config)