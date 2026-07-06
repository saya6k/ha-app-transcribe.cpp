"""PCM helpers."""

import numpy as np


def pcm16_to_float32(audio: bytes) -> np.ndarray:
    """Convert little-endian 16-bit PCM bytes to float32 in [-1, 1]."""
    return np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
