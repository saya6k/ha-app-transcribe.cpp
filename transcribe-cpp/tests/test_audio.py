"""Unit tests for PCM conversion."""

import numpy as np

from wyoming_transcribe_cpp.audio import pcm16_to_float32


def test_negative_full_scale_maps_to_minus_one():
    out = pcm16_to_float32(b"\x00\x80")  # int16 -32768, little-endian
    assert out.dtype == np.float32
    assert out.shape == (1,)
    assert out[0] == -1.0


def test_positive_full_scale_is_just_below_one():
    out = pcm16_to_float32(b"\xff\x7f")  # int16 32767
    assert 0.9998 < out[0] < 1.0


def test_zero_and_empty():
    assert pcm16_to_float32(b"\x00\x00")[0] == 0.0
    assert pcm16_to_float32(b"").size == 0
