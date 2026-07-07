"""Unit tests for EnhanceStream hop buffering (stub ONNX session)."""

import numpy as np

from wyoming_transcribe_cpp.enhance import _N_FFT, EnhanceStream

HOP = 256


class _Input:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _EchoSession:
    """Identity denoiser: wav_out == wav_in, one cache tensor incremented."""

    def get_inputs(self):
        return [_Input("wav_in", [1, HOP]), _Input("cache_in_0", [1, 4])]

    def run(self, _outputs, feeds):
        assert feeds["wav_in"].shape == (1, HOP)
        return [feeds["wav_in"].copy(), feeds["cache_in_0"] + 1]


def test_process_returns_only_full_hops():
    stream = EnhanceStream(_EchoSession())
    out = stream.process(np.arange(300, dtype=np.float32))
    assert out.size == HOP


def test_remainder_is_carried_into_the_next_chunk():
    stream = EnhanceStream(_EchoSession())
    first = stream.process(np.arange(300, dtype=np.float32))
    second = stream.process(np.arange(300, 512, dtype=np.float32))
    assert np.array_equal(
        np.concatenate([first, second]), np.arange(512, dtype=np.float32)
    )


def test_short_chunk_yields_nothing_until_a_hop_fills():
    stream = EnhanceStream(_EchoSession())
    assert stream.process(np.ones(10, dtype=np.float32)).size == 0


def test_flush_drains_the_buffer_and_model_latency():
    stream = EnhanceStream(_EchoSession())
    stream.process(np.ones(10, dtype=np.float32))
    tail = stream.flush()
    # 10 buffered + _N_FFT zeros, truncated to whole hops.
    expected = (10 + _N_FFT) // HOP * HOP
    assert tail.size == expected
    assert np.array_equal(tail[:10], np.ones(10, dtype=np.float32))


def test_cache_state_is_threaded_between_hops():
    stream = EnhanceStream(_EchoSession())
    stream.process(np.zeros(3 * HOP, dtype=np.float32))
    assert np.array_equal(
        stream._state["cache_in_0"], np.full((1, 4), 3, dtype=np.float32)
    )
