"""Unit tests for streaming transcript delta computation and decode mode."""

from wyoming_transcribe_cpp.streaming import decode_mode, text_delta


class TestDecodeMode:
    def test_streaming_model_streams_by_default(self):
        assert decode_mode(True, enhancement=False, diarization=False) == "stream"

    def test_batch_model_always_batches(self):
        assert decode_mode(False, enhancement=False, diarization=False) == "batch"

    def test_enhancement_forces_batch(self):
        # GTCRN denoises the whole utterance — incompatible with partials.
        assert decode_mode(True, enhancement=True, diarization=False) == "batch"

    def test_diarization_forces_batch(self):
        # Tag rendering needs the final Result segments (timestamps).
        assert decode_mode(True, enhancement=False, diarization=True) == "batch"


def test_growing_hypothesis_emits_only_the_new_tail():
    assert text_delta("hello", "hello world") == " world"


def test_first_partial_is_emitted_whole():
    assert text_delta("", "hello") == "hello"


def test_revised_hypothesis_is_re_emitted_whole():
    # The decoder revised its committed prefix — resend the full text.
    assert text_delta("helo wor", "hello world") == "hello world"


def test_unchanged_hypothesis_emits_nothing():
    assert text_delta("hello", "hello") == ""
