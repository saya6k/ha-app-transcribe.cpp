"""Unit tests for streaming transcript delta computation."""

from wyoming_transcribe_cpp.streaming import text_delta


def test_growing_hypothesis_emits_only_the_new_tail():
    assert text_delta("hello", "hello world") == " world"


def test_first_partial_is_emitted_whole():
    assert text_delta("", "hello") == "hello"


def test_revised_hypothesis_is_re_emitted_whole():
    # The decoder revised its committed prefix — resend the full text.
    assert text_delta("helo wor", "hello world") == "hello world"


def test_unchanged_hypothesis_emits_nothing():
    assert text_delta("hello", "hello") == ""
