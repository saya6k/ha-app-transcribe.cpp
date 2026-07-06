"""Unit tests for diarization tag rendering (pure logic, no sherpa-onnx)."""

from wyoming_transcribe_cpp.diarize import DiarSegment, render_tagged


def seg(start, end, speaker):
    return DiarSegment(start=start, end=end, speaker=speaker)


class TestRenderTagged:
    def test_two_speakers_get_inline_tags(self):
        asr = [(0.0, 2.0, "hello there"), (2.5, 4.0, "hi how are you")]
        diar = [seg(0.0, 2.2, 0), seg(2.2, 4.0, 1)]
        assert render_tagged(asr, diar) == (
            "[Speaker 1] hello there [Speaker 2] hi how are you"
        )

    def test_consecutive_segments_of_same_speaker_share_one_tag(self):
        asr = [(0.0, 1.0, "one"), (1.0, 2.0, "two"), (2.5, 3.5, "three")]
        diar = [seg(0.0, 2.0, 0), seg(2.0, 3.5, 1)]
        assert render_tagged(asr, diar) == "[Speaker 1] one two [Speaker 2] three"

    def test_speaker_is_picked_by_max_overlap(self):
        # ASR segment 0.0-3.0 overlaps speaker 0 for 1s and speaker 1 for 2s.
        asr = [(0.0, 3.0, "mostly second speaker")]
        diar = [seg(0.0, 1.0, 0), seg(1.0, 3.0, 1)]
        assert render_tagged(asr, diar) == "[Speaker 2] mostly second speaker"

    def test_no_diarization_segments_returns_plain_text(self):
        asr = [(0.0, 1.0, "hello"), (1.0, 2.0, "world")]
        assert render_tagged(asr, []) == "hello world"

    def test_single_speaker_still_tagged(self):
        asr = [(0.0, 2.0, "just me")]
        diar = [seg(0.0, 2.0, 0)]
        assert render_tagged(asr, diar) == "[Speaker 1] just me"

    def test_asr_without_timestamps_uses_dominant_speaker(self):
        # Some models emit no per-segment timestamps: one segment, zero span.
        asr = [(0.0, 0.0, "all the text")]
        diar = [seg(0.0, 1.0, 1), seg(1.0, 4.0, 0)]
        assert render_tagged(asr, diar) == "[Speaker 1] all the text"
