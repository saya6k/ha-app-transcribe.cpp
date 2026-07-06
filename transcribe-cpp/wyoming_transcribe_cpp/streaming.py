"""Streaming transcript helpers."""


def decode_mode(
    supports_streaming: bool, *, enhancement: bool, diarization: bool
) -> str:
    """'stream' or 'batch' for this session.

    Enhancement (GTCRN, whole-utterance denoise) and diarization (needs the
    final Result's segment timestamps) both require the full buffered
    utterance, so either one forces batch even on streaming-capable models.
    """
    if supports_streaming and not enhancement and not diarization:
        return "stream"
    return "batch"


def text_delta(previous: str, current: str) -> str:
    """Delta between two hypothesis snapshots for TranscriptChunk events.

    TranscriptChunk.text is a delta: emit only the new tail while the
    hypothesis grows. If the decoder revised an earlier part (rare), resend
    the whole current text — best effort; the final Transcript corrects it.
    """
    if current == previous:
        return ""
    if current.startswith(previous):
        return current[len(previous):]
    return current
