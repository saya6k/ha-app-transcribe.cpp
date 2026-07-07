"""Streaming transcript helpers."""


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
