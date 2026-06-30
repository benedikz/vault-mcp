"""Split a note body into overlapping passages for embedding.

The embedding model only reads ~512 tokens, so a long note embedded whole would
have most of its text silently truncated away from dense retrieval. We split on
natural boundaries (blank lines, then sentences) and pack into ~target-sized
passages with a little overlap so context that straddles a boundary is not lost.
Keyword search still indexes the full body, so this only affects the dense half.
"""

import re

from vault import config

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _segments(text: str) -> list[str]:
    out: list[str] = []
    for para in _PARA.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= config.CHUNK_TARGET_CHARS:
            out.append(para)
            continue
        # Paragraph alone exceeds the target — fall back to sentence splitting,
        # and hard-cut any single sentence that is still too long.
        for sent in _SENT.split(para):
            sent = sent.strip()
            if not sent:
                continue
            while len(sent) > config.CHUNK_TARGET_CHARS:
                out.append(sent[: config.CHUNK_TARGET_CHARS])
                sent = sent[config.CHUNK_TARGET_CHARS :]
            if sent:
                out.append(sent)
    return out


def split(
    text: str,
    target: int = config.CHUNK_TARGET_CHARS,
    overlap: int = config.CHUNK_OVERLAP_CHARS,
    max_chunks: int = config.CHUNK_MAX,
) -> list[str]:
    """Return a bounded list of passage strings covering ``text``."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= target:
        return [text]

    segments = _segments(text)
    chunks: list[str] = []
    buf = ""
    for seg in segments:
        if buf and len(buf) + 1 + len(seg) > target:
            chunks.append(buf)
            # Carry a tail of the previous chunk so cross-boundary context survives.
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + "\n" + seg).strip() if tail else seg
        else:
            buf = (buf + "\n" + seg).strip() if buf else seg
    if buf:
        chunks.append(buf)

    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
    return chunks
