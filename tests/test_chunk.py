from vault import chunk, config


def test_short_text_is_single_chunk():
    assert chunk.split("a short note") == ["a short note"]


def test_empty_is_no_chunks():
    assert chunk.split("   \n  ") == []


def test_long_text_splits_into_multiple_bounded_chunks():
    body = "\n\n".join(f"Paragraph number {i}. " * 20 for i in range(40))
    chunks = chunk.split(body)
    assert len(chunks) > 1
    assert all(len(c) <= config.CHUNK_TARGET_CHARS + config.CHUNK_OVERLAP_CHARS + 50 for c in chunks)


def test_chunk_count_is_capped():
    body = "\n\n".join("word " * 300 for _ in range(2000))
    assert len(chunk.split(body)) <= config.CHUNK_MAX


def test_oversized_single_sentence_is_hard_cut():
    body = "x" * (config.CHUNK_TARGET_CHARS * 3)
    chunks = chunk.split(body)
    assert len(chunks) >= 3
    # a chunk is at most a carried overlap tail + one target-sized segment + newline
    bound = config.CHUNK_TARGET_CHARS + config.CHUNK_OVERLAP_CHARS + 1
    assert all(len(c) <= bound for c in chunks)
