from vault import retrieve


def test_search_finds_relevant_note(conn, embedder, make_note):
    target = make_note("Espresso", "notes about espresso machines and grinders")
    make_note("Gardening", "tomato seedlings and compost in spring")
    vec = embedder.embed_query("espresso grinder")
    results = retrieve.search(conn, vec, "espresso grinder", 5)
    assert results
    assert results[0].id == target


def test_dense_knn_collapses_passages_to_notes(conn, embedder, make_note):
    body = "\n\n".join(f"Espresso paragraph {i} about pressure and grind. " * 15 for i in range(20))
    nid = make_note("Big espresso note", body)
    vec = embedder.embed_query("espresso pressure grind")
    ids = retrieve.dense_knn(conn, vec, 10)
    # the note appears once despite contributing many passages
    assert ids.count(nid) <= 1
    assert nid in ids


def test_keyword_bm25_matches_full_body_not_just_first_chunk(conn, make_note):
    # a distinctive term late in a long body must still be keyword-findable
    long_body = ("filler sentence. " * 400) + " zarquon-marker-term appears here."
    nid = make_note("Long", long_body)
    ids = retrieve.keyword_bm25(conn, "zarquon-marker-term", 5)
    assert nid in ids


def test_recall_spreads_over_synapses(conn, embedder, make_note):
    from vault import store

    seed = make_note("Coffee", "coffee brewing methods and ratios")
    linked = make_note("Water", "water chemistry totally unrelated words xyzzy")
    # explicit strong synapse so spreading activation reaches `linked`
    store.connect_notes(conn, seed, linked, "link", 1.0)
    vec = embedder.embed_query("coffee brewing")
    found = {c.id for c in retrieve.recall(conn, vec, "coffee brewing", depth=2, k=10)}
    assert seed in found
    assert linked in found


def test_recency_prefers_newer_among_equal(conn, embedder, make_note):
    older = make_note("Note one", "identical body content alpha")
    newer = make_note("Note two", "identical body content alpha")
    # backdate `older`
    conn.execute("UPDATE notes SET updated_at = datetime('now', '-2000 days') WHERE id = ?", (older,))
    conn.commit()
    scores = {older: 1.0, newer: 1.0}
    retrieve.apply_recency(conn, scores)
    assert scores[newer] > scores[older]
