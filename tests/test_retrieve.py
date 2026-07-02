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


def test_activation_does_not_route_through_trashed_note(conn, make_note):
    from vault import store

    a = make_note("Coffee", "coffee brewing methods")
    b = make_note("Middle", "middle hop qwerty")
    c = make_note("Target", "target plugh xyzzy")
    store.connect_notes(conn, a, b, "link", 1.0)
    store.connect_notes(conn, b, c, "link", 1.0)
    store.trash_note(conn, b)
    activation = retrieve.spread_activation(conn, {a: 1.0}, depth=2)
    assert b not in activation
    assert c not in activation


def test_neighbors_includes_incoming_edges(conn, make_note):
    from vault import store

    src = make_note("Src", "source note body")
    dst = make_note("Dst", "destination note body")
    store.connect_notes(conn, src, dst, "link", 1.0)
    assert any(r["id"] == dst for r in retrieve.neighbors(conn, src))
    assert any(r["id"] == src for r in retrieve.neighbors(conn, dst))


def test_recency_prefers_newer_among_equal(conn, embedder, make_note):
    older = make_note("Note one", "identical body content alpha")
    newer = make_note("Note two", "identical body content alpha")
    # backdate `older`
    conn.execute("UPDATE notes SET updated_at = datetime('now', '-2000 days') WHERE id = ?", (older,))
    conn.commit()
    scores = {older: 1.0, newer: 1.0}
    retrieve.apply_recency(conn, scores)
    assert scores[newer] > scores[older]


def test_practice_boosts_recalled_note(conn, make_note):
    practiced = make_note("One", "identical body content alpha")
    fresh = make_note("Two", "identical body content alpha")
    conn.execute(
        "UPDATE node_stats SET access_count = 10, last_accessed_at = datetime('now') "
        "WHERE note_id = ?",
        (practiced,),
    )
    conn.commit()
    scores = {practiced: 1.0, fresh: 1.0}
    retrieve.apply_strength(conn, scores)
    assert scores[practiced] > scores[fresh]


def test_practice_boost_fades_when_stale(conn, make_note):
    recent = make_note("One", "identical body content alpha")
    stale = make_note("Two", "identical body content alpha")
    conn.execute(
        "UPDATE node_stats SET access_count = 10, last_accessed_at = datetime('now') "
        "WHERE note_id = ?",
        (recent,),
    )
    conn.execute(
        "UPDATE node_stats SET access_count = 10, "
        "last_accessed_at = datetime('now', '-1000 days') WHERE note_id = ?",
        (stale,),
    )
    conn.commit()
    scores = {recent: 1.0, stale: 1.0}
    retrieve.apply_strength(conn, scores)
    assert scores[recent] > scores[stale]


def test_importance_boosts_ranking(conn, make_note):
    vital = make_note("One", "identical body content alpha")
    plain = make_note("Two", "identical body content alpha")
    conn.execute("UPDATE notes SET importance = 1.0 WHERE id = ?", (vital,))
    conn.commit()
    scores = {vital: 1.0, plain: 1.0}
    retrieve.apply_strength(conn, scores)
    assert scores[vital] > scores[plain]


def test_archived_note_demoted_but_still_searchable(conn, embedder, make_note):
    archived = make_note("Buried", "quantum widget calibration procedure")
    live = make_note("Active", "quantum widget calibration procedure")
    conn.execute("UPDATE notes SET archived_at = datetime('now') WHERE id = ?", (archived,))
    conn.commit()
    vec = embedder.embed_query("quantum widget calibration")
    results = retrieve.search(conn, vec, "quantum widget calibration", 5)
    ids = [c.id for c in results]
    assert archived in ids and live in ids     # never hidden
    assert ids.index(live) < ids.index(archived)  # but demoted
    flags = {c.id: c.archived for c in results}
    assert flags[archived] is True and flags[live] is False


def test_nearest_notes_finds_similar_and_excludes_self(conn, embedder, make_note):
    espresso = make_note("Espresso", "espresso machines grinders portafilter")
    make_note("Gardening", "tomato seedlings and compost")
    vec = embedder.embed_query("espresso machines grinders")
    hits = retrieve.nearest_notes(conn, vec, 3)
    assert hits and hits[0]["id"] == espresso
    assert 0.0 < hits[0]["similarity"] <= 1.0
    excluded = retrieve.nearest_notes(conn, vec, 3, exclude=espresso)
    assert all(h["id"] != espresso for h in excluded)
