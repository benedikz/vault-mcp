from vault import consolidate, store


def test_consolidation_sweeps_orphaned_synapses(conn, make_note):
    a = make_note("A", "alpha")
    ghost = "00000000-0000-0000-0000-000000000000"
    with conn:
        conn.execute(
            "INSERT INTO synapses(src, dst, kind, weight) VALUES(?, ?, 'hebbian', 1.0)",
            (a, ghost),
        )
        conn.execute("INSERT INTO node_stats(note_id) VALUES(?)", (ghost,))
    stats = consolidate.consolidate(conn)
    assert stats["orphaned"] >= 1
    left = conn.execute("SELECT COUNT(*) AS c FROM synapses WHERE dst = ?", (ghost,)).fetchone()["c"]
    assert left == 0
    assert conn.execute("SELECT 1 FROM node_stats WHERE note_id = ?", (ghost,)).fetchone() is None


def test_semantic_edges_built_between_similar_notes(conn, make_note):
    a = make_note("Cats one", "cats are small domestic feline animals")
    b = make_note("Cats two", "cats small domestic feline animals pets")
    make_note("Rockets", "orbital rocket propulsion thrust nozzle")
    stats = consolidate.consolidate(conn)
    assert stats["semantic"] >= 1
    edge = conn.execute(
        "SELECT weight FROM synapses WHERE kind = 'semantic' AND src = ? AND dst = ?",
        (a, b),
    ).fetchone()
    assert edge is not None


def test_wikilinks_become_link_edges(conn, make_note):
    target = make_note("Target Note", "the destination body")
    make_note("Source Note", "see [[Target Note]] for details")
    consolidate.consolidate(conn)
    edge = conn.execute(
        "SELECT 1 FROM synapses WHERE kind = 'link' AND dst = ?", (target,)
    ).fetchone()
    assert edge is not None


def test_duplicates_flagged_for_near_identical_notes(conn, make_note):
    make_note("Recipe", "sourdough bread recipe with rye flour and starter")
    make_note("Recipe copy", "sourdough bread recipe with rye flour and starter")
    stats = consolidate.consolidate(conn)
    assert stats["duplicates"] >= 1
    dups = store.list_duplicates(conn)
    assert dups and dups[0]["score"] >= 0.9


def test_salience_normalized_to_unit_peak(conn, make_note):
    a = make_note("Hub", "central topic many references")
    b = make_note("Spoke one", "central topic many references slightly varied")
    store.connect_notes(conn, b, a, "link", 1.0)
    consolidate.consolidate(conn)
    peak = conn.execute("SELECT MAX(salience) AS m FROM node_stats").fetchone()["m"]
    assert peak is None or peak <= 1.0 + 1e-9


def test_stale_note_archived_important_and_recalled_notes_kept(conn, make_note):
    stale = make_note("Stale", "old forgotten fragment")
    vital = make_note("Vital", "old but declared important")
    used = make_note("Used", "old but recently recalled")
    conn.execute("UPDATE notes SET updated_at = datetime('now', '-400 days')")
    conn.execute("UPDATE notes SET importance = 0.9 WHERE id = ?", (vital,))
    conn.execute(
        "UPDATE node_stats SET last_accessed_at = datetime('now') WHERE note_id = ?", (used,)
    )
    conn.commit()
    stats = consolidate.consolidate(conn)
    assert stats["archived"] == 1
    rows = {
        r["id"]: r["archived_at"]
        for r in conn.execute("SELECT id, archived_at FROM notes").fetchall()
    }
    assert rows[stale] is not None
    assert rows[vital] is None
    assert rows[used] is None
    # archiving never touches content, versions, chunks or vectors
    assert store.read_note(conn, stale)["body"] == "old forgotten fragment"


def test_archived_note_leaves_semantic_web(conn, make_note):
    a = make_note("Cats one", "cats are small domestic feline animals")
    b = make_note("Cats two", "cats small domestic feline animals pets")
    make_note("Cats three", "small domestic cats feline pets animals")
    conn.execute("UPDATE notes SET archived_at = datetime('now') WHERE id = ?", (a,))
    conn.commit()
    consolidate.consolidate(conn)
    touching = conn.execute(
        "SELECT COUNT(*) AS c FROM synapses WHERE kind = 'semantic' AND (src = ? OR dst = ?)",
        (a, a),
    ).fetchone()["c"]
    assert touching == 0
    edge = conn.execute(
        "SELECT 1 FROM synapses WHERE kind = 'semantic' AND src = ?", (b,)
    ).fetchone()
    assert edge is not None


def test_cluster_flagged_for_dense_group(conn, make_note):
    a = make_note("Sourdough one", "sourdough bread recipe rye flour starter hydration")
    b = make_note("Sourdough two", "sourdough bread recipe rye flour starter proofing")
    c = make_note("Sourdough three", "sourdough bread recipe rye flour starter baking")
    make_note("Rockets", "orbital rocket propulsion thrust nozzle")
    stats = consolidate.consolidate(conn)
    assert stats["clusters"] == 1
    clusters = store.list_clusters(conn)
    assert len(clusters) == 1
    member_ids = {n["id"] for n in clusters[0]["notes"]}
    assert member_ids == {a, b, c}


def test_cluster_with_gist_hub_not_reflagged(conn, make_note):
    a = make_note("Sourdough one", "sourdough bread recipe rye flour starter hydration")
    b = make_note("Sourdough two", "sourdough bread recipe rye flour starter proofing")
    c = make_note("Sourdough three", "sourdough bread recipe rye flour starter baking")
    make_note("Gist", f"Sourdough overview: [[id:{a}]] [[id:{b}]] [[id:{c}]]")
    stats = consolidate.consolidate(conn)
    assert stats["clusters"] == 0
    assert store.list_clusters(conn) == []


def test_hebbian_decay_and_prune(conn, make_note):
    a = make_note("A", "alpha")
    b = make_note("B", "beta")
    store.reinforce(conn, [a, b])
    # force the edge old and tiny so consolidation prunes it
    conn.execute(
        "UPDATE synapses SET weight = 0.01, last_fired_at = datetime('now', '-100 days') "
        "WHERE kind = 'hebbian'"
    )
    conn.commit()
    stats = consolidate.consolidate(conn)
    assert stats["pruned"] >= 1
    remaining = conn.execute("SELECT COUNT(*) AS c FROM synapses WHERE kind = 'hebbian'").fetchone()["c"]
    assert remaining == 0
