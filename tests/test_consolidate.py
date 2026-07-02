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
