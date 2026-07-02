import pytest

from vault import store
from vault.errors import NotFoundError, ValidationError


def _chunk_count(conn, note_id):
    return conn.execute("SELECT COUNT(*) AS c FROM chunks WHERE note_id = ?", (note_id,)).fetchone()["c"]


def _vec_count(conn):
    return conn.execute("SELECT COUNT(*) AS c FROM notes_vec").fetchone()["c"]


def test_create_indexes_chunks_and_vectors(conn, make_note):
    nid = make_note("Title", "hello world body")
    assert _chunk_count(conn, nid) == 1
    assert _vec_count(conn) == 1
    note = store.read_note(conn, nid)
    assert note["title"] == "Title"
    assert note["version"] == 1


def test_long_note_produces_many_chunks_and_vectors(conn, make_note):
    body = "\n\n".join(f"Section {i} discusses an important topic. " * 20 for i in range(30))
    nid = make_note("Long", body)
    n = _chunk_count(conn, nid)
    assert n > 1
    assert _vec_count(conn) == n


def test_update_snapshots_version_and_rebuilds_chunks(conn, make_note, embedder):
    from vault import server

    nid = make_note("Orig", "first body")
    before = _vec_count(conn)
    new_body = "a completely different and much longer body " * 50
    chunks, vectors = server._embed_note(embedder, "Orig", new_body)
    store.update_note(conn, nid, None, new_body, None, chunks, vectors)
    note = store.read_note(conn, nid)
    assert note["version"] == 2
    versions = conn.execute(
        "SELECT COUNT(*) AS c FROM note_versions WHERE note_id = ?", (nid,)
    ).fetchone()["c"]
    assert versions == 1
    assert _vec_count(conn) > before


def test_tag_only_update_keeps_vectors(conn, make_note):
    nid = make_note("T", "body text here")
    before = _vec_count(conn)
    store.update_note(conn, nid, None, None, ["x"], None, None)
    assert _vec_count(conn) == before
    assert store.read_note(conn, nid)["tags"] == ["x"]


def test_trash_removes_chunks_and_hides_note(conn, make_note):
    nid = make_note("Gone", "to be trashed")
    store.trash_note(conn, nid)
    assert _chunk_count(conn, nid) == 0
    assert _vec_count(conn) == 0
    with pytest.raises(NotFoundError):
        store.read_note(conn, nid)


def test_reinforce_creates_symmetric_hebbian_edges(conn, make_note):
    a = make_note("A", "alpha")
    b = make_note("B", "beta")
    store.reinforce(conn, [a, b])
    rows = conn.execute(
        "SELECT src, dst, weight FROM synapses WHERE kind = 'hebbian'"
    ).fetchall()
    assert len(rows) == 2
    assert all(r["weight"] > 0 for r in rows)


def test_connect_rejects_unknown_note(conn, make_note):
    a = make_note("A", "alpha")
    with pytest.raises(NotFoundError):
        store.connect_notes(conn, a, "00000000-0000-0000-0000-000000000000", "link", 1.0)


def test_trash_removes_synapses_and_node_stats(conn, make_note):
    a = make_note("A", "alpha")
    b = make_note("B", "beta")
    store.connect_notes(conn, a, b, "link", 1.0)
    store.reinforce(conn, [a, b])
    store.trash_note(conn, b)
    touching = conn.execute(
        "SELECT COUNT(*) AS c FROM synapses WHERE src = ? OR dst = ?", (b, b)
    ).fetchone()["c"]
    assert touching == 0
    assert conn.execute("SELECT 1 FROM node_stats WHERE note_id = ?", (b,)).fetchone() is None


def test_connect_rejects_self_loop(conn, make_note):
    a = make_note("A", "alpha")
    with pytest.raises(ValidationError):
        store.connect_notes(conn, a, a, "link", 1.0)


def test_reinforce_duplicate_ids_create_no_self_loop(conn, make_note):
    a = make_note("A", "alpha")
    b = make_note("B", "beta")
    store.reinforce(conn, [a, a, b])
    self_loops = conn.execute("SELECT COUNT(*) AS c FROM synapses WHERE src = dst").fetchone()["c"]
    assert self_loops == 0
    edges = conn.execute("SELECT COUNT(*) AS c FROM synapses WHERE kind = 'hebbian'").fetchone()["c"]
    assert edges == 2


def test_importance_stored_and_updatable(conn, embedder):
    from vault import server

    chunks, vectors = server._embed_note(embedder, "Key fact", "must not fade")
    nid = store.create_note(conn, "Key fact", "must not fade", [], chunks, vectors, importance=0.9)
    assert store.read_note(conn, nid)["importance"] == 0.9
    store.update_note(conn, nid, None, None, None, None, None, importance=0.2)
    assert store.read_note(conn, nid)["importance"] == 0.2
    # an update that does not mention importance keeps it
    store.update_note(conn, nid, None, None, ["x"], None, None)
    assert store.read_note(conn, nid)["importance"] == 0.2


def test_temporal_edges_bind_notes_created_together(conn, make_note):
    a = make_note("First", "written in the same session")
    b = make_note("Second", "written moments later")
    bound = store.add_temporal_edges(conn, b)
    assert a in bound
    edges = conn.execute(
        "SELECT src, dst FROM synapses WHERE kind = 'temporal'"
    ).fetchall()
    directed = {(r["src"], r["dst"]) for r in edges}
    assert (b, a) in directed and (a, b) in directed


def test_temporal_edges_skip_old_notes(conn, make_note):
    old = make_note("Old", "from another day")
    conn.execute(
        "UPDATE notes SET created_at = datetime('now', '-1 day') WHERE id = ?", (old,)
    )
    conn.commit()
    fresh = make_note("Fresh", "today's thought")
    assert store.add_temporal_edges(conn, fresh) == []


def test_add_semantic_edges_creates_provisional_bidirectional_edges(conn, make_note):
    a = make_note("A", "espresso machines")
    b = make_note("B", "espresso grinders")
    store.add_semantic_edges(conn, b, [{"id": a, "title": "A", "similarity": 0.8}])
    directed = {
        (r["src"], r["dst"])
        for r in conn.execute("SELECT src, dst FROM synapses WHERE kind = 'semantic'").fetchall()
    }
    assert (a, b) in directed and (b, a) in directed
    # below the floor nothing is created
    c = make_note("C", "totally unrelated")
    store.add_semantic_edges(conn, c, [{"id": a, "title": "A", "similarity": 0.1}])
    touching_c = conn.execute(
        "SELECT COUNT(*) AS c FROM synapses WHERE src = ? OR dst = ?", (c, c)
    ).fetchone()["c"]
    assert touching_c == 0


def test_read_note_revives_archived(conn, make_note):
    nid = make_note("Dormant", "an archived memory")
    conn.execute("UPDATE notes SET archived_at = datetime('now') WHERE id = ?", (nid,))
    conn.commit()
    store.read_note(conn, nid)
    row = conn.execute("SELECT archived_at FROM notes WHERE id = ?", (nid,)).fetchone()
    assert row["archived_at"] is None
    revived = conn.execute(
        "SELECT COUNT(*) AS c FROM audit_log WHERE action = 'revive' AND note_id = ?", (nid,)
    ).fetchone()["c"]
    assert revived == 1


def test_touch_access_revives_archived(conn, make_note):
    nid = make_note("Dormant", "an archived memory")
    conn.execute("UPDATE notes SET archived_at = datetime('now') WHERE id = ?", (nid,))
    conn.commit()
    store.touch_access(conn, [nid])
    row = conn.execute("SELECT archived_at FROM notes WHERE id = ?", (nid,)).fetchone()
    assert row["archived_at"] is None


def test_update_note_revives_archived(conn, make_note):
    nid = make_note("Dormant", "an archived memory")
    conn.execute("UPDATE notes SET archived_at = datetime('now') WHERE id = ?", (nid,))
    conn.commit()
    store.update_note(conn, nid, None, None, ["seen"], None, None)
    row = conn.execute("SELECT archived_at FROM notes WHERE id = ?", (nid,)).fetchone()
    assert row["archived_at"] is None
