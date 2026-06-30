import pytest

from vault import store
from vault.errors import NotFoundError


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
