import json
import uuid

import sqlite_vec

from vault import config
from vault.errors import NotFoundError


def _audit(conn, action: str, note_id: str | None = None, detail: str | None = None) -> None:
    conn.execute(
        "INSERT INTO audit_log(action, note_id, detail) VALUES(?, ?, ?)",
        (action, note_id, detail),
    )


def _vector(values) -> bytes:
    return sqlite_vec.serialize_float32(values)


def _insert_chunks(conn, note_id: str, chunks: list[str], vectors: list[list[float]]) -> None:
    for ord_, (text, vector) in enumerate(zip(chunks, vectors)):
        cur = conn.execute(
            "INSERT INTO chunks(note_id, ord, text) VALUES(?, ?, ?)",
            (note_id, ord_, text),
        )
        conn.execute(
            "INSERT INTO notes_vec(rowid, embedding) VALUES(?, ?)",
            (cur.lastrowid, _vector(vector)),
        )


def _drop_chunks(conn, note_id: str) -> None:
    rows = conn.execute("SELECT seq FROM chunks WHERE note_id = ?", (note_id,)).fetchall()
    for row in rows:
        conn.execute("DELETE FROM notes_vec WHERE rowid = ?", (row["seq"],))
    conn.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))


def create_note(conn, title: str, body: str, tags: list[str],
                chunks: list[str], vectors: list[list[float]]) -> str:
    note_id = str(uuid.uuid4())
    with conn:
        cur = conn.execute(
            "INSERT INTO notes(id, title, body, tags) VALUES(?, ?, ?, ?)",
            (note_id, title, body, json.dumps(tags, ensure_ascii=False)),
        )
        seq = cur.lastrowid
        conn.execute("INSERT INTO notes_fts(rowid, title, body) VALUES(?, ?, ?)", (seq, title, body))
        _insert_chunks(conn, note_id, chunks, vectors)
        conn.execute("INSERT INTO node_stats(note_id) VALUES(?)", (note_id,))
        _audit(conn, "create", note_id)
    return note_id


def peek(conn, note_id: str) -> dict:
    row = conn.execute(
        "SELECT title, body FROM notes WHERE id = ? AND deleted_at IS NULL", (note_id,)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"note {note_id} not found")
    return {"title": row["title"], "body": row["body"]}


def read_note(conn, note_id: str) -> dict:
    row = conn.execute(
        "SELECT id, title, body, tags, version, updated_at "
        "FROM notes WHERE id = ? AND deleted_at IS NULL",
        (note_id,),
    ).fetchone()
    if row is None:
        raise NotFoundError(f"note {note_id} not found")
    with conn:
        conn.execute(
            "UPDATE node_stats SET access_count = access_count + 1, "
            "last_accessed_at = datetime('now') WHERE note_id = ?",
            (note_id,),
        )
        _audit(conn, "read", note_id)
    return {
        "id": row["id"],
        "title": row["title"],
        "body": row["body"],
        "tags": json.loads(row["tags"]),
        "version": row["version"],
        "updated_at": row["updated_at"],
    }


def update_note(conn, note_id, title, body, tags, chunks, vectors) -> None:
    """Apply an edit. If ``chunks``/``vectors`` are not None the passages are
    rebuilt; pass None when only tags changed so embeddings are left intact."""
    with conn:
        row = conn.execute(
            "SELECT seq, title, body, tags, version "
            "FROM notes WHERE id = ? AND deleted_at IS NULL",
            (note_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"note {note_id} not found")
        seq = row["seq"]
        conn.execute(
            "INSERT INTO note_versions(note_id, title, body, tags, version) VALUES(?, ?, ?, ?, ?)",
            (note_id, row["title"], row["body"], row["tags"], row["version"]),
        )
        new_title = title if title is not None else row["title"]
        new_body = body if body is not None else row["body"]
        new_tags = json.dumps(tags, ensure_ascii=False) if tags is not None else row["tags"]
        conn.execute(
            "UPDATE notes SET title = ?, body = ?, tags = ?, version = version + 1, "
            "updated_at = datetime('now') WHERE seq = ?",
            (new_title, new_body, new_tags, seq),
        )
        conn.execute("DELETE FROM notes_fts WHERE rowid = ?", (seq,))
        conn.execute("INSERT INTO notes_fts(rowid, title, body) VALUES(?, ?, ?)", (seq, new_title, new_body))
        if chunks is not None:
            _drop_chunks(conn, note_id)
            _insert_chunks(conn, note_id, chunks, vectors)
        _audit(conn, "update", note_id)


def trash_note(conn, note_id: str) -> None:
    with conn:
        row = conn.execute(
            "SELECT seq FROM notes WHERE id = ? AND deleted_at IS NULL", (note_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"note {note_id} not found")
        seq = row["seq"]
        conn.execute("UPDATE notes SET deleted_at = datetime('now') WHERE seq = ?", (seq,))
        conn.execute("DELETE FROM notes_fts WHERE rowid = ?", (seq,))
        _drop_chunks(conn, note_id)
        conn.execute("DELETE FROM duplicate_candidates WHERE a = ? OR b = ?", (note_id, note_id))
        _audit(conn, "trash", note_id)


def connect_notes(conn, src: str, dst: str, kind: str, weight: float) -> None:
    for ident in (src, dst):
        if conn.execute(
            "SELECT 1 FROM notes WHERE id = ? AND deleted_at IS NULL", (ident,)
        ).fetchone() is None:
            raise NotFoundError(f"note {ident} not found")
    with conn:
        conn.execute(
            "INSERT INTO synapses(src, dst, kind, weight, last_fired_at) "
            "VALUES(?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(src, dst, kind) DO UPDATE SET weight = excluded.weight, "
            "last_fired_at = datetime('now')",
            (src, dst, kind, weight),
        )
        _audit(conn, "connect", src, dst)


def reinforce(conn, ids: list[str], delta: float = config.HEBB_DELTA, cap: float = config.HEBB_CAP) -> None:
    pairs: list[tuple[str, str]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs.append((ids[i], ids[j]))
            pairs.append((ids[j], ids[i]))
    if not pairs:
        return
    with conn:
        conn.executemany(
            "INSERT INTO synapses(src, dst, kind, weight, last_fired_at) "
            "VALUES(?, ?, 'hebbian', ?, datetime('now')) "
            "ON CONFLICT(src, dst, kind) DO UPDATE SET weight = min(weight + ?, ?), "
            "last_fired_at = datetime('now')",
            [(src, dst, delta, delta, cap) for src, dst in pairs],
        )


def touch_access(conn, ids: list[str]) -> None:
    if not ids:
        return
    with conn:
        conn.executemany(
            "UPDATE node_stats SET access_count = access_count + 1, "
            "last_accessed_at = datetime('now') WHERE note_id = ?",
            [(i,) for i in ids],
        )


def replace_chunks(conn, note_id: str, chunks: list[str], vectors: list[list[float]]) -> None:
    """Rebuild a note's passages and vectors in place (used by reindex)."""
    with conn:
        if conn.execute("SELECT 1 FROM notes WHERE id = ?", (note_id,)).fetchone() is None:
            raise NotFoundError(f"note {note_id} not found")
        _drop_chunks(conn, note_id)
        _insert_chunks(conn, note_id, chunks, vectors)


def iter_live_notes(conn):
    """Yield (id, title, body) for every non-trashed note."""
    for row in conn.execute(
        "SELECT id, title, body FROM notes WHERE deleted_at IS NULL ORDER BY seq"
    ).fetchall():
        yield row["id"], row["title"], row["body"]


def list_duplicates(conn, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT d.a AS a, d.b AS b, d.score AS score, "
        "       na.title AS a_title, nb.title AS b_title "
        "FROM duplicate_candidates d "
        "JOIN notes na ON na.id = d.a AND na.deleted_at IS NULL "
        "JOIN notes nb ON nb.id = d.b AND nb.deleted_at IS NULL "
        "ORDER BY d.score DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {"a": r["a"], "a_title": r["a_title"], "b": r["b"], "b_title": r["b_title"],
         "score": round(r["score"], 4)}
        for r in rows
    ]
