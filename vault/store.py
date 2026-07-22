import json
import re
import uuid

import sqlite_vec

from vault import config
from vault.errors import NotFoundError, ValidationError

# Leading YAML frontmatter block: --- ... --- at the very start of a body.
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---", re.DOTALL)


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
                chunks: list[str], vectors: list[list[float]],
                importance: float = 0.0) -> str:
    note_id = str(uuid.uuid4())
    with conn:
        cur = conn.execute(
            "INSERT INTO notes(id, title, body, tags, importance) VALUES(?, ?, ?, ?, ?)",
            (note_id, title, body, json.dumps(tags, ensure_ascii=False), importance),
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


def _revive(conn, ids: list[str]) -> None:
    """Any retrieval of an archived note returns it to the live tier
    (spontaneous recovery) — archiving demotes, it never loses anything.
    Caller holds the transaction."""
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id FROM notes WHERE id IN ({placeholders}) AND archived_at IS NOT NULL",
        ids,
    ).fetchall()
    for row in rows:
        conn.execute("UPDATE notes SET archived_at = NULL WHERE id = ?", (row["id"],))
        _audit(conn, "revive", row["id"])


def read_note(conn, note_id: str) -> dict:
    row = conn.execute(
        "SELECT id, title, body, tags, version, importance, updated_at "
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
        _revive(conn, [note_id])
        _audit(conn, "read", note_id)
    return {
        "id": row["id"],
        "title": row["title"],
        "body": row["body"],
        "tags": json.loads(row["tags"]),
        "version": row["version"],
        "importance": row["importance"],
        "updated_at": row["updated_at"],
    }


def update_note(conn, note_id, title, body, tags, chunks, vectors,
                importance: float | None = None) -> None:
    """Apply an edit. If ``chunks``/``vectors`` are not None the passages are
    rebuilt; pass None when only tags changed so embeddings are left intact.
    Editing counts as use, so an archived note is revived."""
    with conn:
        row = conn.execute(
            "SELECT seq, title, body, tags, version, importance "
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
        new_importance = importance if importance is not None else row["importance"]
        conn.execute(
            "UPDATE notes SET title = ?, body = ?, tags = ?, importance = ?, "
            "version = version + 1, updated_at = datetime('now'), archived_at = NULL "
            "WHERE seq = ?",
            (new_title, new_body, new_tags, new_importance, seq),
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
        # A trashed note must leave the graph entirely, or its edges keep
        # routing spreading activation and occupying recall candidate slots.
        conn.execute("DELETE FROM synapses WHERE src = ? OR dst = ?", (note_id, note_id))
        conn.execute("DELETE FROM node_stats WHERE note_id = ?", (note_id,))
        _audit(conn, "trash", note_id)


def connect_notes(conn, src: str, dst: str, kind: str, weight: float) -> None:
    if src == dst:
        raise ValidationError("cannot connect a note to itself")
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
    # A repeated id would pair with itself and create a self-loop synapse,
    # which self-amplifies under spreading activation.
    ids = list(dict.fromkeys(ids))
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
        _revive(conn, ids)


def add_temporal_edges(conn, note_id: str) -> list[str]:
    """Temporal contiguity: weakly bind a fresh note to the last few notes
    created inside the window, so notes written together recall together.
    Returns the ids that were bound."""
    rows = conn.execute(
        "SELECT id FROM notes WHERE id != ? AND deleted_at IS NULL "
        "AND created_at >= datetime('now', ?) "
        "ORDER BY seq DESC LIMIT ?",
        (note_id, f"-{config.TEMPORAL_WINDOW_MINUTES} minutes", config.TEMPORAL_TOPK),
    ).fetchall()
    if not rows:
        return []
    pairs = [(note_id, row["id"]) for row in rows] + [(row["id"], note_id) for row in rows]
    with conn:
        conn.executemany(
            "INSERT INTO synapses(src, dst, kind, weight) VALUES(?, ?, 'temporal', ?) "
            "ON CONFLICT(src, dst, kind) DO NOTHING",
            [(src, dst, config.TEMPORAL_WEIGHT) for src, dst in pairs],
        )
    return [row["id"] for row in rows]


def add_semantic_edges(conn, note_id: str, related: list[dict]) -> None:
    """Provisional semantic edges at encoding time, so a new note joins the
    associative graph immediately instead of waiting for the next
    consolidation (whose exact top-k rebuild supersedes these)."""
    strong = [r for r in related if r["similarity"] >= config.SEM_FLOOR and r["id"] != note_id]
    if not strong:
        return
    with conn:
        for r in strong:
            for src, dst in ((note_id, r["id"]), (r["id"], note_id)):
                conn.execute(
                    "INSERT INTO synapses(src, dst, kind, weight) VALUES(?, ?, 'semantic', ?) "
                    "ON CONFLICT(src, dst, kind) DO UPDATE SET "
                    "weight = max(weight, excluded.weight)",
                    (src, dst, r["similarity"]),
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


def list_clusters(conn, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT key, members, score FROM cluster_candidates ORDER BY score DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out: list[dict] = []
    for row in rows:
        member_ids = json.loads(row["members"])
        placeholders = ",".join("?" for _ in member_ids)
        notes = conn.execute(
            f"SELECT id, title FROM notes WHERE id IN ({placeholders}) AND deleted_at IS NULL",
            member_ids,
        ).fetchall()
        # members trashed since the flag can shrink a group below usefulness
        if len(notes) < config.CLUSTER_MIN_SIZE:
            continue
        out.append({
            "key": row["key"],
            "score": round(row["score"], 4),
            "notes": [{"id": n["id"], "title": n["title"]} for n in notes],
        })
    return out


# Frontmatter scalars surfaced by list_notes(include_meta=True). A deliberate
# whitelist — never the body, only these few catalog fields (kb.* live nested
# under a `kb:` key but a flat search over the frontmatter block finds them).
_META_FIELDS = ("type", "area", "confidence", "kind", "verified")


def _frontmatter_scalars(body: str) -> dict:
    """Extract a whitelist of scalar fields from a note's leading YAML
    frontmatter without a YAML dependency. Returns {} when there is no
    frontmatter. Never returns the body itself."""
    match = _FRONTMATTER_RE.match(body or "")
    if not match:
        return {}
    block = match.group(1)
    out: dict = {}
    for field in _META_FIELDS:
        found = re.search(rf"(?m)^\s*{field}:\s*(.+?)\s*$", block)
        if found:
            out[field] = found.group(1).strip().strip("\"'")
    if "confidence" in out:
        try:
            out["confidence"] = float(out["confidence"])
        except ValueError:
            pass
    return out


def list_notes(conn, limit: int | None = None, offset: int = 0,
               include_meta: bool = False) -> list[dict]:
    """Complete metadata catalog of every live (non-trashed) note, oldest-first.
    Never returns the body. Archived notes are included, flagged. With
    include_meta, a whitelist of frontmatter scalars is parsed server-side and
    returned under "meta" — still no body."""
    cols = ("n.id, n.title, n.tags, n.importance, n.version, "
            "n.created_at, n.updated_at, n.archived_at, "
            "COALESCE(s.salience, 0) AS salience")
    if include_meta:
        cols += ", n.body AS body"
    sql = (f"SELECT {cols} FROM notes n "
           "LEFT JOIN node_stats s ON s.note_id = n.id "
           "WHERE n.deleted_at IS NULL ORDER BY n.seq")
    params: list = []
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params = [limit, offset]
    rows = conn.execute(sql, params).fetchall()
    # degree = number of distinct neighbours (synapses are stored per direction)
    degree = {
        row["src"]: row["c"]
        for row in conn.execute(
            "SELECT src, COUNT(DISTINCT dst) AS c FROM synapses GROUP BY src"
        ).fetchall()
    }
    out: list[dict] = []
    for row in rows:
        item = {
            "id": row["id"],
            "title": row["title"],
            "tags": json.loads(row["tags"]),
            "importance": row["importance"],
            "version": row["version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "archived": row["archived_at"] is not None,
            "salience": round(row["salience"], 4),
            "degree": degree.get(row["id"], 0),
        }
        if include_meta:
            item["meta"] = _frontmatter_scalars(row["body"])
        out.append(item)
    return out
