import json
import re

import numpy as np

from vault import config, db

_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _note_centroids(conn) -> tuple[list[str], np.ndarray]:
    """Mean of each note's passage vectors, L2-normalized. One row per note."""
    rows = conn.execute(
        "SELECT c.note_id AS id, vec_to_json(v.embedding) AS vec "
        "FROM notes_vec v JOIN chunks c ON c.seq = v.rowid "
        "JOIN notes n ON n.id = c.note_id WHERE n.deleted_at IS NULL"
    ).fetchall()
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for row in rows:
        vec = np.array(json.loads(row["vec"]), dtype=np.float32)
        nid = row["id"]
        if nid in sums:
            sums[nid] += vec
            counts[nid] += 1
        else:
            sums[nid] = vec.copy()
            counts[nid] = 1
    ids = list(sums.keys())
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = np.array([sums[i] / counts[i] for i in ids], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return ids, matrix / norms


def _rebuild_semantic(conn, stats: dict) -> None:
    ids, matrix = _note_centroids(conn)
    count = len(ids)
    if count < 2:
        return
    block = 512
    top_k = min(config.SEM_TOPK, count - 1)
    with conn:
        conn.execute("DELETE FROM synapses WHERE kind = 'semantic'")
        conn.execute("DELETE FROM duplicate_candidates")
        for start in range(0, count, block):
            sims = matrix[start : start + block] @ matrix.T
            for local, global_i in enumerate(range(start, min(start + block, count))):
                row = sims[local]
                row[global_i] = -1.0
                for j in np.argpartition(-row, top_k)[:top_k]:
                    score = float(row[j])
                    if score < config.SEM_FLOOR:
                        continue
                    j = int(j)
                    conn.execute(
                        "INSERT INTO synapses(src, dst, kind, weight) VALUES(?, ?, 'semantic', ?) "
                        "ON CONFLICT(src, dst, kind) DO UPDATE SET weight = excluded.weight",
                        (ids[global_i], ids[j], score),
                    )
                    stats["semantic"] += 1
                    if score >= config.DEDUP_FLOOR and ids[global_i] < ids[j]:
                        conn.execute(
                            "INSERT INTO duplicate_candidates(a, b, score) VALUES(?, ?, ?) "
                            "ON CONFLICT(a, b) DO UPDATE SET score = excluded.score, "
                            "created_at = datetime('now')",
                            (ids[global_i], ids[j], score),
                        )
                        stats["duplicates"] += 1


def _rebuild_links(conn, stats: dict) -> None:
    rows = conn.execute("SELECT id, title, body FROM notes WHERE deleted_at IS NULL").fetchall()
    title_index = {row["title"].strip().lower(): row["id"] for row in rows}
    with conn:
        conn.execute("DELETE FROM synapses WHERE kind = 'link'")
        for row in rows:
            for target in _LINK_RE.findall(row["body"]):
                key = target.strip().lower()
                dst = title_index.get(key)
                if dst is None and key.startswith("id:"):
                    candidate = target.strip()[3:]
                    if conn.execute(
                        "SELECT 1 FROM notes WHERE id = ? AND deleted_at IS NULL", (candidate,)
                    ).fetchone():
                        dst = candidate
                if dst and dst != row["id"]:
                    conn.execute(
                        "INSERT INTO synapses(src, dst, kind, weight) VALUES(?, ?, 'link', 1.0) "
                        "ON CONFLICT(src, dst, kind) DO UPDATE SET weight = 1.0",
                        (row["id"], dst),
                    )
                    stats["link"] += 1


def _decay_and_prune(conn, stats: dict) -> None:
    with conn:
        conn.execute(
            "UPDATE synapses SET weight = weight * ? "
            "WHERE kind = 'hebbian' AND (last_fired_at IS NULL OR last_fired_at < datetime('now', ?))",
            (config.HEBB_DECAY, f"-{config.HEBB_DECAY_DAYS} days"),
        )
        cur = conn.execute(
            "DELETE FROM synapses WHERE kind = 'hebbian' AND weight < ?",
            (config.HEBB_PRUNE_FLOOR,),
        )
        stats["pruned"] = cur.rowcount


def _recompute_salience(conn, stats: dict) -> None:
    with conn:
        conn.execute("UPDATE node_stats SET salience = 0")
        conn.execute(
            "INSERT INTO node_stats(note_id, salience) "
            "SELECT dst, SUM(weight) FROM synapses GROUP BY dst "
            "ON CONFLICT(note_id) DO UPDATE SET salience = excluded.salience"
        )
        peak = conn.execute("SELECT MAX(salience) AS m FROM node_stats").fetchone()["m"] or 0.0
        if peak > 0.0:
            conn.execute("UPDATE node_stats SET salience = salience / ?", (peak,))
        stats["salient_nodes"] = conn.execute(
            "SELECT COUNT(*) AS c FROM node_stats WHERE salience > 0"
        ).fetchone()["c"]


def consolidate(conn) -> dict:
    stats = {"semantic": 0, "link": 0, "pruned": 0, "salient_nodes": 0, "duplicates": 0}
    with db.lock:
        _rebuild_semantic(conn, stats)
        _rebuild_links(conn, stats)
        _decay_and_prune(conn, stats)
        _recompute_salience(conn, stats)
        with conn:
            conn.execute(
                "INSERT INTO audit_log(action, detail) VALUES('consolidate', ?)",
                (json.dumps(stats),),
            )
    return stats
