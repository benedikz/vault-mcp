import hashlib
import json
import re

import numpy as np

from vault import config, db

_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _note_centroids(conn) -> tuple[list[str], np.ndarray]:
    """Mean of each note's passage vectors, L2-normalized. One row per note.
    Archived notes are excluded: they leave the semantic web (and the O(n^2)
    rebuild) but stay fully searchable and revive on use."""
    rows = conn.execute(
        "SELECT c.note_id AS id, vec_to_json(v.embedding) AS vec "
        "FROM notes_vec v JOIN chunks c ON c.seq = v.rowid "
        "JOIN notes n ON n.id = c.note_id "
        "WHERE n.deleted_at IS NULL AND n.archived_at IS NULL"
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


def _rebuild_semantic(conn, stats: dict) -> list[tuple[str, str, float]]:
    """Rebuild semantic edges and flag duplicates. Returns the strongly
    similar pairs (cosine >= CLUSTER_FLOOR) for gist-cluster detection."""
    cluster_pairs: list[tuple[str, str, float]] = []
    ids, matrix = _note_centroids(conn)
    count = len(ids)
    if count < 2:
        return cluster_pairs
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
                    if score >= config.CLUSTER_FLOOR and ids[global_i] < ids[j]:
                        cluster_pairs.append((ids[global_i], ids[j], score))
                    if score >= config.DEDUP_FLOOR and ids[global_i] < ids[j]:
                        conn.execute(
                            "INSERT INTO duplicate_candidates(a, b, score) VALUES(?, ?, ?) "
                            "ON CONFLICT(a, b) DO UPDATE SET score = excluded.score, "
                            "created_at = datetime('now')",
                            (ids[global_i], ids[j], score),
                        )
                        stats["duplicates"] += 1
    return cluster_pairs


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


def _archive_stale(conn, stats: dict) -> None:
    """Active forgetting, without erasing. Demote notes that are idle past
    ARCHIVE_AFTER_DAYS, structurally peripheral (salience from the previous
    pass below the ceiling) and not declared important. An archived note keeps
    its content, versions, chunks and index entries; it is only score-penalized
    in retrieval and excluded from the semantic rebuild, and any recall, read
    or edit revives it (store._revive)."""
    with conn:
        cur = conn.execute(
            "UPDATE notes SET archived_at = datetime('now') WHERE id IN ("
            "  SELECT n.id FROM notes n LEFT JOIN node_stats s ON s.note_id = n.id"
            "  WHERE n.deleted_at IS NULL AND n.archived_at IS NULL"
            "    AND n.importance < ?"
            "    AND COALESCE(s.salience, 0) < ?"
            "    AND julianday('now') - julianday(MAX(n.updated_at, "
            "        COALESCE(s.last_accessed_at, n.updated_at))) > ?"
            ")",
            (config.ARCHIVE_IMPORTANCE_EXEMPT, config.ARCHIVE_SALIENCE_CEILING,
             config.ARCHIVE_AFTER_DAYS),
        )
        stats["archived"] = cur.rowcount


def _flag_clusters(conn, pairs: list[tuple[str, str, float]], stats: dict) -> None:
    """Systems consolidation, agent-in-the-loop: connected components of the
    strongly-similar pairs are flagged so the agent can distill each group
    into a gist note that wiki-links its members. A group already covered by
    such a hub is not re-flagged, so the queue self-heals after synthesis.
    Must run after _rebuild_links so hub detection sees fresh link edges."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        parent[find(a)] = find(b)

    groups: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    for a, b, score in pairs:
        root = find(a)
        groups.setdefault(root, [])
        scores.setdefault(root, []).append(score)
    for note_id in parent:
        groups[find(note_id)].append(note_id)

    with conn:
        conn.execute("DELETE FROM cluster_candidates")
        for root, members in groups.items():
            if len(members) < config.CLUSTER_MIN_SIZE:
                continue
            if _has_hub(conn, members):
                continue
            members = sorted(members)
            key = hashlib.sha1("|".join(members).encode("utf-8")).hexdigest()[:16]
            mean_score = sum(scores[root]) / len(scores[root])
            conn.execute(
                "INSERT INTO cluster_candidates(key, members, score) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET score = excluded.score, "
                "created_at = datetime('now')",
                (key, json.dumps(members), mean_score),
            )
            stats["clusters"] += 1


def _has_hub(conn, members: list[str]) -> bool:
    """True when one note's link edges already cover most of the group — the
    gist note exists, whether it is a member or an outside summary note."""
    placeholders = ",".join("?" for _ in members)
    row = conn.execute(
        f"SELECT src, COUNT(DISTINCT dst) AS covered FROM synapses "
        f"WHERE kind = 'link' AND dst IN ({placeholders}) "
        f"GROUP BY src ORDER BY covered DESC LIMIT 1",
        members,
    ).fetchone()
    if row is None:
        return False
    linkable = len(members) - 1 if row["src"] in members else len(members)
    return row["covered"] >= config.CLUSTER_HUB_COVERAGE * linkable


def _decay_and_prune(conn, stats: dict) -> None:
    with conn:
        # trash_note removes a note's edges, but vaults trashed under older
        # code still carry them — sweep any edge touching a non-live note.
        cur = conn.execute(
            "DELETE FROM synapses "
            "WHERE src NOT IN (SELECT id FROM notes WHERE deleted_at IS NULL) "
            "OR dst NOT IN (SELECT id FROM notes WHERE deleted_at IS NULL)"
        )
        stats["orphaned"] = cur.rowcount
        conn.execute(
            "DELETE FROM node_stats WHERE note_id NOT IN "
            "(SELECT id FROM notes WHERE deleted_at IS NULL)"
        )
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
    stats = {"semantic": 0, "link": 0, "pruned": 0, "orphaned": 0,
             "salient_nodes": 0, "duplicates": 0, "archived": 0, "clusters": 0}
    with db.lock:
        # archive first (using last pass's salience) so demoted notes drop out
        # of this pass's semantic rebuild
        _archive_stale(conn, stats)
        cluster_pairs = _rebuild_semantic(conn, stats)
        _rebuild_links(conn, stats)
        _flag_clusters(conn, cluster_pairs, stats)
        _decay_and_prune(conn, stats)
        _recompute_salience(conn, stats)
        with conn:
            conn.execute(
                "INSERT INTO audit_log(action, detail) VALUES('consolidate', ?)",
                (json.dumps(stats),),
            )
    return stats
