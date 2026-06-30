import re
from dataclasses import dataclass

import sqlite_vec

from vault import config

_WS = re.compile(r"\s+")


@dataclass
class Candidate:
    id: str
    title: str
    body: str


def _fts_query(query: str) -> str:
    tokens = [tok for tok in _WS.split(query.strip()) if tok]
    return " OR ".join('"' + tok.replace('"', '""') + '"' for tok in tokens)


def dense_knn(conn, query_vector, k: int) -> list[str]:
    """Nearest passages, collapsed to their parent notes (best passage wins)."""
    blob = sqlite_vec.serialize_float32(query_vector)
    fetch = min(k * 8, config.MAX_FETCH)
    rows = conn.execute(
        "SELECT c.note_id AS id, knn.distance AS distance "
        "FROM (SELECT rowid, distance FROM notes_vec WHERE embedding MATCH ? AND k = ?) AS knn "
        "JOIN chunks c ON c.seq = knn.rowid "
        "JOIN notes n ON n.id = c.note_id "
        "WHERE n.deleted_at IS NULL "
        "ORDER BY knn.distance",
        (blob, fetch),
    ).fetchall()
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        ordered.append(row["id"])
        if len(ordered) >= k:
            break
    return ordered


def keyword_bm25(conn, query: str, k: int) -> list[str]:
    match = _fts_query(query)
    if not match:
        return []
    rows = conn.execute(
        "SELECT n.id AS id "
        "FROM notes_fts JOIN notes n ON n.seq = notes_fts.rowid "
        "WHERE notes_fts MATCH ? AND n.deleted_at IS NULL "
        "ORDER BY bm25(notes_fts) LIMIT ?",
        (match, k),
    ).fetchall()
    return [row["id"] for row in rows]


def rrf_fuse(ranked_lists: list[list[str]], k_rrf: int = config.RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k_rrf + rank + 1)
    return scores


def spread_activation(conn, seeds: dict[str, float], depth: int, decay: float = config.SPREAD_DECAY) -> dict[str, float]:
    activation = dict(seeds)
    frontier = list(seeds.keys())
    for hop in range(1, depth + 1):
        if not frontier:
            break
        frontier = sorted(frontier, key=lambda i: activation.get(i, 0.0), reverse=True)[: config.SPREAD_FRONTIER_CAP]
        placeholders = ",".join("?" for _ in frontier)
        rows = conn.execute(
            f"SELECT src, dst, weight FROM synapses WHERE src IN ({placeholders})",
            frontier,
        ).fetchall()
        reached: set[str] = set()
        for row in rows:
            delta = activation.get(row["src"], 0.0) * row["weight"] * (decay ** hop)
            if delta <= 1e-3:
                continue
            activation[row["dst"]] = activation.get(row["dst"], 0.0) + delta
            reached.add(row["dst"])
        frontier = list(reached)
    return activation


def apply_salience(conn, activation: dict[str, float]) -> None:
    if not activation:
        return
    ids = list(activation.keys())
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT note_id, salience FROM node_stats WHERE note_id IN ({placeholders})",
        ids,
    ).fetchall()
    salience = {row["note_id"]: row["salience"] for row in rows}
    for note_id in activation:
        activation[note_id] *= 1.0 + 0.25 * salience.get(note_id, 0.0)


def apply_recency(conn, scores: dict[str, float],
                  halflife: float = config.RECENCY_HALFLIFE_DAYS,
                  weight: float = config.RECENCY_WEIGHT) -> None:
    """Blend a time-decay factor in: a brand-new note keeps its full score, an
    infinitely old one is scaled to (1 - weight). Age is read from the DB clock
    so behaviour does not depend on the process wall clock."""
    if not scores or weight <= 0.0:
        return
    ids = list(scores.keys())
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, julianday('now') - julianday(updated_at) AS age_days "
        f"FROM notes WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    ages = {row["id"]: max(0.0, row["age_days"] or 0.0) for row in rows}
    for note_id in scores:
        age = ages.get(note_id, 0.0)
        decay = 0.5 ** (age / halflife)
        scores[note_id] *= (1.0 - weight) + weight * decay


def load(conn, ids: list[str]) -> list[Candidate]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, title, body FROM notes WHERE id IN ({placeholders}) AND deleted_at IS NULL",
        ids,
    ).fetchall()
    by_id = {row["id"]: row for row in rows}
    out: list[Candidate] = []
    for note_id in ids:
        row = by_id.get(note_id)
        if row is not None:
            out.append(Candidate(id=row["id"], title=row["title"], body=row["body"]))
    return out


def search(conn, query_vector, query: str, k: int) -> list[Candidate]:
    seeds = rrf_fuse([dense_knn(conn, query_vector, k), keyword_bm25(conn, query, k)])
    apply_recency(conn, seeds)
    ordered = [i for i, _ in sorted(seeds.items(), key=lambda kv: kv[1], reverse=True)][:k]
    return load(conn, ordered)


def recall(conn, query_vector, query: str, depth: int, k: int) -> list[Candidate]:
    seeds = rrf_fuse([dense_knn(conn, query_vector, k), keyword_bm25(conn, query, k)])
    activation = spread_activation(conn, seeds, depth)
    top = sorted(activation.items(), key=lambda kv: kv[1], reverse=True)[: config.CANDIDATE_CAP]
    subset = {note_id: score for note_id, score in top}
    apply_salience(conn, subset)
    apply_recency(conn, subset)
    ordered = [i for i, _ in sorted(subset.items(), key=lambda kv: kv[1], reverse=True)]
    return load(conn, ordered)


def neighbors(conn, note_id: str, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT s.dst AS id, s.kind AS kind, s.weight AS weight, n.title AS title "
        "FROM synapses s JOIN notes n ON n.id = s.dst "
        "WHERE s.src = ? AND n.deleted_at IS NULL "
        "ORDER BY s.weight DESC LIMIT ?",
        (note_id, limit),
    ).fetchall()
    return [{"id": r["id"], "title": r["title"], "kind": r["kind"], "weight": r["weight"]} for r in rows]
