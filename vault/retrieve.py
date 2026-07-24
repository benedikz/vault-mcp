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
    archived: bool = False


def _fts_query(query: str) -> str:
    tokens = [tok for tok in _WS.split(query.strip()) if tok]
    return " OR ".join('"' + tok.replace('"', '""') + '"' for tok in tokens)


def dense_knn(conn, query_vector, k: int, sim_floor: float | None = None) -> list[str]:
    """Nearest passages, collapsed to their parent notes (best passage wins).

    Passages whose cosine similarity (1 - distance) falls below ``sim_floor``
    are dropped, so a query with no genuinely near note yields no dense seed
    (defaults to ``config.RETRIEVE_SIM_FLOOR`` when not given)."""
    if sim_floor is None:
        sim_floor = config.RETRIEVE_SIM_FLOOR
    max_distance = 1.0 - sim_floor
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
        if row["distance"] > max_distance:
            continue
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        ordered.append(row["id"])
        if len(ordered) >= k:
            break
    return ordered


def nearest_notes(conn, vector, k: int, exclude: str | None = None) -> list[dict]:
    """Nearest live notes by best-passage similarity. The vec0 table uses
    cosine distance, so similarity = 1 - distance."""
    blob = sqlite_vec.serialize_float32(vector)
    fetch = min(k * 8, config.MAX_FETCH)
    rows = conn.execute(
        "SELECT c.note_id AS id, n.title AS title, knn.distance AS distance "
        "FROM (SELECT rowid, distance FROM notes_vec WHERE embedding MATCH ? AND k = ?) AS knn "
        "JOIN chunks c ON c.seq = knn.rowid "
        "JOIN notes n ON n.id = c.note_id "
        "WHERE n.deleted_at IS NULL "
        "ORDER BY knn.distance",
        (blob, fetch),
    ).fetchall()
    out: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if row["id"] in seen or row["id"] == exclude:
            continue
        seen.add(row["id"])
        out.append({
            "id": row["id"],
            "title": row["title"],
            "similarity": round(max(0.0, 1.0 - row["distance"]), 4),
        })
        if len(out) >= k:
            break
    return out


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


def apply_strength(conn, scores: dict[str, float]) -> None:
    """Memory-strength factors:
    - retrieval practice (testing effect): notes actually recalled or read
      before are easier to recall again — saturating in access count, fading
      with time since last access;
    - importance declared at encoding;
    - archive penalty: archived notes are demoted in ranking, never hidden.
    """
    if not scores:
        return
    ids = list(scores.keys())
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT n.id AS id, n.importance AS importance, "
        f"n.archived_at IS NOT NULL AS archived, "
        f"COALESCE(s.access_count, 0) AS accesses, "
        f"julianday('now') - julianday(COALESCE(s.last_accessed_at, n.updated_at)) AS idle_days "
        f"FROM notes n LEFT JOIN node_stats s ON s.note_id = n.id "
        f"WHERE n.id IN ({placeholders})",
        ids,
    ).fetchall()
    for row in rows:
        practice = row["accesses"] / (row["accesses"] + config.PRACTICE_SAT)
        practice *= 0.5 ** (max(0.0, row["idle_days"] or 0.0) / config.PRACTICE_HALFLIFE_DAYS)
        factor = (1.0 + config.PRACTICE_WEIGHT * practice)
        factor *= (1.0 + config.IMPORTANCE_WEIGHT * row["importance"])
        if row["archived"]:
            factor *= config.ARCHIVE_PENALTY
        scores[row["id"]] *= factor


def load(conn, ids: list[str]) -> list[Candidate]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, title, body, archived_at FROM notes "
        f"WHERE id IN ({placeholders}) AND deleted_at IS NULL",
        ids,
    ).fetchall()
    by_id = {row["id"]: row for row in rows}
    out: list[Candidate] = []
    for note_id in ids:
        row = by_id.get(note_id)
        if row is not None:
            out.append(Candidate(id=row["id"], title=row["title"], body=row["body"],
                                 archived=row["archived_at"] is not None))
    return out


def search(conn, query_vector, query: str, k: int) -> list[Candidate]:
    seeds = rrf_fuse([dense_knn(conn, query_vector, k), keyword_bm25(conn, query, k)])
    apply_recency(conn, seeds)
    apply_strength(conn, seeds)
    ordered = [i for i, _ in sorted(seeds.items(), key=lambda kv: kv[1], reverse=True)][:k]
    return load(conn, ordered)


def recall(conn, query_vector, query: str, depth: int, k: int) -> list[Candidate]:
    seeds = rrf_fuse([dense_knn(conn, query_vector, k), keyword_bm25(conn, query, k)])
    activation = spread_activation(conn, seeds, depth)
    top = sorted(activation.items(), key=lambda kv: kv[1], reverse=True)[: config.CANDIDATE_CAP]
    subset = {note_id: score for note_id, score in top}
    apply_salience(conn, subset)
    apply_recency(conn, subset)
    apply_strength(conn, subset)
    ordered = [i for i, _ in sorted(subset.items(), key=lambda kv: kv[1], reverse=True)]
    return load(conn, ordered)


def neighbors(conn, note_id: str, limit: int = 20) -> list[dict]:
    # Edges are directional (semantic top-k is not symmetric, links point one
    # way), so look both ways and keep the strongest edge per (note, kind).
    rows = conn.execute(
        "SELECT id, title, kind, MAX(weight) AS weight FROM ("
        "  SELECT s.dst AS id, n.title AS title, s.kind AS kind, s.weight AS weight"
        "  FROM synapses s JOIN notes n ON n.id = s.dst"
        "  WHERE s.src = ? AND n.deleted_at IS NULL"
        "  UNION ALL"
        "  SELECT s.src AS id, n.title AS title, s.kind AS kind, s.weight AS weight"
        "  FROM synapses s JOIN notes n ON n.id = s.src"
        "  WHERE s.dst = ? AND n.deleted_at IS NULL"
        ") GROUP BY id, kind ORDER BY weight DESC LIMIT ?",
        (note_id, note_id, limit),
    ).fetchall()
    return [{"id": r["id"], "title": r["title"], "kind": r["kind"], "weight": r["weight"]} for r in rows]
