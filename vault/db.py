import threading

import sqlite_vec
from sqlcipher3 import dbapi2 as sqlcipher

from vault import config
from vault.errors import KeyVerificationError, VaultError

lock = threading.RLock()


def schema_sql(dim: int) -> str:
    return f"""
    -- archived_at marks the demoted ("forgotten") tier: the note keeps its
    -- content, versions and index entries, but is score-penalized in retrieval
    -- and excluded from the semantic-edge rebuild. Any recall or read clears
    -- it. Nothing is ever erased by archiving.
    CREATE TABLE IF NOT EXISTS notes (
      seq         INTEGER PRIMARY KEY,
      id          TEXT NOT NULL UNIQUE,
      title       TEXT NOT NULL,
      body        TEXT NOT NULL,
      tags        TEXT NOT NULL DEFAULT '[]',
      embeddable  INTEGER NOT NULL DEFAULT 1,
      version     INTEGER NOT NULL DEFAULT 1,
      importance  REAL NOT NULL DEFAULT 0,
      created_at  TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
      archived_at TEXT,
      deleted_at  TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_notes_deleted ON notes(deleted_at);
    CREATE INDEX IF NOT EXISTS idx_notes_archived ON notes(archived_at);

    CREATE TABLE IF NOT EXISTS note_versions (
      id         INTEGER PRIMARY KEY,
      note_id    TEXT NOT NULL,
      title      TEXT NOT NULL,
      body       TEXT NOT NULL,
      tags       TEXT NOT NULL,
      version    INTEGER NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_versions_note ON note_versions(note_id);

    -- One row per embedded passage. notes_vec is keyed by chunks.seq so a long
    -- note contributes several vectors instead of one truncated vector.
    CREATE TABLE IF NOT EXISTS chunks (
      seq        INTEGER PRIMARY KEY,
      note_id    TEXT NOT NULL,
      ord        INTEGER NOT NULL,
      text       TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_note ON chunks(note_id);

    CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
      USING fts5(title, body, tokenize='unicode61 remove_diacritics 2');

    CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec
      USING vec0(embedding float[{dim}] distance_metric=cosine);

    CREATE TABLE IF NOT EXISTS synapses (
      src           TEXT NOT NULL,
      dst           TEXT NOT NULL,
      kind          TEXT NOT NULL CHECK (kind IN ('link','semantic','hebbian','temporal')),
      weight        REAL NOT NULL DEFAULT 0,
      last_fired_at TEXT,
      PRIMARY KEY (src, dst, kind)
    );
    CREATE INDEX IF NOT EXISTS idx_syn_src ON synapses(src);
    CREATE INDEX IF NOT EXISTS idx_syn_dst ON synapses(dst);

    CREATE TABLE IF NOT EXISTS node_stats (
      note_id          TEXT PRIMARY KEY,
      access_count     INTEGER NOT NULL DEFAULT 0,
      salience         REAL NOT NULL DEFAULT 0,
      last_accessed_at TEXT
    );

    -- Near-duplicate note pairs surfaced by consolidation for the agent to
    -- review. The vault never auto-merges; it only flags.
    CREATE TABLE IF NOT EXISTS duplicate_candidates (
      a          TEXT NOT NULL,
      b          TEXT NOT NULL,
      score      REAL NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      PRIMARY KEY (a, b)
    );

    -- Dense groups of mutually similar notes surfaced by consolidation for
    -- the agent to synthesize into a gist note. Flag-only, like duplicates.
    CREATE TABLE IF NOT EXISTS cluster_candidates (
      key        TEXT PRIMARY KEY,
      members    TEXT NOT NULL,
      score      REAL NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS audit_log (
      id         INTEGER PRIMARY KEY,
      action     TEXT NOT NULL,
      note_id    TEXT,
      detail     TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);

    CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
    """


def connect_raw(db_path: str, key_hex: str):
    conn = sqlcipher.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlcipher.Row
    conn.execute(f"PRAGMA key = \"x'{key_hex}'\"")
    conn.execute("PRAGMA cipher_memory_security = ON")
    try:
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlcipher.DatabaseError as exc:
        conn.close()
        raise KeyVerificationError("wrong key or not a vault database") from exc
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def connect(cfg):
    return connect_raw(cfg.db_path, cfg.key_hex)


def init_schema(conn, dim: int) -> None:
    conn.executescript(schema_sql(dim))
    set_meta(conn, "schema_version", str(config.SCHEMA_VERSION))


def get_meta(conn, key: str):
    try:
        row = conn.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
    except sqlcipher.OperationalError:
        return None
    return row["v"] if row else None


def set_meta(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        (key, value),
    )


def _migrate_1_to_2(conn) -> None:
    """v1 stored one vector per note (notes_vec keyed by notes.seq). v2 keys
    vectors by chunk. Any v1 vectors are dropped; bodies are re-chunked and
    re-embedded by a one-off reindex (see bin/reindex.py)."""
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              seq INTEGER PRIMARY KEY, note_id TEXT NOT NULL, ord INTEGER NOT NULL, text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_note ON chunks(note_id);
            CREATE TABLE IF NOT EXISTS duplicate_candidates (
              a TEXT NOT NULL, b TEXT NOT NULL, score REAL NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now')), PRIMARY KEY (a, b)
            );
            DELETE FROM notes_vec;
            """
        )


def _migrate_2_to_3(conn) -> None:
    """v3 adds notes.importance / notes.archived_at, the 'temporal' synapse
    kind (a CHECK change, so the table is rebuilt in place), and the
    cluster_candidates table. Column/DDL checks make it safe to re-run on a
    partially migrated vault. No re-embedding is needed."""
    note_cols = {row["name"] for row in conn.execute("PRAGMA table_info(notes)").fetchall()}
    with conn:
        if "importance" not in note_cols:
            conn.execute("ALTER TABLE notes ADD COLUMN importance REAL NOT NULL DEFAULT 0")
        if "archived_at" not in note_cols:
            conn.execute("ALTER TABLE notes ADD COLUMN archived_at TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_archived ON notes(archived_at)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cluster_candidates (
              key TEXT PRIMARY KEY, members TEXT NOT NULL, score REAL NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now')))
            """
        )
    ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'synapses'"
    ).fetchone()
    if ddl and "'temporal'" not in ddl["sql"]:
        conn.executescript(
            """
            CREATE TABLE synapses_v3 (
              src           TEXT NOT NULL,
              dst           TEXT NOT NULL,
              kind          TEXT NOT NULL CHECK (kind IN ('link','semantic','hebbian','temporal')),
              weight        REAL NOT NULL DEFAULT 0,
              last_fired_at TEXT,
              PRIMARY KEY (src, dst, kind)
            );
            INSERT INTO synapses_v3 SELECT src, dst, kind, weight, last_fired_at FROM synapses;
            DROP TABLE synapses;
            ALTER TABLE synapses_v3 RENAME TO synapses;
            CREATE INDEX IF NOT EXISTS idx_syn_src ON synapses(src);
            CREATE INDEX IF NOT EXISTS idx_syn_dst ON synapses(dst);
            """
        )


_MIGRATIONS = {2: _migrate_1_to_2, 3: _migrate_2_to_3}


def migrate(conn) -> int:
    """Bring an existing vault up to the current schema version. Returns the
    number of migration steps applied."""
    current = int(get_meta(conn, "schema_version") or "1")
    applied = 0
    for version in range(current + 1, config.SCHEMA_VERSION + 1):
        step = _MIGRATIONS.get(version)
        if step is not None:
            step(conn)
        with conn:
            set_meta(conn, "schema_version", str(version))
        applied += 1
    return applied


def require_initialized(conn, embed_dim: int) -> None:
    if get_meta(conn, "canary") != "ok":
        raise VaultError("vault is not initialized — run: python bin/init.py")
    stored = get_meta(conn, "dim")
    if stored != str(embed_dim):
        raise VaultError(
            f"embedding dim {embed_dim} does not match vault dim {stored}; "
            "set VAULT_MODEL back to the original model or re-initialize"
        )
    version = int(get_meta(conn, "schema_version") or "1")
    if version < config.SCHEMA_VERSION:
        raise VaultError(
            f"vault schema is v{version}, code expects v{config.SCHEMA_VERSION}; "
            "run: python bin/reindex.py to migrate"
        )
