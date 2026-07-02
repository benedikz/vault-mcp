import os
import secrets

from vault import config, db


def test_e5_models_get_prefixes():
    q, p = config.default_prefixes("intfloat/multilingual-e5-small")
    assert q == "query: " and p == "passage: "


def test_non_e5_models_get_no_prefix():
    assert config.default_prefixes("BAAI/bge-small-en") == ("", "")


def test_prefix_env_override(monkeypatch):
    monkeypatch.setenv("VAULT_TRANSPORT", "stdio")
    monkeypatch.setenv("VAULT_KEY", secrets.token_hex(32))
    monkeypatch.setenv("VAULT_QUERY_PREFIX", "")
    monkeypatch.setenv("VAULT_MODEL", "intfloat/multilingual-e5-small")
    cfg = config.load()
    assert cfg.query_prefix == ""        # explicitly disabled
    assert cfg.passage_prefix == "passage: "  # still defaulted


def test_init_sets_schema_version(conn):
    assert db.get_meta(conn, "schema_version") == str(config.SCHEMA_VERSION)


def test_migrate_from_v1_is_idempotent(conn):
    # simulate an old vault
    with conn:
        db.set_meta(conn, "schema_version", "1")
    applied = db.migrate(conn)
    assert applied == config.SCHEMA_VERSION - 1
    assert db.get_meta(conn, "schema_version") == str(config.SCHEMA_VERSION)
    # running again does nothing
    assert db.migrate(conn) == 0


def test_migrate_2_to_3_restores_v3_shape(conn, make_note):
    a = make_note("A", "alpha")
    b = make_note("B", "beta")
    with conn:
        conn.execute(
            "INSERT INTO synapses(src, dst, kind, weight) VALUES(?, ?, 'link', 1.0)", (a, b)
        )
    # devolve the schema to a genuine v2 shape
    with conn:
        conn.executescript(
            """
            DROP INDEX idx_notes_archived;
            ALTER TABLE notes DROP COLUMN importance;
            ALTER TABLE notes DROP COLUMN archived_at;
            DROP TABLE cluster_candidates;
            DROP TABLE synapses;
            CREATE TABLE synapses (
              src TEXT NOT NULL, dst TEXT NOT NULL,
              kind TEXT NOT NULL CHECK (kind IN ('link','semantic','hebbian')),
              weight REAL NOT NULL DEFAULT 0, last_fired_at TEXT,
              PRIMARY KEY (src, dst, kind));
            """
        )
        conn.execute(
            "INSERT INTO synapses(src, dst, kind, weight) VALUES(?, ?, 'link', 1.0)", (a, b)
        )
        db.set_meta(conn, "schema_version", "2")
    assert db.migrate(conn) == 1
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(notes)").fetchall()}
    assert {"importance", "archived_at"} <= cols
    # existing edges survive the synapses rebuild and 'temporal' is now allowed
    kept = conn.execute("SELECT COUNT(*) AS c FROM synapses WHERE kind = 'link'").fetchone()["c"]
    assert kept == 1
    with conn:
        conn.execute(
            "INSERT INTO synapses(src, dst, kind, weight) VALUES(?, ?, 'temporal', 0.3)", (b, a)
        )
    conn.execute("SELECT COUNT(*) FROM cluster_candidates").fetchone()
    # running again does nothing
    assert db.migrate(conn) == 0
