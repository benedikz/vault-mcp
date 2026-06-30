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
