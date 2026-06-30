import os
import secrets
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vault import db, embed  # noqa: E402
from vault.errors import AlreadyInitializedError  # noqa: E402


def main() -> None:
    db_path = os.environ.get("VAULT_DB_PATH", "").strip() or os.path.expanduser("~/.vault/brain.db")
    model = os.environ.get("VAULT_MODEL", "intfloat/multilingual-e5-small").strip()
    cache_dir = os.environ.get("VAULT_CACHE", "").strip() or None

    if os.path.exists(db_path):
        raise AlreadyInitializedError(f"{db_path} already exists; refusing to overwrite")

    parent = os.path.dirname(db_path) or "."
    os.makedirs(parent, exist_ok=True)
    os.chmod(parent, 0o700)

    key_hex = secrets.token_hex(32)
    embedder = embed.Embedder(model, "", False, cache_dir)

    conn = db.connect_raw(db_path, key_hex)
    db.init_schema(conn, embedder.dim)
    with conn:
        db.set_meta(conn, "canary", "ok")
        db.set_meta(conn, "dim", str(embedder.dim))
        db.set_meta(conn, "model", model)
    conn.close()
    os.chmod(db_path, 0o600)

    print(f"Vault initialized at {db_path}")
    print(f"Embedding model: {model}  (dim {embedder.dim})")
    print()
    print("=== ENCRYPTION KEY — shown once ===")
    print(key_hex)
    print()
    print("1. Set it for the server:   export VAULT_KEY=<the key above>")
    print("2. Split it for recovery (offline), see RECOVERY.md (SLIP-39 / ssss).")
    print("3. Store NO plaintext copy alongside the database.")


if __name__ == "__main__":
    main()
