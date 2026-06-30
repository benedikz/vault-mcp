"""Make a consistent, encrypted backup of the vault.

Uses SQLCipher's ``sqlcipher_export`` so the copy is a fully-encrypted database
(same key as the live vault), taken without stopping the server. Optionally
seals that copy to an *offline* public key with `age`, so an off-site backup
needs both the vault key AND the offline private key to open — the asymmetric
"server can encrypt it, only the offline holder can decrypt it" property.

    export VAULT_KEY=... VAULT_DB_PATH=...
    .venv/bin/python bin/backup.py /mnt/ssd/vault/backups
    # optional off-site sealing:
    export VAULT_BACKUP_RECIPIENT="age1...."        # an age public key
    .venv/bin/python bin/backup.py /mnt/ssd/vault/backups
"""

import datetime
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vault import config, db  # noqa: E402


def _seal(path: str, recipient: str) -> str:
    if shutil.which("age") is None:
        raise SystemExit("VAULT_BACKUP_RECIPIENT set but `age` is not on PATH")
    sealed = path + ".age"
    subprocess.run(["age", "-r", recipient, "-o", sealed, path], check=True)
    os.chmod(sealed, 0o600)
    os.remove(path)  # keep only the public-key-sealed copy off-site
    return sealed


def main() -> None:
    dest_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VAULT_BACKUP_DIR", "").strip()
    if not dest_dir:
        print("usage: python bin/backup.py <dir>   (or set VAULT_BACKUP_DIR)", file=sys.stderr)
        sys.exit(2)
    os.makedirs(dest_dir, exist_ok=True)
    os.chmod(dest_dir, 0o700)

    cfg = config.load()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(dest_dir, f"brain-{stamp}.db")
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists")

    conn = db.connect(cfg)
    if db.get_meta(conn, "canary") != "ok":
        raise SystemExit("vault not initialized or wrong key")
    # key_hex is validated to be exactly 64 hex chars (config), so inlining it
    # here cannot inject; the filename stays a bound parameter.
    conn.execute(f"ATTACH DATABASE ? AS backup KEY \"x'{cfg.key_hex}'\"", (out,))
    conn.execute("SELECT sqlcipher_export('backup')")
    conn.execute("DETACH DATABASE backup")
    conn.close()
    os.chmod(out, 0o600)

    recipient = os.environ.get("VAULT_BACKUP_RECIPIENT", "").strip()
    if recipient:
        out = _seal(out, recipient)
        print(f"sealed backup written: {out}  (needs the age private key + vault key to open)")
    else:
        print(f"encrypted backup written: {out}  (opens with the vault key)")


if __name__ == "__main__":
    main()
