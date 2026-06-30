import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vault import db  # noqa: E402

_KEY_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")


def main() -> None:
    if len(sys.argv) != 2 or not _KEY_RE.match(sys.argv[1].strip()):
        print("usage: python bin/recover.py <64-hex-key>", file=sys.stderr)
        sys.exit(2)

    key_hex = sys.argv[1].strip().lower()
    db_path = os.environ.get("VAULT_DB_PATH", "").strip() or os.path.expanduser("~/.vault/brain.db")
    if not os.path.exists(db_path):
        print(f"no database at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = db.connect_raw(db_path, key_hex)
    if db.get_meta(conn, "canary") != "ok":
        print("database opened but canary missing — unexpected state", file=sys.stderr)
        sys.exit(1)
    conn.close()
    print("Key verified. Set it with:  export VAULT_KEY=" + key_hex)


if __name__ == "__main__":
    main()
