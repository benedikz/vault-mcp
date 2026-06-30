"""Run one consolidation pass — the nightly "sleep" pass.

Rebuilds semantic and link edges from stored vectors, decays/prunes stale
Hebbian edges, flags near-duplicates, and recomputes salience. Operates purely
on the database (no embedding model is loaded), so it is cheap to schedule.

    export VAULT_KEY=... VAULT_DB_PATH=... VAULT_TRANSPORT=stdio
    .venv/bin/python bin/consolidate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vault import config, consolidate, db  # noqa: E402


def main() -> None:
    cfg = config.load()
    conn = db.connect(cfg)
    if db.get_meta(conn, "canary") != "ok":
        raise SystemExit("vault is not initialized")
    stats = consolidate.consolidate(conn)
    conn.close()
    print(stats)


if __name__ == "__main__":
    main()
