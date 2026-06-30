# Key recovery runbook

The vault is encrypted with one 256-bit key (64 hex chars), printed once by
`bin/init.py`. Lose it and the data is unrecoverable. This is how to back it up with
split trust so no single location holds the whole key.

## Split the key (offline, audited tooling)

Do this on an offline machine. Use an audited Shamir implementation — do **not** roll
your own.

### Option A — SLIP-39 (mnemonic shares, recommended)

```bash
pip install shamir-mnemonic          # SatoshiLabs reference implementation
# 2-of-3 example: any 2 of 3 shares reconstruct the key
shamir create 2of3 --master-secret <64-hex-key>
```

Distribute the three word-list shares to three separate, secure locations.

### Option B — ssss (classic CLI)

```bash
sudo apt install ssss
echo -n <64-hex-key> | ssss-split -t 2 -n 3 -x
```

## Rules

- Threshold **2-of-3** (or 3-of-5): survives losing one share, needs two to steal.
  Never 1-of-N (just copies) or N-of-N (one loss = locked out forever).
- Store shares in **separate** places. Never beside `brain.db` or in `secrets.env`.
- Plain Shamir is unauthenticated — always verify a reconstruction before trusting it
  (next section).
- Wipe reconstructed material from shell history and clipboard afterward.

## Recover

Reconstruct the key from the shares, then verify it actually opens the vault:

```bash
# SLIP-39: shamir recover    (enter the shares)
export VAULT_DB_PATH=/home/pi/.vault/brain.db
python bin/recover.py <reconstructed-64-hex-key>
# -> "Key verified" means SQLCipher opened the DB and the canary matched.
```

Then set `VAULT_KEY` (in `secrets.env`) to the verified key and start the service.

## Rotate the key

SQLCipher can re-key in place. With the vault open under the current key:

```sql
PRAGMA rekey = "x'<new-64-hex-key>'";
```

After rotating, re-split the new key and securely destroy the old shares.
