# Running the collector continuously — operations guide

Vendor-neutral. Nothing here assumes a hosting provider, and nothing below was
tested on one: the examples are shapes to adapt, and the only environments this
has actually run in are a local Windows host and CI.

## What it needs

| resource | minimum | why |
| --- | --- | --- |
| CPU | 1 vCPU | a cycle is I/O-bound; extraction is rule-based by default |
| RAM | 512 MB | DuckDB plus one cycle's documents |
| Disk | 5 GB to start | see the storage projection below |
| **Persistent volume** | required | the one thing that must not be ephemeral |
| Outbound HTTPS | required | to the feeds you configure, nothing else |
| Scheduler | cron, systemd timer, or any container scheduler | no always-on process |

Python 3.11+ and the market-intelligence requirements. No Kubernetes, no message
broker, no external database.

## The one command a scheduler calls

```
btc-intel collect --config providers.json --origin "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                  --manifest "$BTC_INTEL_STATE_ROOT/manifests/collect-$(date -u +%Y%m%dT%H%M%SZ).json"
```

It takes the run lock, validates storage, runs one cycle, and writes the manifest
last. Exit codes: `0` ran, `2` failed, `3` another collector holds the lock
(normal — the scheduler fired during a long cycle), `4` nothing was due under the
configured cadence.

`3` and `4` are not failures. A scheduler that treats them as errors trains
operators to ignore the collector's exit code, which is the state you least want
to be in on the day it returns `2`.

### cron

```cron
# Collect twice an hour; the cadence floor decides what is actually due.
*/30 * * * *  cd /srv/btc-intel && ./venv/bin/btc-intel collect ... >> "$STATE/logs/collect.log" 2>&1
# Health, once an hour. Exit 2 is worth waking someone.
17 * * * *    cd /srv/btc-intel && ./venv/bin/btc-intel ops-watch --json || true
# Verify and back up nightly.
30 3 * * *    cd /srv/btc-intel && ./venv/bin/btc-intel corpus-verify --json
45 3 * * *    cd /srv/btc-intel && ./venv/bin/btc-intel corpus-backup --output "$STATE/backups/corpus-$(date -u +%Y%m%d).tar.gz"
```

### systemd timer

```ini
# btc-intel-collect.service
[Service]
Type=oneshot
WorkingDirectory=/srv/btc-intel
Environment=BTC_INTEL_STATE_ROOT=/var/lib/btc-intel
EnvironmentFile=/etc/btc-intel/secrets.env   # credentials live here, not in the unit
ExecStart=/srv/btc-intel/venv/bin/btc-intel collect ...
SuccessExitStatus=0 3 4                      # lock-held and nothing-due are not failures
```

```ini
# btc-intel-collect.timer
[Timer]
OnCalendar=*:0/30
Persistent=true      # catch up one missed run after a reboot
RandomizedDelaySec=300
```

`Persistent=true` matters: without it a host that was down over a maintenance
window silently skips those cycles, and the gap only shows up weeks later as a
coverage hole in a study.

### Container scheduler

Run the same command as a scheduled job with the state volume mounted. The image
needs no entrypoint beyond the CLI, and the container should exit after each
cycle — a long-lived container adds a supervision problem the lock already
solves.

## Persistent state

Everything lives under one root, resolved from `BTC_INTEL_STATE_ROOT`:

```
$BTC_INTEL_STATE_ROOT/
  intelligence.duckdb     the corpus: documents, events, sightings, watermarks, catalogs
  manifests/              one immutable manifest per collection run
  corpus/                 exported corpus artifacts
  backups/                verifiable archives
  logs/                   whatever your scheduler redirects here
  collector.lock          the run lock
```

Individual paths override one at a time: `BTC_INTEL_DATABASE`,
`BTC_INTEL_MANIFEST_DIR`, `BTC_INTEL_CORPUS_DIR`, `BTC_INTEL_BACKUP_DIR`,
`BTC_INTEL_LOG_DIR`, `BTC_INTEL_LOCK_FILE`.

`btc-intel ops-paths` prints the resolved paths and proves each is writable by
writing to it. **Run it once after any deployment change.** `exists()` and
`os.access` both pass on a read-only mount, a full volume and a stale NFS
handle; only a real write catches those, and catching them at startup is much
cheaper than catching them mid-cycle.

It also warns when the root looks ephemeral. That check exists because the most
expensive misconfiguration in this system produces no error at all: a container
writing to a layer that is discarded on the next deploy collects perfectly for
months and accumulates nothing.

## Docker with persistent state

```dockerfile
FROM python:3.11-slim
WORKDIR /srv/btc-intel
COPY requirements-market-intelligence.txt .
RUN pip install --no-cache-dir -r requirements-market-intelligence.txt
COPY market_intelligence ./market_intelligence
COPY btc-intel.py .
ENV BTC_INTEL_STATE_ROOT=/data
VOLUME ["/data"]
ENTRYPOINT ["python", "btc-intel.py"]
```

```bash
docker volume create btc-intel-state

# One collection cycle
docker run --rm -v btc-intel-state:/data --env-file /etc/btc-intel/secrets.env \
  btc-intel collect --config /data/providers.json --origin "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --manifest /data/manifests/collect.json

# Health, verification, backup
docker run --rm -v btc-intel-state:/data btc-intel --db /data/intelligence.duckdb ops-watch
docker run --rm -v btc-intel-state:/data btc-intel --db /data/intelligence.duckdb corpus-verify
docker run --rm -v btc-intel-state:/data btc-intel --db /data/intelligence.duckdb \
  corpus-backup --output /data/backups/corpus.tar.gz
```

A named volume, not a bind mount into the image, and not a container-local path.
`VOLUME ["/data"]` documents the intent; it does not enforce it, which is why
`ops-paths` exists.

## Cadence

Each provider declares a `minimum_interval_seconds` this system will not poll
faster than — 900 s for public feeds, 1800–3600 s for contracted APIs. That is a
floor imposed on ourselves, not the provider's published limit: research needs
one pass every few hours, and polling harder buys nothing and risks the
relationship with the source.

Run the scheduler more often than the floor. The collector skips what is not due
and exits `4`.

## Monitoring

| command | answers | exit codes |
| --- | --- | --- |
| `ops-status` | last run, providers, 24h counts, coverage gaps, corpus, backup age, integrity, readiness | 0 |
| `ops-watch` | the alerts only, as JSON | 0 ok, 1 warning, 2 critical |
| `corpus-verify` | is the corpus internally consistent | 0 ok, 2 corrupt |
| `corpus-status` | what the corpus holds and whether B4 can be re-run | 0 |

Wire `ops-watch` to whatever you already have. Its output is plain JSON with
stable alert codes; no vendor is assumed, and matching on the code rather than
the message is the supported contract.

**A quiet day is not a failure.** Regulator feeds produce nothing
bitcoin-related most days. The watchdog alerts on providers *failing*, not on
documents being absent — and separately on the implausible case where providers
succeed and return nothing for a fortnight, because that is what a silent
collection defect looks like from outside.

## Backup and restore

```bash
btc-intel --db "$STATE/intelligence.duckdb" corpus-backup --output "$STATE/backups/corpus-$(date -u +%Y%m%d).tar.gz"
btc-intel corpus-restore --archive .../corpus-20260901.tar.gz --destination /tmp/restore-check
```

Restore writes to a **new** location and refuses a non-empty one. The operation
you want when checking an archive is "prove this is good", not "replace the
corpus with it". It re-hashes every member against the archive's own manifest
and compares the restored counts, so a truncated transfer is caught before
anyone builds on it.

Restore a backup somewhere harmless on a schedule. An archive nobody has
restored is a hypothesis.

## Retention

Research-critical point-in-time metadata is never purged. If raw-content
retention is limited for licensing reasons, the provenance and hash that
establish availability and source identity survive regardless — the hash is what
lets a holder of the original prove it is the one this corpus used.

## Secrets

Environment variables or a secret store. Never in the repository, the manifests,
the backups, the daily summary or an API response. `btc-intel providers` reports
whether a credential is *present*, never its value.

## Storage

`ops-status --json` reports measured bytes per document, per event and per day,
with 30/180/365-day projections. They are linear extrapolations from what is
actually stored and are labelled as such: they exist to size a volume, not to
justify compressing anything.

## What this does not do

No forecasting, no signal fusion, no BTC-return statistics. The dashboard shows
`NOT_READY` until B4's frozen adequacy criteria are met, and shows no
preliminary effect estimates before then — a number on the morning page would
make ignoring the readiness gate the path of least resistance.
