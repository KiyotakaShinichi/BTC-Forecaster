# Production market-intelligence collection

The collector exists to accumulate one thing that cannot be recovered later: a
record of what was publicly available, and *when this system first saw it*.
`available_at` is the moment of retrieval. A document that was retrievable at
09:00 on a Tuesday cannot be given that availability afterwards, so every hour
the collector is not running is a permanent hole in the corpus that B5's Gate 1
is waiting for.

This is the runbook for running it on a host. It is operations, not research:
nothing here studies events, mines signals, tunes a threshold or trades.

## Status: repository ready, not deployed

| | State |
|---|---|
| **REPOSITORY READY** | **Yes.** Units, installer, configuration check, provider probe, health check, alerting, verified backups, restore rehearsal, a deterministic readiness status and a bounded smoke test all exist and are tested; CI runs them without a host. |
| **ACTUALLY DEPLOYED** | **No.** No always-on host, SSH access or operator contact address was available to the work that produced this package, so no production collection is running. The last successful collection cycle anywhere is 2026-09-03. |
| **Deployment verdict** | `DEPLOYMENT_BLOCKED_EXTERNALLY` — see the dry run at the end of this document. |
| **Gate 1** | `INTELLIGENCE_CORPUS_INSUFFICIENT`. The 180-day accumulation has **not** begun on a persistent host, let alone completed. |

Nothing in this document should be read as a claim that production collection
exists. It will be true when a host has been installed, `ops-smoke` has passed on
it, and the collector has been running since; the operator checklist in section
17 is how to get there.

---

## 1. Host requirements

| | |
|---|---|
| **Host** | A Linux machine that stays on. 1 vCPU, 1 GB RAM and 10 GB of disk is ample. **A laptop is not a host**: it sleeps, and a sleeping collector records a gap indistinguishable from a quiet week. |
| **Init** | systemd. Timers with `Persistent=true` are what carry collection across reboots. |
| **Python** | 3.11 or newer, with `venv`. |
| **Clock** | NTP-synchronised. Retrieval time *is* availability; a clock that drifts moves every document it collects. |
| **Disk** | Persistent, not tmpfs. The collector warns loudly on a root that looks ephemeral and refuses one it cannot write. |
| **Network** | Outbound HTTPS to `www.sec.gov`, `www.federalreserve.gov`, `www.cftc.gov`, `www.bls.gov`. Nothing listens. |
| **Access** | root once, to install. The service runs as an unprivileged `btc-intel` account with no shell and no home. |
| **Credentials** | **None.** Every feed is public. |
| **Contact** | A dedicated address a publisher can write to (section 3). Required. |

No GPU, no container runtime, no database server and no managed cloud service
is needed or used.

## 2. Installation

```sh
git clone https://github.com/KiyotakaShinichi/BTC-Forecaster /opt/btc-intel
cd /opt/btc-intel
git checkout <the reviewed commit>
sudo ./deploy/install.sh
```

The first run creates the service account, the virtualenv and
`/etc/btc-intel/collector.env` from `deploy/collector.env.example`, installs the
units, and then **stops before enabling any timer**: the configuration check
fails because no contact is set yet.

```sh
sudoedit /etc/btc-intel/collector.env      # set BTC_INTEL_CONTACT
sudo ./deploy/install.sh                   # now enables the four timers
```

What the installer does, in order: service account; code copied to the prefix
(never with `--delete`, so it cannot eat a state directory placed inside it);
virtualenv from `requirements-market-intelligence.txt`; code owned by root and
read-only to the service; the env file written once and never overwritten; the
deployed commit recorded as `BTC_INTEL_SOURCE_SHA`; units installed; a storage
preflight (`ops-paths`) and a configuration check (`ops-config-check`), both as
the service user; then the four timers. It is idempotent: re-running it updates
code and units and leaves everything else alone.

Then, before walking away, run the smoke test (section 17).

**Running a command as the service.** Every command below that needs the
contact or the state root runs in exactly the unit's environment -- its user,
its env file, its paths -- through `systemd-run`, so what it checks is what the
timers will do:

```sh
sudo systemd-run --pipe --wait --quiet -p User=btc-intel -p Group=btc-intel \
  -p EnvironmentFile=/etc/btc-intel/collector.env \
  -p Environment=PYTHONPATH=/opt/btc-intel -p Environment=BTC_INTEL_STATE_ROOT=/var/lib/btc-intel \
  /opt/btc-intel/.venv/bin/python /opt/btc-intel/btc-intel.py <command> [flags]
```

The rest of this document writes that prefix as `btc-intel-as-service`.

## 3. Environment configuration

`/etc/btc-intel/collector.env`, mode `0640`, owner `root:btc-intel`. Never
commit a filled-in copy; `.env` files are gitignored, and backups exclude any
`.env` file wherever it sits.

| Variable | | Purpose |
|---|---|---|
| `BTC_INTEL_CONTACT` | **required** | Goes into every feed request's User-Agent as `BTC-Forecaster Research <address>`. The SEC answers 403 to automated clients it cannot contact. The collector, the configuration check and the smoke test all refuse to run without it rather than advertise a placeholder. |
| `BTC_INTEL_ALERT_COMMAND` | optional | Run through the shell with each alert's message on standard input (section 10). May carry a webhook token. |
| `BTC_INTEL_STATE_ROOT` | set by the units | systemd's `StateDirectory`, `/var/lib/btc-intel`. Leave it alone. |
| `BTC_INTEL_DATABASE`, `BTC_INTEL_BACKUP_DIR`, `…_MANIFEST_DIR`, `…_CORPUS_DIR`, `…_LOG_DIR`, `…_LOCK_FILE` | optional | Split paths across disks, e.g. backups on a larger mount. |
| `BTC_INTEL_LOG_LEVEL` | optional | `INFO` by default. |
| `BTC_INTEL_SOURCE_SHA` | set by the installer | Recorded in every manifest. |

**What never leaves the host.** The contact address and the alert command are
treated as secrets: fields named like a secret are dropped from every log line,
and the *values* of these variables are scrubbed from every remaining field.
`ops-config-check`, `ops-probe` and `ops-status` show the User-Agent only in its
committed shape, `BTC-Forecaster Research <<redacted>>`. The configuration
fingerprint in each manifest records only *whether* a contact was configured.

## 4. Provider configuration

The deployed profile is `deploy/collection-profile.json` (`official-sources-v1`).
Feeds are selected by id from the committed catalogue in
`market_intelligence/collection/feeds.py`; adding a source is a reviewed code
change, never an edit on the host.

| Feed | Endpoint | Stream | State |
|---|---|---|---|
| `sec-press` | `https://www.sec.gov/news/pressreleases.rss` | regulatory announcement | active |
| `sec-admin-proceedings` | `https://www.sec.gov/rss/litigation/admin.xml` | administrative proceedings | active |
| `federalreserve-press` | `https://www.federalreserve.gov/feeds/press_all.xml` | regulatory announcement | active |
| `federalreserve-monetary` | `https://www.federalreserve.gov/feeds/press_monetary.xml` | monetary policy | active |
| `cftc-press` | `https://www.cftc.gov/RSS/RSSGP/rssgp.xml` | regulatory announcement | active |
| `bls-news` | `https://www.bls.gov/feed/bls_latest.rss` | economic statistics | active |
| `sec-litigation` | `https://www.sec.gov/rss/litigation/litreleases.xml` | civil litigation | **retired** 2026-09-03 (404); refused by the configuration check |
| `treasury-press` | `https://home.treasury.gov/rss/press.xml` | regulatory announcement | **retired** 2026-09-03; refused by the configuration check |

The provider contract, as `ops-config-check` prints it:

| | |
|---|---|
| User-Agent | `BTC-Forecaster Research <BTC_INTEL_CONTACT>`, on every request |
| Timeout | 20 s per request |
| Retries | 3 attempts per feed, and only for `TRANSIENT` and `RATE_LIMIT` failures; `AUTH`, `PERMANENT` (404, retired), `SCHEMA` and `CONTENT` are never retried |
| Backoff | exponential from 2 s, capped at 60 s, full jitter; rate-limit responses wait four times as long |
| Failure isolation | one unreadable feed costs only itself; a search in which *no* feed could be read is a failed attempt, never an empty success |
| Parsing | RSS 2.0 and Atom; strict parse first, then only a narrow bare-ampersand repair, and anything else is a `SCHEMA` failure |
| Cadence | `btc-intel-collect.timer`, every three hours at :07 plus up to five minutes' jitter; no provider is polled within 900 s (the declared floor) of its last success |

**Request volume, measured.** Each of the profile's 13 queries (its watch
entities' topics) searches every configured feed, so one cycle makes 78
requests: 26 each to `www.sec.gov` and `www.federalreserve.gov`, 13 each to
`www.cftc.gov` and `www.bls.gov`. At eight cycles a day that is 624 requests,
208 of them to the SEC — inside its published fair-access limit of ten requests
per second by several orders of magnitude, but far more than the "56 a day"
earlier documentation claimed, which counted one fetch per feed per cycle.
Fetching each feed once per cycle would cut it thirteen-fold without changing
what is collected; that is a change to the frozen collection path and is
recommended, not made here. Never raise the cadence to collect more: the
cadence is a research decision, not a throughput knob.

Before the first cycle, check every feed from the host, as the collector would
fetch it, storing nothing:

```sh
btc-intel-as-service ops-probe --profile /opt/btc-intel/deploy/collection-profile.json
```

One request per feed, no retries; exit 0 every feed readable, 1 some, 2 none. A
`PERMANENT` failure on an SEC feed almost always means the contact is missing or
not accepted.

## 5. systemd configuration

| Unit | When | Runs | Exit codes |
|---|---|---|---|
| `btc-intel-collect.timer` / `.service` | every 3 h at :07 | `collect-scheduled` | 0 ran; **2 failed — including a cycle that read nothing from any provider**; 3 another cycle holds the lock; 4 nothing due. 3 and 4 are success. |
| `btc-intel-verify.timer` / `.service` | daily 05:20 | `corpus-verify` | 0 sound, 2 corrupt |
| `btc-intel-backup.timer` / `.service` | daily 05:40 | `ops-backup --retain 30 --wait-seconds 600` | 0 archived and verified, 2 failed, 3 a cycle held the lock for the whole wait (success) |
| `btc-intel-watch.timer` / `.service` | hourly at :37 | `ops-watch --profile …` | 0 healthy, 1 warning (success, logged), 2 critical |
| `btc-intel-alert@.service` | on any failure above | `ops-alert --unit %i` | — |

Every timer is `Persistent=true`: a host that was off when one should have fired
runs it on the way back up. Every unit runs as `btc-intel` with
`NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp` and a
system-call filter; only the collector and the alert may open a network socket.
The watch runs at :37 so that it and a cycle do not reach for the database
together — DuckDB allows one writer per file — and the backup takes the
collector's run lock for the same reason.

**Why not Docker.** systemd already provides the scheduling, the reboot
survival, the sandboxing and the logs this needs. A container would add a
daemon, an image to rebuild and patch, and a volume to get wrong, and buy
nothing the units do not already give. The repository's Dockerfiles are for the
quantitative core and the API, not the collector.

## 6. Storage layout

```
/var/lib/btc-intel/              BTC_INTEL_STATE_ROOT (systemd StateDirectory)
  intelligence.duckdb            the corpus: documents, events, runs, attempts,
                                 watermarks, sightings, corrections, evidence
  manifests/                     one manifest per cycle, written last; configuration preimages
  corpus/                        corpus snapshots
  backups/
    corpus-YYYYMMDDTHHMMSSZ.tar.gz         verified archives
    corpus-YYYYMMDDTHHMMSSZ.tar.gz.sha256  `sha256sum -c` checksums
    backup-status.json                     the last backup attempt, read by the health check
  logs/collect-*.json            what each scheduled invocation decided
  collector.lock                 present only while a cycle or a backup runs
```

**Growth.** The corpus grows by documents, not by cycles: a cycle that finds
nothing new writes a run record and a few attempt rows. `ops-status --json`
reports measured bytes per document and per event and a linear projection;
expect megabytes a month, and plan the disk around thirty backup archives, each
roughly the database's size.

**Disk thresholds.** The health check warns below 1 GiB free on the state root
and is critical below 256 MiB; the collector itself refuses to start below
64 MiB. A warning is days of notice.

**Off-host copies.** A backup on the same disk survives a mistake, not a dead
disk:

```sh
# as root, e.g. from a daily timer on another host
rsync -a btc-intel-host:/var/lib/btc-intel/backups/ /srv/btc-intel-backups/
cd /srv/btc-intel-backups && sha256sum -c corpus-*.sha256
```

## 7. Backup

`btc-intel-backup.service` runs `ops-backup` daily at 05:40:

1. takes the collector's run lock, waiting up to ten minutes for a running cycle;
2. writes `corpus-<UTC stamp>.tar.gz` — the database (DuckDB's own consistent
   copy, never a copy of the open file), the manifests, and a manifest of its own
   with every member's sha256, the counts and a **content fingerprint** of the
   document, event and correction ids, all taken from the archived copy;
3. writes `<archive>.sha256`;
4. **proves the archive by restoring it** into a scratch location: checksum,
   member hashes, the integrity check on the restored corpus, counts and
   fingerprint against the manifest — then deletes the rehearsal. The live corpus
   is never opened by the rehearsal;
5. only if that passed, prunes to the **30 newest** scheduled archives. An
   archive named by hand (`corpus-before-upgrade.tar.gz`) is never pruned, the
   newest is never pruned, and nothing is pruned after a failed verification;
6. records the attempt in `backups/backup-status.json`, which the health check
   reads: a failed attempt is `BACKUP_FAILURE`, critical, the same day.

Credentials, `.env` files, locks and temporary files never enter an archive.

## 8. Restore

**Rehearse** — any time, touching nothing:

```sh
btc-intel-as-service corpus-backup-verify --latest
# or, on another machine: btc-intel corpus-backup-verify --archive corpus-20270101T054000Z.tar.gz
```

Exit 0 sound, 2 not, with every finding named.

**Restore for real** — only when the live corpus is lost or proven damaged:

1. `systemctl stop btc-intel-collect.timer btc-intel-backup.timer btc-intel-watch.timer`
2. Rehearse the archive you intend to use (above).
3. Restore it into a **new** directory — `corpus-restore` refuses a non-empty
   destination:
   `btc-intel corpus-restore --archive <archive> --destination /var/lib/btc-intel.restored`
4. Verify it there: `btc-intel --db /var/lib/btc-intel.restored/intelligence.duckdb corpus-verify`,
   and compare `ops-status` counts with the archive's manifest.
5. Move the damaged state root aside — **keep it**; it is evidence about the
   failure — and move the restored one into place, owned by `btc-intel`.
6. Start the timers and run `ops-smoke` (section 17).

Everything collected between the archive and the loss is gone and cannot be
re-collected with its original availability. The coverage record will show the
gap; do not backfill it.

## 9. Health check

One command, non-zero when the collector is unhealthy:

```sh
btc-intel-as-service --db /var/lib/btc-intel/intelligence.duckdb ops-watch \
  --profile /opt/btc-intel/deploy/collection-profile.json
```

0 healthy, 1 warning, 2 critical. What it detects:

| Code | Severity | When |
|---|---|---|
| `COLLECTION_STALE` | critical | no successful cycle for 6 h (the scheduler stopped, the host slept, every cycle failed) |
| `ALL_PROVIDERS_FAILED` | critical | the last cycle read nothing from any provider |
| `STORAGE_FAILURE` | critical | the state root cannot be written, or is below the collector's own minimum |
| `DISK_LOW` | warning < 1 GiB, critical < 256 MiB | free space on the state root |
| `CONFIGURATION_INVALID` | critical | the deployed profile no longer loads (missing contact, unknown feed, bad JSON) |
| `CORPUS_INTEGRITY_FAILURE` | critical (warning if only degraded) | the corpus no longer satisfies its invariants |
| `BACKUP_FAILURE` | critical | the last backup attempt failed |
| `BACKUP_STALE` | warning | no backup for 7 days |
| `PROVIDER_STALE` | warning | one provider silent for 2 days while others work |
| `QUARANTINE_SPIKE` | warning | 25 records quarantined in 24 h |
| `IMPLAUSIBLE_ZERO_COLLECTION` | warning | 14 consecutive days of successful cycles that returned no documents |
| `LOCK_STALE_BROKEN` | info | a previous cycle did not exit cleanly and its lock was broken |

**No events is not unhealthy.** A quiet day, and a fresh deployment whose feeds
were all read and nothing yet matched, are both `COLLECTION_HEALTHY`.

`ops-status` answers the rest of an operator's questions in one place: when the
last run was, its status, documents and events accepted and rejected, which
providers failed, collection coverage and lag, free disk, integrity, backup age
and the last backup attempt, and the watchdog's verdict.

## 10. Watchdog and alerting

`btc-intel-watch.timer` runs the health check hourly. A warning is logged to the
journal and waits for the morning; a critical fails the unit.

Every unit that can fail — collect, verify, backup, watch — names
`btc-intel-alert@.service` as its `OnFailure`. The alert:

- is always written to the journal at `CRITICAL` as one JSON record
  (`journalctl -u 'btc-intel-alert@*'`);
- if `BTC_INTEL_ALERT_COMMAND` is set, runs it with the message on standard
  input. Examples: `mail -s "btc-intel alert" ops@example.org`, or
  `curl -fsS -X POST --data-binary @- https://hooks.example.org/<token>`. A failing
  command is reported by exit code only; its text is never printed.

How the operator hears each failure:

| Failure | Alert |
|---|---|
| The collector stops (timer disabled, host slept, process wedged) | `COLLECTION_STALE` from the next hourly check after 6 h |
| Every feed fails | at once: the cycle exits 2 and its unit fails; `ALL_PROVIDERS_FAILED` on the next check |
| The watchdog finds stale collection | the watch unit fails with `COLLECTION_STALE` |
| A backup fails or does not verify | at once: the backup unit fails; `BACKUP_FAILURE` on the next check |
| The disk becomes low | `DISK_LOW` warning in the journal below 1 GiB; an alert below 256 MiB |
| The corpus fails verification | at once: the verify unit fails |

**The one thing a host cannot report is its own death.** If the machine is off,
nothing on it alerts. Pair the collector with an external dead-man's switch if
that matters: any heartbeat service the operator already uses, pinged from a
timer — outside this repository by design.

Test the path end to end after installing:

```sh
sudo systemctl start btc-intel-alert@manual_test.service
```

## 11. Logs

- `journalctl -u btc-intel-collect -o cat | jq` — one JSON object per line per
  event: `collection_started`, `provider_result`, `collection_finished` with the
  run status and counts; a failed cycle is logged at `ERROR`.
- `journalctl -u btc-intel-backup -o cat | jq` — `backup_finished` or `backup_failed`.
- `journalctl -u 'btc-intel-alert@*'` — every alert raised.
- `/var/lib/btc-intel/logs/collect-*.json` — what each scheduled invocation
  decided and why, kept beside the corpus.

No credential, contact address or alert command reaches any of these.

## 12. Troubleshooting

| Symptom | Likely cause | Do |
|---|---|---|
| collect unit failed, exit 2 | a cycle read nothing from any provider | `ops-status` → `last_run.failed_providers`; `ops-probe` from the host |
| `ops-probe`: SEC feeds `PERMANENT` / 403 | contact missing or refused | set a real, dedicated `BTC_INTEL_CONTACT`; `ops-config-check` |
| `ops-probe`: a feed `PERMANENT` / 404 | the publisher moved or retired it | report it; a feed change is a reviewed change to `feeds.py` |
| `CONFIGURATION_INVALID` | profile or env file edited on the host | restore the committed profile; `ops-config-check` |
| collect exits 4 every time | nothing due yet: the 900 s floor | normal |
| collect exits 3 repeatedly | a wedged cycle holds the lock | it is broken as stale after 30 min and reported as `LOCK_STALE_BROKEN`; check the journal for the wedged run |
| `BACKUP_FAILURE` | archive or verification failed | `journalctl -u btc-intel-backup`; `corpus-backup-verify --latest`; do not prune by hand |
| `DISK_LOW` | disk filling | move backups to a larger mount with `BTC_INTEL_BACKUP_DIR`, or copy older archives off-host and delete them |
| zero documents for days | a quiet period, or a feed returning nothing | `IMPLAUSIBLE_ZERO_COLLECTION` after 14 days; `ops-probe` shows each feed's entries |
| `corpus-verify` exits 2 | integrity failure | stop the timers first; section 8 |

## 13. Upgrade procedure

```sh
sudo systemctl start btc-intel-backup.service        # a fresh, verified backup first
cd /opt/btc-intel && git fetch && git checkout <new sha>
sudo ./deploy/install.sh                              # re-runs the config check before touching timers
btc-intel-as-service ops-smoke --profile /opt/btc-intel/deploy/collection-profile.json --no-backup
```

The installer does not stop the timers and does not need to: the lock means a
cycle in flight finishes on the old code and the next starts on the new.

## 14. Rollback

**Code rolls back; the corpus does not.** Check out the previous commit and
re-run the installer; it never touches `/var/lib/btc-intel`. Documents collected
in between are exactly the ones that cannot be collected again, so a rollback
must never discard them.

Rolling back past B5.1 would put the collector back on the `rules-v1` extractor.
Its events would be written beside the `rules-v2` ones — never pooled, because
event identity carries the extractor version — but they would not be countable by
Gate 1. Do not roll back past B5.1 without a reason written down.

If a release wrote bad *data*, not just behaved badly: stop the timers, run
`corpus-verify --json`, find the runs from their manifests' `source_sha`, and
correct through the append-only ledger (`corrections/README.md`). Never delete.

## 15. Secret rotation

- **Contact address.** Edit `BTC_INTEL_CONTACT` in the env file. The next cycle
  uses it; nothing else changes, because no manifest records the address itself.
  Then `ops-config-check` and `ops-probe`.
- **Alert command or its token.** Edit `BTC_INTEL_ALERT_COMMAND`; test with
  `systemctl start btc-intel-alert@manual_test.service`.
- **Licensed-provider credentials.** None are in use. If a contract ever adds
  one, it goes in the env file, and naming it does not enable the provider — the
  profile must too.

Keep the file `0640 root:btc-intel`. Nothing needs restarting: every unit is a
oneshot that reads the file when it starts.

## 16. Disaster recovery

The host is lost:

1. Provision a new host (section 1) and install (section 2), but **do not** let
   the installer enable timers against an empty state root yet — leave
   `BTC_INTEL_CONTACT` unset until step 4.
2. Take the newest off-host archive, check it (`sha256sum -c`), and rehearse it
   (`corpus-backup-verify --archive …`).
3. Restore it into `/var/lib/btc-intel` while that directory is still empty
   (section 8, step 3), owned by `btc-intel`.
4. Set the contact, re-run the installer, run `ops-smoke`.

The time between the last archive and the new host is a real collection gap. The
coverage record shows it as missed days; it must not be filled in. A daily
off-host copy bounds the loss to a day.

## 17. Operator checklist

**Before installing**
- [ ] A Linux host with systemd that stays on, NTP-synchronised, with persistent disk
- [ ] A dedicated contact address for publishers
- [ ] Somewhere alerts will actually be read, and an off-host place for backups

**Installing**
- [ ] `sudo ./deploy/install.sh`; set `BTC_INTEL_CONTACT`; run it again
- [ ] `btc-intel-as-service ops-config-check --profile /opt/btc-intel/deploy/collection-profile.json` — deployable
- [ ] `btc-intel-as-service ops-probe --profile …` — every feed readable
- [ ] `btc-intel-as-service ops-smoke --profile …` — no check `FAIL` (`NOT_OBSERVED` on a quiet week is fine)
- [ ] `systemctl list-timers 'btc-intel-*'` — four timers scheduled
- [ ] `systemctl start btc-intel-alert@manual_test.service` — the alert arrives
- [ ] off-host `rsync` of `backups/` scheduled

**Daily** — nothing, unless an alert arrives.

**Weekly**
- [ ] `btc-intel-as-service --db /var/lib/btc-intel/intelligence.duckdb ops-status --profile …` — last run, coverage, lag, backup
- [ ] `python -m market_intelligence.b5 status --db /var/lib/btc-intel/intelligence.duckdb --as-of <now, UTC>` — progress toward Gate 1

**Monthly**
- [ ] rehearse a restore of an off-host archive on another machine

**After every upgrade** — `ops-smoke --no-backup`.

**Gate 1.** Re-run it only when `b5 status` shows a family that could plausibly
pass — never on a schedule, never with a lowered threshold. The 180-day span
starts when a persistent host starts collecting; it has not started yet.

---

## Collection readiness status

```sh
python -m market_intelligence.b5 status \
  --db /var/lib/btc-intel/intelligence.duckdb --as-of 2027-03-10T00:00:00+00:00
```

Read-only and deterministic: the instant is an argument, never the clock. It
reports the elapsed collection span, expected and successful days and cycles,
coverage, documents, events at each stage of the funnel (raw, eligible, above
the quality floor, independent, effective), publishers and providers, the
longest family span, and Gate 1's status with every unmet clause — all from the
audit Gate 1 itself runs, under the committed preregistration. It restates no
threshold. It says "passed" only when Gate 1 passed; until then it ends *"The
corpus is not sufficient."*

---

## Dry run, 2026-09-11: what was verified without a host

No always-on host, SSH access or operator contact address was available, so the
package could not be deployed and the verdict is **`DEPLOYMENT_BLOCKED_EXTERNALLY`**.
Instead every operator command that needs no network was run against a **copy**
of the collected store, in a scratch state root, with the code at `99fb45b` --
the final code; the one later commit changes only documentation and a unit-file
comment. The original store's sha256 was `fe9ae0a0…` before and after. No
request left the machine.

| Step | Result |
|---|---|
| `ops-config-check`, no contact set | **not deployable**, exit 2, naming `BTC_INTEL_CONTACT` -- the installer's gate would hold every timer back |
| `ops-status --profile` | last run `DEGRADED`, 2026-09-03 09:47Z: 3 documents accepted, 7 rejected, 1 event; 2 of 10 elapsed days collected (20%), 5 of 74 cycles; lag median 16.5 h, p90 19.2 h; integrity OK; configuration invalid; health critical |
| `ops-watch --profile` | exit 2: `CONFIGURATION_INVALID` and `COLLECTION_STALE` (8 days), with `PROVIDER_STALE` and `BACKUP_STALE` warnings -- the right verdict for a collector that stopped on 2026-09-03 and has no contact |
| `b5 status`, as of 2026-09-11 | 6 documents, 7 events, 0 above the quality floor; 0 of 15 families ready; the longest family spans 0 of the 180 days required; `INTELLIGENCE_CORPUS_INSUFFICIENT` |
| `ops-backup` | archived and proved by restoring it: 6 documents, 7 events, fingerprint `e054651e…`, integrity OK, checksum matches |
| `corpus-backup-verify --latest` | sound: counts and fingerprint match the manifest, integrity OK |
| `ops-watch` after the backup | the backup warning cleared; still critical for staleness and configuration |
| `ops-alert`, no command configured | written to the log at CRITICAL, delivered, exit 0 |
| offline smoke suite | 9 passed: `ops-smoke` end to end on a healthy, a quiet and a failing feed, and the run lock held across real processes -- refused while alive, recovered once killed |
| request volume | one cycle of the deployed profile makes 78 requests (section 4) |

What a dry run cannot show, and the first things to check on a real host:

- that every feed answers *this* host, with *this* contact -- `ops-probe`;
- that the host's systemd accepts the units -- `systemctl daemon-reload`,
  `systemctl list-timers 'btc-intel-*'`;
- that a real cycle stores, extracts and backs up on the host -- `ops-smoke`;
- that an alert reaches a person -- `systemctl start btc-intel-alert@manual_test.service`;
- that collection runs for 180 days at 80% coverage -- which only running it can
  show, and which has not begun.
