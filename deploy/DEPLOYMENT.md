# Deploying the forward intelligence collector

This deploys one thing: a process that wakes up every three hours, fetches seven
government feeds, and writes down what it saw and *when it saw it*. Nothing here
forecasts, scores, or trades.

The reason it needs a deployment at all — rather than someone running a script
when they remember to — is that the corpus is built from `available_at`, and
`available_at` is the moment of retrieval. A document that was retrievable at
09:00 on a Tuesday in March cannot be given that availability later. Miss the
window and it is gone; the study you wanted to run in six months quietly loses
the evidence it needed, and nothing about the resulting dataset looks wrong.

That asymmetry drives every choice below.

This file is the short version. The full production runbook -- configuration
check, provider probe, health check and alerting, verified backups, restore,
upgrade, rollback, secret rotation, disaster recovery and an operator checklist
-- is [`docs/production-collection.md`](../docs/production-collection.md).

---

## What you need before starting

| | |
|---|---|
| **Host** | A Linux machine that stays on. Small: 1 vCPU, 1 GB RAM, 10 GB disk is ample. |
| **Init** | systemd. Timers are what make this survive reboots. |
| **Python** | 3.11 or newer, with `venv`. |
| **Access** | root, or sudo, once — to install the units. |
| **Credentials** | **None.** All seven feeds are public. |

A laptop is not a host. It sleeps, and the corpus records a gap that is
indistinguishable from a quiet news week. See *Why not a laptop* below.

## Install

```sh
git clone <this repo> /opt/btc-intel
cd /opt/btc-intel
sudo ./deploy/install.sh
```

The installer is idempotent — re-running it updates code and units and leaves
everything else alone. It refuses to enable the timers if the state directory is
not provably writable, because a collector that runs and cannot write looks
exactly like a collector that ran and found nothing.

### Set a contact address first

Set `BTC_INTEL_CONTACT` in `/etc/btc-intel/collector.env` to a dedicated project
address a publisher can reply to:

```sh
BTC_INTEL_CONTACT=btc-forecaster@example.org
```

Every feed request then identifies itself as
`BTC-Forecaster Research <btc-forecaster@example.org>`. The address lives on the
host and never in the repository — the profile carries only the shape. The
collector refuses to start if the profile asks for a contact and the variable is
unset, rather than advertising an unexpanded `${BTC_INTEL_CONTACT}`, which looks
like a contact and is not one.

This is not decoration. The SEC's access policy asks automated clients to
identify themselves with a contact, and answers `403` to those that do not. A
single manual fetch usually slips through; a collector polling eight times a day
for months is precisely what gets blocked — and it would arrive weeks in, as a
`PERMANENT` failure class on a feed everybody assumed was working. The profile
refuses a `user_agent` with no contact in it rather than letting you deploy a
string that cannot be answered.

### Corrections

If an observation turns out to be an artefact of a defect rather than a fact
about the world, it is invalidated, never deleted. See
[`corrections/README.md`](../corrections/README.md).

```sh
btc-intel --db <corpus> corpus-corrections     # what is excluded, and why
```

## Verify it is actually running

```sh
systemctl list-timers 'btc-intel-*'          # when does it next fire
systemctl start btc-intel-collect.service    # run one cycle now
journalctl -u btc-intel-collect.service -n 50
sudo -u btc-intel /opt/btc-intel/.venv/bin/python \
    /opt/btc-intel/btc-intel.py ops-status --json
```

`ops-status` is the one to read. It reports what has been collected, whether
storage is healthy, how old the last backup is, and how far the corpus is from
being able to support a B4 event study. That last number will say `NOT_READY`
for months. That is the expected state, not a fault.

## Exit codes

The exit code is the entire interface between the collector and the scheduler.

| Code | Meaning | Is it a problem? |
|---|---|---|
| `0` | A cycle ran. | No |
| `2` | The cycle failed -- including one that ran and read nothing from any provider. | **Yes** |
| `3` | Another cycle holds the lock. | No |
| `4` | Nothing was due yet. | No |

`3` and `4` are the collector working correctly, which is why the unit carries
`SuccessExitStatus=3 4`. Configure any other scheduler the same way. A unit that
treats them as failures mails an operator every night until they stop reading
the mail — and the one night it matters, they will not read that one either.

## What runs, and when

| Unit | Cadence | Purpose |
|---|---|---|
| `btc-intel-collect.timer` | every 3h, at :07 | one collection cycle |
| `btc-intel-verify.timer` | daily, 05:20 | corpus integrity, fails closed |
| `btc-intel-backup.timer` | daily, 05:40 | archive, proved by restoring it; keeps the newest 30 |
| `btc-intel-watch.timer` | hourly, :37 | health check; a critical raises an alert |

All four carry `Persistent=true`, so a host that was off when a timer should
have fired runs it on the way back up instead of skipping the day. Every
service raises `btc-intel-alert@.service` when it fails, which writes the
alert to the journal and runs `BTC_INTEL_ALERT_COMMAND` if one is set.

### Why three hours

The cadence is decided once and cannot be revisited, because it sets a floor
under how sharply any future event study can place an announcement:

| cadence | mean lag | worst | share of a 24h horizon | cycles/day | requests/day |
|---|---|---|---|---|---|
| 12h | 6.0h | 12h | 25% | 2 | 156 |
| 6h | 3.0h | 6h | 12.5% | 4 | 312 |
| **3h** | **1.5h** | **3h** | **6.2%** | **8** | **624** |
| 1h | 0.5h | 1h | 2.1% | 24 | 1,872 |

A 168-hour horizon tolerates any of these. A 24-hour horizon does not: at twelve
hours a quarter of the horizon is retrieval lag.

Each cycle searches every feed once per watchlist query: the deployed profile's
13 queries against its 6 feeds are 78 requests a cycle, 26 of them to
`www.sec.gov` -- measured, not estimated (B5.2). Earlier versions of this table
counted one fetch per feed per cycle and said 56 a day; the true figure is
thirteen times that. It remains far inside every publisher's published limits --
the SEC's fair-access limit is ten requests a *second* -- and inside the
900-second floor the provider declares, so the politeness argument still does
not buy back what a coarser cadence costs. Fetching each feed once per cycle
would cut it thirteen-fold without changing what is collected; that is a change
to the frozen collection path and has not been made.

To change it, edit `OnCalendar` in `btc-intel-collect.timer` and re-run the
installer. Going *below* 900 seconds is refused — the provider declaration is
the contract, and if that is wrong the argument belongs in `feeds.py`.

## Backups

The corpus is the only thing on this host that cannot be rebuilt. Code can be
cloned again and a host reinstalled from scratch; the availability record cannot
be reconstructed at any price.

`btc-intel-backup.timer` writes a verifiable archive daily. **A backup on the
same disk as the corpus is not a backup** — it survives a mistake, not a dead
disk. Copy them off the host:

```sh
# in root's crontab, or as another timer
rsync -a /var/lib/btc-intel/backups/ backup-host:/srv/btc-intel-backups/
```

### Rehearsing a restore

Restore into a **new** directory and verify it there. Never restore over a live
corpus to find out whether the archive was good:

```sh
btc-intel corpus-restore \
    --archive /var/lib/btc-intel/backups/corpus-<stamp>.tar.gz \
    --destination /tmp/restore-rehearsal
```

`corpus-restore` refuses a destination that already holds a corpus, so the
dangerous version of this command is not available to a tired operator at 3am.

## Updating

```sh
cd /opt/btc-intel
git fetch && git checkout <new-sha>
sudo ./deploy/install.sh
```

The installer copies code, refreshes units, reloads systemd and re-runs the
storage preflight. It does not stop the timers first, and does not need to: the
lock means a cycle already running finishes on the old code, and the next one
starts on the new.

Take a backup first anyway. It costs seconds.

## Rolling back

**Code rolls back. The corpus does not.**

```sh
cd /opt/btc-intel
git checkout <previous-sha>
sudo ./deploy/install.sh
```

That is the whole procedure, and it is deliberately unable to touch
`/var/lib/btc-intel`. Reverting to yesterday's code must never revert to
yesterday's evidence: the documents collected in between are exactly the ones
that cannot be collected again, and a rollback that quietly discarded them would
destroy the thing this deployment exists to accumulate.

If a bad release wrote *bad data* — not merely behaved badly — that is a
different problem, and the corpus's own history is the tool for it:

1. Stop the timers: `systemctl stop btc-intel-collect.timer`
2. Establish what is wrong: `btc-intel corpus-verify --json`
3. Find which runs produced it: manifests in `/var/lib/btc-intel/manifests`,
   each naming its `source_sha` and a `configuration_fingerprint` whose preimage
   is stored beside it.
4. Restore the last known-good archive **into a new directory**, confirm it,
   then swap deliberately, with the damaged corpus kept.

Never delete the damaged corpus until the replacement has been verified. It is
evidence about the bug as well as about the world.

## Why not a laptop

A laptop sleeps, closes, travels and reboots for updates. Every one of those is
a hole in the availability record, and a hole is invisible afterwards: a week
with no documents reads as a quiet week, not as a laptop that was in a bag.
`Persistent=true` recovers a missed cycle on a machine that comes back within
the hour; it cannot recover three days.

If a laptop is genuinely all that is available, deploy anyway — a gappy corpus
beats no corpus — but record the machine's uptime alongside it, so the gaps can
later be attributed to the collector rather than to the world.

## Troubleshooting

**Timer active, nothing collected.** Almost always exit `4`: the cadence floor
has not elapsed. `journalctl -u btc-intel-collect.service` says so explicitly.

**Cycles succeed and collect nothing, for days.** This is the failure mode worth
being suspicious of, because a green run with zero documents is indistinguishable
from a quiet news day. `ops-watch` raises `IMPLAUSIBLE_ZERO_COLLECTION` after a
run of them. Check `ops-status` for per-provider failure classes — a feed that
has started returning `403` or `404` will show as `PERMANENT`.

**`corpus-verify` exits 2.** Stop the timers before investigating. Do not back
up over the last good archive.

**A cycle was killed mid-run.** Nothing to do. The manifest is written last, so
a partial cycle leaves no manifest and no half-registered corpus; the next run
breaks the stale lock and reports that it did.
