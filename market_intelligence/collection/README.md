# Forward point-in-time collection — operator guide

Everything an operator needs before switching a provider on. Run
`btc-intel providers` for the same facts as JSON, generated from the
declarations in code rather than from this file.

## The rule everything rests on

**A document discovered today was available today.** `published_at` is recorded;
`available_at` is retrieval. A press release dated three weeks ago that this
system first saw an hour ago became usable an hour ago, and treating its
publication date as its availability would fabricate three weeks of hindsight
into every study built on the corpus.

The only exception would be a provider that supplies independently defensible
evidence of an earlier availability — a vendor first-seen timestamp under a
documented contract. None of the providers below does.

## Provider classes

| class | meaning |
| --- | --- |
| `PUBLIC_DOCUMENTED` | a feed or API the publisher documents and intends to be consumed |
| `AUTHENTICATED_LICENSED` | a commercial or contracted API, used with credentials under its terms |
| `USER_SUPPLIED` | a dataset the operator supplies and vouches for |
| `DISABLED` | declared but not operable here, and reported as such |

What is deliberately absent: scraping a site whose terms or technical controls
prohibit it, CAPTCHA bypass, and login circumvention. A corpus whose provenance
cannot be defended is worth less than no corpus.

## Providers

### `syndication` — PUBLIC_DOCUMENTED, no credentials

Collects from RSS **and Atom** feeds the operator lists. Only the configured
URLs are fetched: no crawling, no link-following, no discovery. The allowlist is
the policy.

* **Credentials** — none.
* **Cost** — free.
* **Cadence** — 900 s minimum. Government sites do rate-limit and some require a
  descriptive `User-Agent`. Poll for research need, not for maximum extraction.
* **Returns** — entry title, link, summary, publication timestamp, author.
* **Raw retention** — `FULL`. Public government feeds may be stored and shared.
* **Disable** — `enabled: false` in the provider configuration.
* **Health check** — `btc-intel health`, or `btc-intel providers`.

Suggested feeds ship in `feeds.py` (SEC press and litigation, Federal Reserve
press and monetary, CFTC, Treasury, BLS), all classified `primary_source` and
`official_source`. **None is enabled by default.** URLs are not guaranteed
stable; a retired feed surfaces as a `PERMANENT` failure rather than a silent
gap.

### `news-api` — AUTHENTICATED_LICENSED

* **Credentials** — `BTC_INTEL_SEARCH_API_KEY`.
* **Cost** — paid contract.
* **Cadence** — 3600 s default; set from the contract.
* **Raw retention** — `NON_REDISTRIBUTABLE_RAW_SOURCE`. Most news APIs forbid
  redistributing article text, so only a hash and permitted metadata leave this
  machine.

### `statements` — AUTHENTICATED_LICENSED, disabled without a contract

The provider B4 needed for Trump, Musk, Powell and Saylor and did not have.

* **Credentials** — `BTC_INTEL_STATEMENTS_API_KEY`, plus an endpoint.
* **Cost** — paid contract.
* **Cadence** — 3600 s. These APIs meter monthly as well as per-minute.
* **Raw retention** — `NON_REDISTRIBUTABLE_RAW_SOURCE`.
* **Without a credential** — the provider is `DISABLED` and reports the reason.
  It does not silently return nothing, because "collected nothing" and "nothing
  was said" are different facts.

No scraping substitute is provided. Without a licensed API this stays off.

### `whales` — AUTHENTICATED_LICENSED, disabled without a contract

* **Credentials** — `BTC_INTEL_WHALE_API_KEY`, plus an endpoint.
* **Cost** — paid contract. The core system runs without it.
* **Cadence** — 1800 s. These meter by request and often by result volume.
* **Raw retention** — `NON_REDISTRIBUTABLE_RAW_SOURCE`.

**Classification is never inferred locally.** A transfer is `EXCHANGE_INFLOW`
only if the provider says so. Reading "binance-hot-wallet" out of an attribution
string and concluding "inflow" would produce a dataset with far fewer `UNKNOWN`s
and far more apparent signal, and it would be fabricated — indistinguishable in
the data from a provider that knew. `UNKNOWN` stays `UNKNOWN`, and is studied as
its own category.

## Third-party terms change

Every rate limit, price and permission above is a snapshot taken while writing
this, not a promise. Confirm current terms before enabling anything, and treat
the `terms_note` on each declaration as a starting point for that check.

## Credentials

Environment variables only. Never committed, never logged, never persisted in a
manifest, never returned through the API. `redact_mapping` redacts by key name
as well as by explicit list, because the explicit list is what falls out of date
when a field is added.

## Operating

```
btc-intel providers                     # what is declared, and what can run
btc-intel collect --config … --origin … --manifest …
btc-intel corpus-status                 # what the corpus holds, and B4 readiness
btc-intel corpus-status --json          # the same, machine-readable
btc-intel corpora                       # registered corpus snapshots
```

Scheduler-agnostic: cron, a systemd timer, a container scheduler or an external
orchestrator all work. Nothing here needs an always-on process.

## Storage

Measured, not modelled. `btc-intel corpus-status --json` reports bytes per
thousand documents and per thousand events from the bytes actually stored, and a
linear projection labelled as one. Raw evidence dominates when retention is
`FULL`; nothing is compressed on the strength of a projection.

## Retention

Research evidence needed for reproducibility is not deleted casually. For
licence-restricted sources, metadata and hash retention is independent of raw
content retention — the hash always survives, so a payload someone else holds
can be checked against the one this system used.

Export (`EvidenceStore.export_redistributable`) writes only what may leave the
machine and marks the rest `OMITTED_LICENSE`, with the hash still present.
