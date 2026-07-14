# Independent Audit & Re-Architecture Mandate — Tennis Model Laboratory

**To:** Codex (independent auditor / architect)
**From:** the current maintainer + Claude (who has been patching this system reactively)
**Purpose:** You are being asked to do an **independent audit** of this tennis
betting pipeline, its data/logging layer, its dashboard, and its models — then
**reason about and implement a robust redesign.** This document is a *problem
statement and context dump*, **not a spec.** Do not treat any suggested fix
here as a requirement. Where you see a better path — different cloud platform,
different persistence model, different dashboard architecture, different model
approach — take it and justify it. The owner explicitly wants you to think from
first principles, not follow a checklist.

---

## 0. What this system is (one paragraph)

A fully-automated, paper-trading tennis betting research pipeline. Every hour a
job scrapes Bovada tennis odds, builds a 141-feature vector per match from a
canonical match/player database, runs 13 models (a neural net, XGBoost, random
forest, and 10 shadow variants — LightGBM, CatBoost, XGB depth/recency variants,
NN architecture variants) plus a de-vigged market baseline, logs each model's
win probability, computes edges, records paper bets with Kelly-fractional
staking, settles completed matches from official sources, and publishes
everything to a static dashboard ("The Ledger"). It runs on GitHub Actions
(free), mirrors to Supabase Postgres, and serves a hand-rolled static
HTML/JS/SVG dashboard on GitHub Pages. The owner also bets small real money off
the model edges and wants the research loop to eventually support serious ML
work (see §7). **The paramount near-term goal: a self-healing pipeline that just
works — reliably fetches, predicts, bets, logs, settles — without a human
babysitting it or a maintainer patching a new failure every hour.**

---

## 1. The mandate, in priority order

1. **Make the data/persistence layer robust and self-healing.** This is the
   root of most pain (see §2). Everything else depends on it.
2. **Fix round resolution & "bettable" determination** so the current slate is
   always correct and matches the book (see §3).
3. **Fix / rethink parallelization and the scraping reliability** (see §4).
4. **Independently audit the models — especially the neural net's
   overconfidence pathology** (see §5). Log what you find; you don't have to
   fully fix the ML in this pass, but characterize it.
5. **Audit and (if you think best) redesign the dashboard and the logging
   semantics** — they are confusing and not clean (see §6). You reportedly have
   strong UI judgment; use it.
6. **Verify feature integrity and training parity** (see §5b) — arguably as
   important as §2, and *dependent* on it. If the live features don't match how
   the models were trained, every probability is suspect regardless of how
   robust the plumbing is.
7. **Be aware of the deferred research directions** (see §7) so your
   architecture doesn't foreclose them — but do NOT build them now.
8. **Surface what we're NOT seeing.** The problems enumerated here are the ones
   we *found*, usually the hard way. The owner explicitly wants you to
   reason from first principles and **call out issues, risks, and improvements
   nobody has mentioned** — silent correctness bugs, leakage vectors, dependency
   fragilities, statistical mistakes in how performance is judged, operational
   blind spots, security/cost concerns, better designs. Treat this list as a
   floor, not a ceiling. If something here is wrong or misdiagnosed, say so.

You have latitude to change anything: the cloud platform (GitHub Actions may be
the wrong tool), the persistence design, the dashboard framework, the repo
structure. Justify big moves; keep the system paper-only and $0-or-cheap unless
you flag a cost explicitly.

---

## 2. The core problem: data persistence is fragile (fix this first)

**Symptom the owner sees:** numbers on the dashboard change and even go
*backwards* between page loads — the settled-prediction count drops, the
"Model Report" cohort size wobbles (they watched it go 113 → 92 → 83), the
current slate froze on a 2-day-old snapshot showing matches whose start times
had passed. It "feels flimsy."

**Root cause (as currently understood):** there is **no single source of truth**,
and the persistence path is a git-committed CSV that multiple writers overwrite.

- Predictions live in `production/prediction_log.csv` (also snapshots, odds
  history, shadow predictions — all CSVs). These are **committed to git** by each
  cloud run so state survives the ephemeral runner.
- The dashboard mirror is Supabase; `production/dashboard_sync.py` **TRUNCATEs
  and reloads** each `dash_*` table from those CSVs every run.
- The GitHub Actions runs **race on `git push`** (two cron schedules at :17 and
  :47, plus the maintainer's dev commits). When a push loses the race it retries,
  then parks the logs on a throwaway `runner-logs-*` branch. **66+ such branches
  have accumulated, and `prediction_log.csv` on `main` has been frozen since
  2026-07-11 20:30** while the pipeline kept running.
- Because `main` is stale, each run checks out a **stale base**, re-runs
  settlement from scratch, and TRUNCATE-reloads the mirror with *that run's*
  partial result. A run that settles fewer matches than the previous run (e.g. a
  bad ITF-scraping hour) **silently un-settled** rows in the mirror → the visible
  regressions.
- A **manual `dashboard_sync` from a maintainer's laptop** (whose local CSV was
  also stale) overwrote the mirror's current slate back to July 11. So even
  human intervention can corrupt it.

**Band-aids already applied (so you don't redo them, and can rip them out if you
redesign):**
- `dashboard_sync.py` now snapshots settled outcomes before the TRUNCATE and
  restores any the reload is missing → "a settlement is permanent," mirror
  monotonic for settlements.
- A **recency guard**: the sync refuses prediction data whose newest
  `latest_run_id` is older than the mirror's, so a stale source can't regress
  the slate.
- A **shrink guard**: refuses a sync that would drop >10% of rows.

These stop the *bleeding* but the design is still CSV-in-git-as-truth with
wipe-and-reload. **The owner and Claude both believe the real fix is to move
prediction/settlement/odds state into a proper database (Supabase already
exists for the canonical match/player store) as the single source of truth,
with additive upserts instead of wipe-reload, and remove git from the hot
persistence path.** *This is a preliminary opinion — you should validate or
reject it.* Open questions you should reason about: is Supabase the right store
or do you want the pipeline to own its own Postgres? Should the runner be
stateless and read/write only the DB? Is GitHub Actions even the right executor
given the push-race problem, or should this move to a scheduled container /
serverless job / a small always-on box? (Note: the owner has a home machine
available; a prior finding is that Tennis Abstract and ITF block datacenter IPs
but a residential IP works — see §4.)

---

## 3. Round resolution & "bettable" determination

A match is only "bettable" when its **round is known** (drives round one-hot
features and the round-offset date heuristic) and all 141 features are real (no
silent defaults). Rounds come from a layered live-source chain: ATP results
pages, tournament **draw** pages, the **daily-schedule** page (the only official
source that labels *qualifying* rounds), a registry-free "both players' last
match was the same event's same round → next round" inference, and for ITF the
itftennis.com order-of-play. Surfaces come from ATP challenger calendar + ITF
API + store history, cross-checked; disagreements flag not-bettable.

**Problems seen:**
- New-week (Sun/Mon) slates repeatedly showed many "round pending" because
  next-week events weren't discoverable (tour events had no calendar source
  until recently) and qualifying rounds aren't on draw sheets.
- Matches vs still-undetermined qualifiers legitimately have no round yet →
  correctly not-bettable, but the UX makes this look broken.
- A match with a **null tournament** was showing bettable (couldn't verify
  surface/level) — now guarded to not-bettable, but it's a symptom of
  under-validated metadata.
- Timing: sources publish on their own clock (draws ~Sat, schedules the
  evening-of), so the pipeline must keep re-asking hourly and *never write a
  guess*. It does, but the whole thing is brittle and hard to reason about.

- **A single flaky fetch collapses the entire slate.** Round resolution depends
  on ATP event *discovery* (the challenger/tour calendar pages). When that fetch
  intermittently returns an empty parse ("calendar discovery unavailable —
  returned no events", the same datacenter-IP flakiness as §4), **no events are
  discovered, so no draws are fetched, so EVERY match that run gets round=None
  and goes not-bettable.** Observed live: Bastad went 11/11 bettable on one run
  to 0/11 the next, purely because that run's calendar fetch came back empty.
- **And "latest run wins" lets a failed run clobber a good one.** Because each
  run TRUNCATE-reloads the mirror and the newest run's data is authoritative, a
  run that resolved *fewer* rounds overwrites a prior run that had them all —
  the round resolution is non-deterministic per run and there is no memory of
  prior success (the stale git base, §2, forgets it). Band-aid added: the mirror
  now treats a **resolved round / complete prediction as sticky** (restores a
  complete prediction the reload downgraded to round-pending) — the same
  monotonic principle as settlement-preservation. But this is a mirror-side
  patch on a design where features are re-derived from scratch every run and any
  one flaky fetch can erase them.

**What to reason about:** is the layered live-scrape round chain the right model,
or should round/draw/schedule state be **ingested into the DB once per event and
served from there** (so a match's round, once known, is a durable fact, not
something re-scraped and re-riskable every hour)? How should "bettable" be
defined and surfaced so it's obviously correct and self-explaining? How should
the pipeline degrade when a source fetch fails — never let one empty parse zero
out the whole slate?

---

## 4. Scraping reliability & parallelization

- **Bovada** (odds): works from cloud IPs. Single long page, scrolled/expanded
  sequentially. Occasionally returns a degraded page (bot-wall) → the run must
  treat a fetch failure as a *tolerated* event, not a crash. (A recent bug made
  odds-fetch failure kill the whole run and discard completed settlement work;
  fixed, but the pattern recurs.)
- **Tennis Abstract** and **ITF (itftennis.com, behind Incapsula)** **block
  datacenter IPs** — TA hard (403), ITF intermittently (serves block pages, JSON
  parse fails). This is a *fundamental* constraint for cloud execution: ITF
  settlement/features fail unpredictably from GitHub's runners. A prior feasibility
  test found a **home/residential IP works for all sources**. This strongly
  shapes the "where should this run" question.
- **Parallelization:** originally one shared Playwright browser, all fetches
  sequential — a Sunday's ~15-tournament discovery took ~40 min and flirted with
  the job timeout. Now thread-local browsers + a small prefetch pool (3 ATP / 2
  ITF workers) warm event pages in parallel (~4½ min). An ITF circuit-breaker
  stops hammering after 6 consecutive Incapsula failures. Runtime and the
  scraping abstraction are still fragile and worth a clean rethink.

**Reason about:** the right execution environment given the residential-IP
constraint; a clean, testable scraping layer with proper backoff/rotation; how
to make a single bad source degrade gracefully instead of cascading.

---

## 5. The models — audit them, especially the neural net

There are 13 logged models + a de-vigged market baseline. On the **clean recent
cohort** the models look profitable and the NN looks strong (best flat & gated
ROI, decent ECE); on the **large older cohort** (mostly the `v1.2.1` regime and
April, i.e. *pre-surface-fix, pre-days-since-fix* — built on since-repaired buggy
features, so **not a fair test of the current models**) the market beats every
model on log loss/Brier and the NN is the *worst* (log loss ~0.73). XGBoost is
consistently well-calibrated (ECE ~0.025, sometimes better than the market).

**The central ML question the owner cares about — the neural net's
overconfidence:** the NN periodically emits **near-certain probabilities
(≥90%, up to 100%)**, and those extreme picks are **only ~60% accurate** — a
few "95%, and wrong" calls carry log loss ~1.84 and single-handedly wreck the
NN's aggregate calibration. If this is fixable, the NN might be the *best*
model rather than XGBoost. **Partial, honest findings (Claude's investigation
stalled — treat as leads, not conclusions):**
- Attempts to reproduce the NN on its own training CSV were **misaligned** (49.7%
  accuracy, outputs never below 0.5) — the raw ml-ready CSV's player-orientation
  didn't match inference. So "does the NN produce extremes on clean training
  data?" is **unresolved.** Resolving it is the fork between "recalibrate the
  model (temperature/Platt scaling)" and "fix a live feature that's pushing it
  out-of-distribution."
- An out-of-distribution feature scan (|z|>8 vs training stats) found mostly
  **rare-but-valid country one-hots** (Swiss/Serbian players) and rare handedness
  matchups — *not* the extreme-driver. So the simple "a wild feature value causes
  it" hypothesis is **not** confirmed.
- A suspected "wrong round (R64 at an ATP 250)" driving a 100% could **not** be
  reproduced in the data — the flagged case was actually Wimbledon (128-draw,
  R64 valid). So that lead didn't hold either.

**What to do:** independently audit the models. Fix the training-reproduction
harness so the "extremes in training vs only live" question can be answered
cleanly. Characterize the NN pathology and recommend (or implement) a fix —
post-hoc calibration is the obvious candidate but validate the cause first.
Note: model versioning is registry-driven (`production/models/model_registry.json`),
semver, pipeline-only bumps documented; the clean-feature regime is `v1.2.3`
(NN) / `v1.2.2` (XGB/RF) and its settled sample is still tiny (~18), so trust it
cautiously.

---

## 5b. Feature integrity & training parity (depends on §2 — do not skip)

The models are only as good as the 141 features they're fed, and those features
must be computed **identically to how they were computed at training time**, or
the model sees a distribution it never learned on. This has been a **persistent,
recurring source of silent bugs** — and, critically, it is **downstream of the
data-layer problems in §2**. A running list of feature bugs found and fixed
*this month alone* (each was silently wrong before it was caught, usually by the
owner noticing an implausible value on the dashboard):

- **Surface resolution** — matches scored on the wrong surface (fallback-to-Hard,
  or a fuzzy tournament match picking the wrong venue/surface). ~158 predictions
  / 35 paper bets were affected before it was caught.
- **Days_Since_Last week-boundary bug** — a reference timestamp carrying a
  clock time made the *current* week count as a "previous tournament," collapsing
  days-since to ~5 for anyone active that week — a value that appears **zero
  times** in 400k training rows. The owner caught it from an impossible "5 days"
  on a player who'd just played.
- **Form-window double-counting** — the same tournament ingested under two event
  names ("Bogota" vs "Bogota, Colombia Kia Open") double-counted a player's
  recent matches, inflating form/win-rate windows (513 duplicate rows).
- **Wrong-week event dating** — finished events still on the live hub were
  stamped with the scrape-week's Monday, landing rows in the wrong week and
  corrupting week-based features.
- **Rank / points conventions** — unranked must be rank 999 / 0 points to match
  training; a flat "500 points" default for a missing rank was a top-110
  player's points. Rank-curve interpolation now used instead.
- **Handedness / height missingness** — unknown-hand acts as a missing-data
  proxy the model leans on; live `Hand_U` ran ~4× the training incidence at
  challenger level until the ATP bio fetch was extended to capture it.
- **Laplace-smoothed form rates**, **round-offset date heuristics**, and
  **P1/P2 orientation** all had to be pinned to training conventions (there are
  now symmetry + window-math pytests, and a per-build audit gate that flags
  |z|>8 features against the training scaler's mean/std).

**The owner's key insight, and the reason this is a §2 dependency:** live
features are computed from the canonical store's match history. **If settlement
and ingestion aren't keeping the store current and complete — which is exactly
the failure mode in §2 — then every history-derived feature is computed on
missing data.** Days-since-last, matches-in-last-14/30-days, current streak,
recent win rate, surface form, H2H — all of them silently wrong when the most
recent matches haven't been ingested or a result wasn't logged. **You cannot
trust the features until the data layer is provably current and complete.** The
two problems are one problem.

**What to reason about / build:**
- A **feature-parity harness**: compute features live for a set of matches and
  compare, field by field, against what the training pipeline produces (and/or a
  golden hand-verified set). Gate on parity. Today there is a partial
  `feature_audit.py` (per-build |z|>8 / one-hot / presence checks) and a
  `FEATURE_AUDIT.md` living doc, but no end-to-end train-vs-live equivalence test.
- Fix the **training-reproduction harness** (see §5) — it's currently misaligned
  (49.7% acc feeding the raw ml-ready CSV), which *also* blocks the NN
  investigation. Getting this right is a prerequisite for measuring parity at all.
- A **data-completeness precondition**: before computing features, verify the
  store actually contains the recent matches it should (is it current? are there
  gaps?), and fail loudly / stitch rather than compute on stale history.
- **Leakage review**: the temporal features are the whole edge and the whole
  risk. There's a documented history of leakage bugs (a `.values` positional
  scramble of inferred match timestamps was a past root cause). An independent
  leakage audit of the current feature build is warranted.

## 6. Dashboard & logging semantics are confusing

"The Ledger" (`docs/index.html`, plus a `docs/v2.html` variant) is hand-rolled
static HTML + vanilla JS + inline SVG charts, reading `dash_*` tables from
Supabase. Recurring confusion:
- The **Model Report cohort count** ("all 13 models — same N matches") is a
  strict filtered intersection (settled ∩ complete ∩ non-surface-suspect ∩ all
  13 models logged the same match by name+date). It is **not** a results
  counter, it is **non-monotonic**, and it moved around alarmingly — the owner
  reasonably read it as "results being lost." The metrics (log loss, ECE, ROI)
  are valid for whatever cohort is shown, but the UX doesn't make that legible.
- **Surface-suspect** exclusion, **era** (pre / hourly-since-2026-07-08),
  **model-version** filters, **flat vs gated (edge≥5pt) ROI**, **all-13 vs
  core-trio** cohorts — lots of correct-but-opaque toggles.
- Recurring **stale-tab** problems: the page fetched data on a timer but never
  reloaded its own code, so an open tab ran old JS forever (a fixed
  duplicate-tiles bug kept recurring in a stale tab; a self-heal reload was
  added). Auto-refresh + a "data as of" stamp exist now.
- Logging is spread across many CSVs with overlapping semantics
  (prediction_log, snapshots, odds_history, shadow predictions, settlement
  audit, run history, bets, bankroll). It is hard to reason about what is
  authoritative.

**Reason about / redesign freely:** a clean data model for
predictions/odds/settlements/bets; a dashboard whose numbers are obviously
trustworthy and self-explaining (what's the headline "is it working" signal vs.
the eval cohort); whether to keep static HTML or move to a real framework. You
have design latitude here.

---

## 7. Deferred research directions (awareness only — do NOT build now)

Once the pipeline is solid, self-healing, and the data is trustworthy, the owner
wants to go deep on the modeling/quant side. Keep the architecture from
foreclosing these; do not implement them in this pass:
- **Backtesting** harness on historical data (the Sackmann dataset, ~916k rows,
  1968→present) with honest chronological splits.
- **Bet sizing**: Kelly variants (fractional, capped), and the owner's own
  scale-based sizing; comparison of sizing policies.
- **Monte Carlo / bootstrapping** for P&L distributions and confidence on ROI /
  calibration given small live samples.
- **Retraining** on more recent data; the current models trained on an honest
  split (train <2022, val 2022, test ≥2023) but the world moves.
- **Feature engineering & model optimization**; deeper NN/DL work — the owner
  explicitly wants to "get deep into machine learning and deep learning" and is
  open to approaches they haven't considered.
- Calibration as a first-class concern (the NN issue above is the leading
  example).

---

## 8. Repo orientation (where to look)

- `production/main.py` — the hourly orchestrator (fetch → features → predict →
  edge/stake → log → settle → sync).
- `production/dashboard_sync.py` — mirror sync (the guards live here).
- `production/canonical_store.py`, `store_history.py` — Supabase match/player DB.
- `production/prediction_logger.py`, `feature_vector_log.py` — the CSV logging.
- `production/auto_settle.py` — multi-source settlement (TA / ATP / ITF).
- `production/features/ta_feature_calculator.py`, `history_stitch.py` — the
  141-feature build + live-source stitching + round resolution.
- `production/scraping/` — Bovada/ATP/ITF/rankings scrapers, shared browser,
  `prefetch.py`.
- `production/models/` — model artifacts, `inference.py`, `model_registry.json`,
  `feature_training_stats.json`.
- `docs/index.html`, `docs/v2.html` — the dashboard. `docs/modeling/FEATURE_AUDIT.md`
  — the living feature-integrity doc (nets, incidents, coverage).
- `.github/workflows/hourly-pipeline.yml` — the cron + commit-race logic.
- `tests/` — 95 passing tests (pytest); includes P1/P2 feature symmetry,
  window-math, settlement, dashboard smoke.

**Engineering standards the owner holds:** no silent fallbacks (real values or
loud failure — never a generic default without logging the break); root-cause
fixes over patches; verify against actual code/data before explaining a cause
(don't fabricate plausible narratives); small focused changes; clean readable
code over clever one-liners; semver + registry for model changes.

---

## 9. Suggested (not mandated) starting sequence

If it helps, a reasonable order — but design your own: (1) audit + write up your
independent findings on the data layer, models, dashboard, and scraping; (2)
propose the persistence redesign (DB-as-truth vs alternatives) and get sign-off
before large moves; (3) implement the robust data/persistence layer; (4) fix
round/bettable and scraping reliability on top of it; (5) rethink the dashboard;
(6) characterize/fix the NN calibration; (7) leave clean seams for the §7
research work. The owner wants your *reasoning*, so narrate the trade-offs.
