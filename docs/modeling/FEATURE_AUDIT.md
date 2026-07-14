# Feature Audit — living document

Scope: the NN-141 base features (all models) + 57 granular aggregates (shadow variants, 198 total).
Regime boundary: **NN v1.2.2 / XGB v1.2.1 / RF v1.2.1** (2026-07-09) = clean-feature pipeline; see model_registry.json notes.

## Verification nets (layered — no single net catches everything)

| Net | Catches | Status |
|---|---|---|
| Cross-source corroboration (surface: live official vs store vs registry) | wrong-source values | ✅ live |
| Training-distribution gate (`feature_audit.py`, every build, z>8 + one-hot + presence) | unit/scale/sign bugs, renames | ✅ live, self-tested |
| First-complete freeze + no-downgrade | silent row corruption | ✅ live |
| Store-vs-TA parity (141 features, 2026-07-06) | pipeline divergence | ✅ 138/145 identical, 7 = rank-lineage (documented) |
| Granular raw-stat verification vs official ATP pages | ingestion errors | ✅ spot-verified (Fritz–Zverev QF, 8/8 columns) |
| P1/P2 symmetry invariant pytest (`test_feature_symmetry.py`, all 141 mirror under swap) | asymmetric feature math | ✅ live in CI — calculator verified orientation-clean |
| Golden-match hand verification (Sinner–Djokovic Wimbledon SF 2026-07-10: bio, context one-hots, H2H, form windows, surface rates, ranks recomputed independently from raw SQL + hand formulas) | formula errors | ✅ 24/24 PASS — sole deviation was the frozen vector's pre-fix Days_Since (regime v1.2.2 vintage, see incidents); fixed code reproduces training semantics (11) on identical history |
| Granular aggregate recompute / window-math fixtures (`test_form_windows.py`) | window math | ✅ live in CI — Laplace constants, EWM half-life, window edges, streaks pinned to hand values |
| Shared train/serve candidate (`test_base_141_shared_parity.py`) | disputed temporal/H2H semantics, ordering, future leakage | 🧪 opt-in candidate — synthetic ledger is field-exact across all 141 fields; activation remains blocked on source provenance and new model releases |
| Cross-source reconciliation (`reconcile_store.py`, hourly): conflicts table, Sackmann-wins level repair, stats gap-fill | source disagreement, silent gaps | ✅ live — activates on source overlap (next Sackmann drop) |

## Known incidents (all fixed, all annotated)

- **Days-since week-boundary collapse** (fixed 2026-07-11, versions NN v1.2.3 / XGB v1.2.2 / RF v1.2.2): refs carrying a clock time (Bovada datetimes) made the current week's rows count as a "previous tournament" (`Mon 00:00 < Mon 12:20`), collapsing `Days_Since_Last` to ~5 for any player with matches this week. Training refs were midnight-based, so training never hit it. Regression-tested.
- **Store label duplicates** (fixed 2026-07-11): the same tournament discovered as "Bogota" (calendar) and "Bogota, Colombia Kia Open" (live hub) across runs double-inserted 513 rows, double-counting form windows. Writes now label-blind (players+date+round identity), reads dedupe defensively.
- **Wrong-week event dating** (fixed 2026-07-11): finished events still on the live hub were stamped with the scrape week's Monday; five events (210 rows) sat one week late — corrupts week-based features. Discovery now takes true start dates from the calendar; `tourney_id` stored for id-keyed repair.
- **Surface fallback-Hard** (fixed 2026-07-09): 158 predictions / 35 paper bets on wrong surfaces; suspect-flagged in UI, excluded from headline metrics by default.
- **Frozen-row downgrade** (fixed 2026-07-09): flaky runs could strip round/completeness; now structurally impossible.
- **Unranked hard-fail** (fixed 2026-07-08): live rejected unranked players; training used rank=999 — now mirrored.

## Hand_U missingness (measured 2026-07-11)

Live `Hand_U` ran hot vs training — unknown handedness was acting as a missingness proxy:
challenger 9.6% live vs 2.4% training (4×), ITF-15 44% vs 18% (2.4×), tour/slam ≈ parity.
Fix shipped: the ATP bio fetch now captures `Plays:` alongside height (same page, zero extra
fetches), caches to `data/atp_hands.json`, and writes through to `players.hand` — every
ranked player self-heals on first encounter. Honest residual: deep-ITF juniors with no ATP
profile stay `U` (they are also the bulk of training's 18% `U` at that level).

## Data coverage (2026 rows, granular serve stats)

| Tier | Coverage |
|---|---|
| Masters | 100% | 
| ATP tour (A) | 99.8% |
| ITF 15/25 | 99.7% |
| Challenger | 86% |
| Slams | 33% ⚠️ (Sackmann lag + enrichment covers stitched rows only) |

## Base-141 inventory by family

### Temporal form & schedule — 33 features
*Source:* store match history (Sackmann 1968→ + live ATP/ITF ingest) + round-offset date heuristic (training parity — NOT calendar dates)

`H2H_P1_WinRate`, `P1_BigMatch_WinRate`, `P1_Days_Since_Last`, `P1_Finals_WinRate`, `P1_Form_Trend_30d`, `P1_Level_Matches_Career`, `P1_Level_WinRate_Career`, `P1_Matches_14d`, `P1_Matches_30d`, `P1_Round_WinRate_Career`, `P1_Semifinals_WinRate`, `P1_Surface_Matches_30d`, `P1_Surface_Matches_90d`, `P1_Surface_WinRate_90d`, `P1_WinRate_Last10_120d`, `P1_WinStreak_Current`, `P1_vs_Lefty_WinRate`, `P2_BigMatch_WinRate`, `P2_Days_Since_Last`, `P2_Finals_WinRate`, `P2_Form_Trend_30d`, `P2_Level_Matches_Career`, `P2_Level_WinRate_Career`, `P2_Matches_14d`, `P2_Matches_30d`, `P2_Round_WinRate_Career`, `P2_Semifinals_WinRate`, `P2_Surface_Matches_30d`, `P2_Surface_Matches_90d`, `P2_Surface_WinRate_90d`, `P2_WinRate_Last10_120d`, `P2_WinStreak_Current`, `P2_vs_Lefty_WinRate`

### Other — 31 features
*Source:* derived/interaction features computed from the above

`Indoor_Season`, `P1_Country_ARG`, `P1_Country_AUS`, `P1_Country_CZE`, `P1_Country_ESP`, `P1_Country_FRA`, `P1_Country_GBR`, `P1_Country_GER`, `P1_Country_ITA`, `P1_Country_Other`, `P1_Country_RUS`, `P1_Country_SRB`, `P1_Country_SUI`, `P1_Country_USA`, `P1_Rust_Flag`, `P1_Sets_14d`, `P2_Country_ARG`, `P2_Country_AUS`, `P2_Country_CZE`, `P2_Country_ESP`, `P2_Country_FRA`, `P2_Country_GBR`, `P2_Country_GER`, `P2_Country_ITA`, `P2_Country_Other`, `P2_Country_RUS`, `P2_Country_SRB`, `P2_Country_SUI`, `P2_Country_USA`, `P2_Rust_Flag`, `P2_Sets_14d`

### Context (round/level/draw) — 25 features
*Source:* round: ATP draws/results inference + ITF order-of-play (certainty, no guessing); level/draw: resolver registry + title parsing

`Level_15`, `Level_25`, `Level_A`, `Level_C`, `Level_D`, `Level_F`, `Level_G`, `Level_M`, `Level_O`, `Level_S`, `Round_BR`, `Round_ER`, `Round_F`, `Round_Q1`, `Round_Q2`, `Round_Q3`, `Round_Q4`, `Round_QF`, `Round_R128`, `Round_R16`, `Round_R32`, `Round_R64`, `Round_RR`, `Round_SF`, `draw_size`

### Physical & bio — 22 features
*Source:* store `players` → TA profile fallback → ATP height scraper; missing → flagged not-bettable

`Age_Diff`, `Avg_Age`, `Avg_Height`, `Handedness_Matchup_LL`, `Handedness_Matchup_LR`, `Handedness_Matchup_RL`, `Handedness_Matchup_RR`, `Height_Diff`, `P1_Hand_A`, `P1_Hand_L`, `P1_Hand_R`, `P1_Hand_U`, `P1_Peak_Age`, `P2_Hand_A`, `P2_Hand_L`, `P2_Hand_R`, `P2_Hand_U`, `P2_Peak_Age`, `Player1_Age`, `Player1_Height`, `Player2_Age`, `Player2_Height`

### Ranking & points — 17 features
*Source:* canonical store `players`+rankings CSV (live atptour scrape, dated cache fallback w/ provenance); unranked=999/0pts = training convention

`Avg_Rank`, `Avg_Rank_Points`, `P1_Rank_Change_30d`, `P1_Rank_Change_90d`, `P1_Rank_Volatility_90d`, `P2_Rank_Change_30d`, `P2_Rank_Change_90d`, `P2_Rank_Volatility_90d`, `Player1_Rank`, `Player1_Rank_Points`, `Player2_Rank`, `Player2_Rank_Points`, `Rank_Diff`, `Rank_Momentum_Diff_30d`, `Rank_Momentum_Diff_90d`, `Rank_Points_Diff`, `Rank_Ratio`

### Surface — 9 features
*Source:* LIVE-FIRST chain: ATP calendar / ITF API surfaceDesc → store history (month-aware) → registry; ambiguous → flagged not-bettable

`Clay_Season`, `Grass_Season`, `P1_Surface_Experience`, `P2_Surface_Experience`, `Surface_Carpet`, `Surface_Clay`, `Surface_Grass`, `Surface_Hard`, `Surface_Transition_Flag`

### Head-to-head — 4 features
*Source:* store match history, both perspectives, dedup vs stitched rows

`H2H_P1_Wins`, `H2H_P2_Wins`, `H2H_Recent_P1_Advantage`, `H2H_Total_Matches`

## Granular 57 (shadow variants only)

Aggregates over per-match serve stats (`w_ace, w_df, w_svpt, w_1stin, w_1stwon, w_2ndwon, w_bpsaved, w_bpfaced` + mirrors):
ace/DF rates, 1st-in %, 1st/2nd-won %, BP save/convert, game/set/tiebreak win rates, with `Stat_Matches_*` coverage counters telling the models how fresh the stat window is.
Sources: Sackmann historical (97% futures coverage) + live ATP stats-page enrichment (fixture-tested); ITF live tail is score-only by design (coverage counters account for it).
