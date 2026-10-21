# Polish round 2 (2026-10-21)

Written for the project owner. Starting point: HEAD `4dcd28e` (equal to `origin/main`, tree clean), 385 tests. Nothing was committed or
staged. `powerbi/EcomDSS.pbix` SHA-256 before and after: `eaa0c60e6df9732abf6c2781f4fe8431e4b15f4f48a7f3dbb48b6f59768e8a45` (unchanged).
Besides the usual headless page tests I ran the real Streamlit server and read what it sends to a browser (a websocket probe, scratch
file in `%TEMP%\ecom_polish2\`): all seven pages rendered with no exception, the tab icon was served, Home had six page links, and the
heading tags were as listed below. I could not look at the pages in a browser, so the visual checks are in `docs/MANUAL_CHECKS.md`.

## Findings fixed, before and after

| ID | Page | Before | After |
|---|---|---|---|
| F1 | Implementation | A seller with too little history (for example `010da060...`: 5 items, 1 delivered order, 1 review) read "Keep" with the grey line "No change is needed for now." | Headline "Keep for now: too few orders to warn or suspend" (smaller type), the grey line gives its items sold, delivered orders and reviewed orders against the minimums, a "Thin evidence" note follows. The minimums come from `explain.evidence_status(facts, rules)`, which reads the Warn and Suspend gates from the rule objects (5 items sold, 3 delivered, 3 reviewed today). A fourth number, Delivered orders, sits beside Items sold, Average review and Late-delivery rate. Rule outputs, the YAML conditions and the batch CSV are unchanged for this fix. |
| F2 | Discovery | Headings "Stronger evidence (15+ items sold): 44 sellers" | "...: top 10 of 44 sellers" and "top 10 of 393 sellers" (counts from files), one note says the full list is in the download and under Technical details; the Technical-details top-gems table keeps one column ("Evidence": Stronger evidence / Early signal) instead of two; the download keeps all its columns. |
| F3 | Design | "Late-delivery rate 7.9%", KPI "Ranking score" | "7.91%" (same formatter as Intelligence); KPI "Late-order ranking score" with a tooltip: the chance a randomly chosen late order is ranked above a randomly chosen on-time one, 50% is a coin flip, not accuracy and not the probability an order is late. |
| F4 | All | Section headings were h3 under an h1 (a skipped level); hover link icons; default tab icon | Section headings are h2 (page title h1); `stHeaderActionElements` hidden by CSS; `app/static/icon.png` (64 px, dark square with three gold bars) used by every page through `page_icon` (no remote asset, no emoji). Title pattern "<Page> \| Olist DSS" unchanged. |
| F5 | Home | Page names were `st.page_link` elements but looked like plain text | Real-server probe: six page links were already sent. They are now underlined in the accent colour; the fallback when no page registry exists is a plain markdown link to the page address instead of bare text. |
| F6 | All | `.flow-box` had `height: 16rem; overflow: hidden` (text clipped when narrow) | `min-height: 16rem`, no clipping. No `min-width` or fixed pixel width above 400 px existed; a test now forbids both. |
| F7 | Intelligence and charts | "Synthetic data" tag 12 px, footer 12.8 px, chart tick/legend/annotation text 12 px | Tag and footer 13.1 px, chart text 13 px; contrast and layout tests unchanged and passing. |
| F9 | `src/load_sql.py` | `SELECT ... INTO #t FROM dbo.<loader table>` would have passed the allow-list | `assert_statement_allowed` also refuses `SELECT ... INTO`, any `#`, `EXEC`/`EXECUTE`, and more than one statement per call (one trailing `;` is fine); the DDL batches read from `sql/schema.sql` are declared `batch=True` and each statement in them is still checked. 28 new rejected-form cases; `--verify-only` and `--dry-run` unchanged. |
| F10 | Makefile | no `verify` or `rebuild` | both added (verify runs every check and fails if any failed, like `tasks.ps1`); a test compares the Makefile and `tasks.ps1` target sets (12 each). |

## V1: rule R07 (read-only first)

| Claim in the `because` text | Actual | Status |
|---|---|---|
| items sold "at or above the 90th percentile" (threshold 83) | 83 is the 90th percentile of all 2,970 sellers: 89.9% sell fewer, 299 sellers (10.1%) sell 83 or more. Among the 1,864 sellers with at least 5 items sold it is only the 84th percentile (16.0% at or above) | supported for all sellers (the text names no other group); text unchanged |
| average review "at least 4.3, a little above the median seller" | all 2,965 reviewed sellers: median 4.281, 4.3 is the 50.7th percentile; the 2,126 sellers with at least 3 reviewed orders: median 4.244, 4.3 is the 55.7th percentile | supported; text unchanged |

## V2: every other `because` text

| Rule | Claim | Actual | Status |
|---|---|---|---|
| R01 | risk in the top quartile of sellers with enough evidence (5 items, 3 orders, 3 reviewed); review below 3.5 | 0.413 is the 75.1st percentile of 1,846 sellers | supported |
| R02 | over 40% late with at least 10 items and 3 orders | fixed levels equal the YAML | supported |
| R03 | top 10% of sellers with 5 items and 3 orders | 0.476 is the 90.1st percentile of 1,847 sellers | supported |
| R04 | review below 3.5, 5 items, 3 reviewed orders | fixed levels equal the YAML | supported |
| R05 | "90th-percentile territory" | 0.25 is the 94.4th percentile of sellers with enough history (91.8th over all sellers): the worst 6% | **corrected** to "only the worst 6% of sellers with enough history reach" |
| R06 | "top 10% of established sellers" | 0.711 is the 90.0th percentile of the 1,864 scored sellers (5+ items) but only the 80.6th of the 681 established sellers (30+); also a stale "recalibrated" remark | **corrected**: "top 10% of the sellers that have one (those with at least 5 items sold)", median of all sellers (0.339 is the 50.0th percentile) |
| R07 | see V1 | supported | unchanged |
| R08 | risk below "CLAUDE.md's original Promote threshold" | CLAUDE.md no longer holds that figure; 0.3 holds for 50.8% of hidden gems | **corrected**: "below 0.3, which holds for about half of the hidden gems" |
| R09 | roughly the lowest-risk quartile of hidden gems | 0.252 is the 25.4th percentile of the 437 gems | supported |
| R10 | at-or-below-median risk and review; stale "recalibrated" remark | risk 0.339 is the 50.0th percentile of all sellers; review 4.28 the 49.9th | **corrected** (remark removed, "median of all sellers") |
| R11 | "bottom decile of established sellers"; 75th percentile of late rate | 0.549 is the 10.0th percentile of scored sellers (11.0th of established); 0.10 is the 75.8th percentile of all sellers | **corrected** to "sellers that have one" / "of all sellers" |
| R12 | "often a very low order-volume seller" | 1,106 of its 1,400 sellers sell fewer than 5 items | supported |

Only the `because` column of 1,151 recommendations changed (R05 31, R06 137, R08 106, R10 171, R11 706). Action and rule id of all 2,970
sellers, the action counts (Keep 2,277, Warn 234, Suspend 75, Feature 174, Promote 210) and every per-rule fire count are identical;
`python -m src.export_sql`, `python -m src.load_sql` and `--verify-only` were run (all ten tables and the view OK). The em-dash count in
the reasons fell from 1,155 to 141 because the new texts use colons (`docs/MANUAL_CHECKS.md` query 3 updated). A test
(`tests/test_polish2.py`) re-derives each corrected claim from the files.

## V3 and V4

- **V3 (Choice: "the shrunk review and on-time values themselves are the same on both" pages):** true. For all 681 sellers in both pools
  the shrunk review, shrunk on-time rate and both raw values from `build_seller_criteria` at 30+ and at 5+ items are identical (maximum
  difference 0), and `reports/ahp_ranking.csv` equals them to 1e-9. Cause: the marketplace means used for the shrinkage are taken over all
  orders before the pool filter. Wording kept; test added.
- **V4 (Intelligence KPI tooltip):** it already names its definition ("one review per seller-order pair", 4.14), gives 4.16 for distinct
  reviewed orders and 4.08 per item, and the share without a review is computed; the existing test checks all three against `RESULTS.md`.

## Docs (D1 to D4)

`docs/DEMO_SCRIPT.md` has a "Before you present" box (open every page once, about 8 to 10 s cold and 2 to 4 s warm, keep the terminal open,
zoom 100 to 125%, what to say about the Keep-for-now headline; the cold and warm times are your browser observation, not measured here).
`docs/MANUAL_CHECKS.md` gained rows for the narrow window and 150% zoom, the download and its tier column, the tab icon, hidden link icons,
Home links and the thin-evidence seller. Section H of `docs/archive/FULL_AUDIT_REPORT.md` now marks each item resolved by this task,
resolved earlier or still open, without changing any of your decisions. `BUILD_LOG.md`, `README.md` (Makefile targets) and `CLAUDE.md`
(layout, `explain.py` description) were updated where a statement had become wrong. `reports/RESULTS.md` regenerated: unchanged.

## Tests and timings

- `.\tasks.ps1 verify`: ruff PASS, mypy PASS (13 files), pytest PASS (**430 passed**, 385 before; 45 new: `tests/test_polish2.py` 28, loader
  rejected forms and view tests 17), SQL verify-only PASS, RESULTS.md fresh PASS, app smoke PASS (all seven pages, Implementation shows four numbers).
- The full suite took 11 min 51 s this time against 3 min 33 s in the audit; it ran while other jobs were using the machine. The slow
  tests are the same few in `tests/test_predict.py`.
- All seven pages ran under AppTest and in the real server without an exception.

## Hash comparison (manifest before and after, 130 files)

Generated files (`reports/`, `data/processed/`, `data/export/`, `models/`, `sql/`): 48 of 50 byte-identical; the two that differ are
`reports/seller_recommendations.csv` and `data/export/seller_recommendations.csv` (the `because` column only). `sql/schema.sql`, every
model, parquet file, other export and report did not change. The SQL table `SellerRecommendations` was reloaded from the new export and
verified. Other changed files, all in the allowed set: `app/` (components, theme, three pages, `static/icon.png`), `src/explain.py`,
`src/load_sql.py`, `rules/seller_rules.yaml` (reason texts and a comment), `Makefile`, tests, docs. `powerbi/` untouched.

## Still open

- The Olist licence text (could not be read by the fetch tool).
- The review tie-break (17 orders), declined by you and documented.
- Everything a browser is needed for: the tick table in `docs/MANUAL_CHECKS.md` (narrow window, zoom, look of the icon and links).
- A refresh in Power BI is not needed for this round's code, but the reason texts in the recommendation table changed, so refresh once.

## Commit commands (PowerShell; one commit; not run; no attribution line)

```powershell
git add -A
git --no-pager diff --cached --stat
git diff --cached --name-only | Select-String -Pattern '\.pbix$|^powerbi/|^data/|^models/|^\.venv/'   # must print nothing
git commit -m "polish: thin-evidence headline, Discovery and Design wording, heading levels and icon, loader hardening, reason texts corrected"
git log -1 --format="%an <%ae>"
git push
```
