# Final polish before Power BI: report

Written for the project owner. Nothing was committed; nothing is staged. Numbers come from the files
(`reports/RESULTS.md` is the single place they are quoted).

## A. Earlier follow-up, verified (not redone)

- The four reason texts (R02, R04, R05, R10) say "items sold", and the `own items` wording is in place.
- `tests/test_config.py` skips with a reason when `data/raw` has no CSV and still fails for an incomplete folder.
- The Makefile `load-sql` target matches `tasks.ps1`.
- `python -m src.load_sql --verify-only`: all ten tables OK (row counts and checksums).
- `.\tasks.ps1 verify` passed at the start and passes again at the end (section G).

## B. Main pages

| Item | What changed |
|---|---|
| B1 Intelligence | Average review is per order: **4.14 / 5** (one review per order, counted once for each seller in the order, the `SellerFacts.avg_review` definition). The distinct-order mean is 4.16 and the per-item mean 4.08; both are in `reports/RESULTS.md`'s new Marketplace section. The tooltip says which one is used. |
| B2 Design | Two sentences under the trade-off sentence: the strongest input (caveated when it is a calendar input) and the forward-looking result ("no model did meaningfully better than chance", chosen by a 0.55 threshold read from `diagnostic_temporal.txt`). |
| B3 Implementation | One recommendation sentence (the subtitle states the meaning only). The default seller has at least 5 reviewed orders and a rule other than the default R12 (now 130 items sold, Feature, R07). Technical details show Items sold, Delivered orders and Reviewed orders side by side. |
| B4 Discovery | Caption about ranking by quality score; "few items sold" wording; tiers shown as **Stronger evidence / Early signal** everywhere, including the CSV download (`src/discovery.py` is locked, so the mapping lives in `app/_labels.py`). |
| B5 scan | "orders" meaning items sold, "Actionable" and "Watchlist" removed from user-facing text in `app/`, `docs/`, `README.md`. The remaining "orders" hits in the page audit are all real distinct-order counts. |

## C. Technical details on every page

- **C0 shared helpers.** `app/_format.py` (no negative zero, thousands separators, fixed decimals), `app/_labels.py`
  (one map from column names to plain headers), `table()` in `app/_components.py`, one risk colour pair
  (`RISK_RAISES`/`RISK_LOWERS`/`RISK_SCALE`, taken from the existing action and quadrant colours), and a three-colour device palette
  separate from every action and quadrant colour.
- **C1 Intelligence.** "Synthetic data" tag on the clickstream parts; friendly stage names; bubble size and colour of the
  call-sentiment chart described in a caption.
- **C2 Design.** Split wording (order-grouped, fold 0 of 5, seed, base rate) and the 0.5 decision threshold (the library default in
  `predict._evaluate`, pinned by a test); one confusion table; heatmap and weights chart share the risk colours; permutation
  importance with plain headers and 4 decimals; "class 1 = late" line above the tree.
- **C3 Choice.** Plain criteria and ranking headers; raw and shrunk values side by side with reviewed and delivered orders; one
  sentence on why a seller scores slightly differently on Choice and Discovery (scores are scaled within each pool: mean
  difference +0.013 over the 681 sellers in both pools, rank correlation 0.998).
- **C4 Discovery.** Friendly headers through `table()`.
- **C5 Implementation.** The rule-engine explanation is rebuilt in the app with clean numbers ("items sold = 5 >= 5") and wraps.
- **C6 DSS Architecture.** The module table adds `src/load_sql.py`, `sql/schema.sql` and `src/explain.py`; the KPI row reads
  16 data files, 10 loaded into SQL Server (counted from `sql/schema.sql`), 12 rules.
- **C7.** "ranking probability" in `src/explain.py` replaced by the chance that a randomly chosen late order is ranked above a
  randomly chosen on-time one. Wording only, plus the public `fact_label()` helper.
- **C8 tests.** 273 to 323 tests: format helpers, shared risk colours, no raw snake_case headers, tier names (UI and download),
  synthetic tag, DSS rows, default seller, KPI equals the file mean, the 0.5 threshold, and the Power BI notes (below).

## D. The 5-item gate: measured only, nothing implemented

Question: is "5 or more items sold" enough evidence for rules that read per-order averages?

- Sellers each rule fires on, with fewer than 3 reviewed orders: R01 1 of 70, R03 0 of 144, R04 7 of 66, R05 1 of 31, R10 3 of 170.
  Warn or Suspend sellers with fewer than 3 reviewed and fewer than 3 delivered orders: 8 of 317.
- 17 sellers have 5 or more items sold but fewer than 3 delivered orders (18 with fewer than 3 reviewed).
- Example: seller `010da060...` has 5 items sold but 1 delivered and 1 reviewed order (average review 1.00, Warn by R04). The R04
  sentence "enough items sold to be a real pattern rather than one bad review" is true only in the weak sense for it.
- Proposed alternative (simulated in memory, not implemented): reviewed orders >= 3 for the rules reading `avg_review` (R01, R04,
  R10), delivered orders >= 3 for the rules reading `late_rate` (R02, R05). Effect: 8 sellers change action (Warn to Keep 7,
  Suspend to Warn 1); Keep 2,269 to 2,276, Warn 241 to 235, Suspend 76 to 75; Feature and Promote unchanged. Gating every rule that
  reads either value (adds R07, R11) gives the same action counts.
- Recommendation: small and defensible, but it would change `seller_recommendations` and the thresholds would need re-deriving, so
  it is a decision for you. The current gate is documented in `docs/knowledge_engineering.md` section 10 and is honest as it stands.

## E. Full-project checks

- **E1 page audit.** 29 numbers shown on the seven pages equal their source files (0 mismatches); no raw snake_case table header on
  any page; no stale literal (0.7841, 0.7928, 0.6738, 0.6347, 454/478/480/452, 423 MB).
- **E2 cross-page.** 20 sellers (default, top Choice, top Discovery gem, seeded sample): items sold, average review and action
  identical across Choice, Discovery, Implementation, `SellerFacts` and the recommendations (live engine equals batch). The only
  difference is the quality score between the 30+ and 5+ item pools, explained on the Choice page.
- **E3 integrity.** 96,470 orders, 110,189 items, unique keys, zero orphans, risk scores in [0, 1], quadrants 495/437/439/493, gem
  tiers 44 stronger / 393 early, 2,970 recommendations (Keep 2,269, Warn 241, Suspend 76, Feature 174, Promote 210), per-rule fire
  counts R01 70, R02 6, R03 144, R04 66, R05 31, R06 137, R07 37, R08 106, R09 104, R10 170, R11 701, R12 1,398.
- **E4 numbers in docs.** `scripts/make_results.py --check` passes; the demo-script and threshold-table tests pass.
- **E5 CLAUDE.md.** Layout and section 7.7 updated for the new files; sections 1 and 10 still match your decisions (SQL exception
  confined to three files).
- **E6 SQL.** `--verify-only` all ten OK; `sql/schema.sql` has 10 tables and 8 foreign keys; 10 export CSVs; each table appears
  in `docs/powerbi_setup.md`.
- **E7 R07 wording.** The `because` text of R07 already reads "items sold at or above the 90th percentile ... at least 4.3, a little
  above the median seller" (the all-seller median is 4.28). No change was needed, so no output changed and SQL was not reloaded.
- **E8 hygiene.** No tracked `.pbix` or `powerbi/` file; `.gitignore` covers raw, processed, export, models, `.venv` and `*.pbix`;
  `AUDIT_REPORT.md` exists only under `docs/archive/`; no file over 1 MB outside ignored folders and caches.

## F. Power BI notes

`docs/powerbi_measures.md` (60 lines): seven measures with DAX, format strings and the expected value on an unfiltered card
(96,470 delivered orders; 110,189 items; R$ 13,220,248.93; 7.91%; 0.361; 437; 681), the eight relationships with cardinality
and filter direction, and a refresh note. Two tests keep it honest: every table and column it names exists in `sql/schema.sql`,
and every expected value equals the one computed from the files. It is in the `docs/README.md` index.

## G. Verification

- `.\tasks.ps1 verify`: ruff PASS, mypy PASS (13 files), pytest PASS (323 passed), SQL verify-only PASS, RESULTS.md fresh PASS, app
  smoke PASS (all seven pages load, no exception).
- Manifest against the start of this round (N0): all 49 of the 50 generated-output files byte-identical (`reports/`,
  `data/processed/`, `data/export/`, `models/`, `sql/`). The one difference is `reports/RESULTS.md`, which gained the four-line
  Marketplace section (the average review figure). `seller_recommendations.csv`, every model and `sql/schema.sql` are identical.
- Other changed files are all in the allowed set: `app/`, docs, tests, `scripts/make_results.py`, `src/explain.py` (wording and
  `fact_label`), `BUILD_LOG.md`, `CLAUDE.md`. New files: `app/_format.py`, `app/_labels.py`, `tests/test_app_polish.py`,
  `docs/powerbi_measures.md`, this report.
- Not touched: `src/ahp.py`, `src/discovery.py`, `src/predict.py`, `src/expert.py`, `rules/seller_rules.yaml`, `sql/schema.sql`.
- `powerbi/EcomDSS.pbix` SHA-256 before and after: `EAA0C60E6DF9732ABF6C2781F4FE8431E4B15F4F48A7F3DBB48B6F59768E8A45` (unchanged).
- Nothing is staged (`git diff --cached --name-only` is empty).

## H. Commit commands (PowerShell, one commit; not run)

```powershell
git add -A
git --no-pager diff --cached --stat
git diff --cached --name-only | Select-String -Pattern '\.pbix$|^powerbi/|^data/|^models/|^\.venv/'   # must print nothing
git commit -m "polish: plain-language pages and Technical details, per-order average review, Power BI measures notes"
git log -1 --format="%an <%ae>"
git push
```

## I. Judgement calls and decisions left to you

1. **Which average review is "4.14".** I used the seller-order version (4.138), which is what every seller's `avg_review` uses.
   The distinct-order mean is 4.156 (shown as 4.16) and the per-item mean 4.082. Both are in `reports/RESULTS.md`.
2. **"Own items arrived late".** The late rate is taken over item rows, so the text says items, not orders.
3. **Forward-looking wording.** The forest's temporal AUC is 0.518, a little above 0.5. The page says "no model did meaningfully
   better than chance" with a 0.55 cut-off rather than "exactly chance".
4. **Tier names** are mapped in the app only, so `discovery_quadrants.csv` and the SQL table still contain `normal` / `low` and the
   internal names "Actionable" / "Watchlist" remain inside the locked `src/discovery.py`.
5. **The gate question in section D** (adopt reviewed/delivered-order gates, or keep the 5-item gate).
6. **`ruff format`** is not enforced across the project (34 older files would be reformatted); only `ruff check` is part of `verify`.
