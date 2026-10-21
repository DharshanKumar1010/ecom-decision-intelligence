# Power BI / SQL Server Setup Notes

Working notes for the manual SSMS + Power BI layer (CLAUDE.md section 16).
This file is maintained by Claude Code as a reference for the human-driven
import/modeling steps; Claude Code does not perform those steps itself.

## Export -> SQL Server table mapping

Every CSV below is written by `python -m src.export_sql` to `data/export/`.
Row counts are from the real last run (`python -m src.export_sql`, after
Stage 3 Part A's expanded predictive model) — reproduce and update this
table if the pipeline is rerun on different data.

| CSV (`data/export/`) | SQL Server table | Role | Row count (last real run) |
|---|---|---|---|
| `FactOrderItems.csv` | `FactOrderItems` | Fact table | 110,189 |
| `DimSeller.csv` | `DimSeller` | Dimension | 3,095 |
| `DimProduct.csv` | `DimProduct` | Dimension | 32,951 |
| `DimCustomer.csv` | `DimCustomer` | Dimension | 99,441 |
| `LatePredictions.csv` | `LatePredictions` | Fact extension (joins to `FactOrderItems` on `item_key`) | 110,189 |
| `CallSentiment.csv` | `CallSentiment` | Supporting fact (per-call/per-agent, **synthetic**) | 300 |
| `Clickstream.csv` | `Clickstream` | Supporting fact (**synthetic**) | 100,000 |
| `ahp_ranking.csv` | `AhpRanking` | Dimension extension (joins to `DimSeller` on `seller_id`) | 681 |
| `discovery_quadrants.csv` | `DiscoveryQuadrants` | Dimension extension (joins to `DimSeller` on `seller_id`) | 1,864 |
| `seller_recommendations.csv` | `SellerRecommendations` | Dimension extension (joins to `DimSeller` on `seller_id`) | 2,970 |

**Total: 10 tables, 461,680 rows exported.**

`AhpRanking` carries two versions of the review and on-time criteria: `avg_review_score` and
`on_time_rate` are the **shrunk** values used for scoring (pulled toward the marketplace mean,
`docs/knowledge_engineering.md` section 8), while `avg_review_raw` and `on_time_rate_raw` are
the raw averages, both computed per distinct order (one review per order); use the raw
columns for any visual labelled "average review" or "on-time rate". `n_reviewed` is the number
of distinct reviewed orders and `n_orders` the number of distinct delivered orders (the n used
for shrinkage); `order_volume` counts items sold (order lines), not distinct orders.

## Load into SQL Server

Use the loader instead of the Import Flat File wizard:

```powershell
.\tasks.ps1 load-sql                       # same as: python -m src.load_sql
python -m src.load_sql --dry-run           # check everything, open no connection
python -m src.load_sql --verify-only       # compare the database with the CSVs, change nothing
python -m src.load_sql --only CallSentiment --only Clickstream   # reload just these tables
```

What it does:

1. Connects to `localhost\SQLEXPRESS` (override with `--server` or `ECOM_SQL_SERVER`) with
   Windows Authentication. No password exists anywhere. It refuses any database other than
   `EcomDecisionIntelligence`, which **must already exist** (create it once in SSMS with
   `CREATE DATABASE [EcomDecisionIntelligence];`; the loader never creates or drops a database).
2. Reads the ten CSVs from `data/export/`, checks every value fits the column type in
   `sql/schema.sql` (nothing is rounded or truncated silently) and checks each CSV's row count
   against the table above, all before touching the database.
3. Drops and recreates only the ten tables and the view `dbo.vw_SellerScorecard`, in dependency
   order, with primary keys, the eight foreign keys and the Power BI join indexes from
   `sql/schema.sql`. Each table is created, loaded in batches of 10,000 and verified inside one
   transaction.
4. Verifies each table: SQL row count = CSV row count = the count in the table above, plus one
   exact checksum column per table. On any mismatch that table is rolled back and the process
   exits non-zero.
5. Prints a final table (table, rows loaded, expected, status, seconds).

After a re-export (`python -m src.export_sql` or `run_all.py --export`) run `load-sql` again; if
the row counts changed, update the table above first (the loader refuses to run while they
disagree, which is how a stale count gets noticed). `sql/schema.sql` also runs on its own in
SSMS if you only want the empty tables. `dbo.vw_SellerScorecard` joins `DimSeller` to
`AhpRanking`, `DiscoveryQuadrants` and `SellerRecommendations` with left joins, so all 3,095
sellers stay in the view; 2,414 of them have no AHP score because the AHP ranking only covers
sellers with 30 or more items sold. Its `items_sold` column counts every seller's items in
`FactOrderItems` (0 for the 125 sellers with no delivered item), so it is filled for all 3,095 rows
and sums to the total number of items sold; it used to come from the 5+ items pool and was empty
for 1,231 sellers.

## Import steps (SSMS wizard, manual; superseded by the loader above)

The wizard guesses types from the first rows of each file and gets several wrong for these
exports: `DimProduct.product_weight_g` has two blanks (the first at data row 8,579) so the column is
guessed NOT NULL and the import fails; `seller_recommendations.because` contains an em dash and
the file has no byte-order mark; and several text columns grow longer than the sampled rows
(`item_key`, the category names, `quadrant`), so they are truncated. Use the loader. The manual
steps are kept only for reference:

1. Open SSMS, connect to the local SQL Server Express instance.
2. Right-click the target database -> **Tasks -> Import Flat File**.
3. For each CSV above, run the wizard once, naming the destination table
   exactly as in the "SQL Server table" column so the relationships below
   line up.
4. Datetime columns in `FactOrderItems.csv` and `Clickstream.csv`
   (`order_purchase_timestamp`, `order_delivered_customer_date`,
   `order_estimated_delivery_date`, `order_date`, `review_creation_date`,
   `event_time`) are pre-formatted as ISO 8601 strings
   (`YYYY-MM-DDTHH:MM:SS`); a missing value is an empty cell, not the text
   `NaT` — the wizard should infer these as `datetime2` / nullable
   correctly. If it infers `varchar` instead, override the column type to
   `datetime2` manually in the wizard's preview step.

## Power BI Model view relationships

- `FactOrderItems` <-> `DimSeller` / `DimProduct` / `DimCustomer` on their
  respective id columns.
- `FactOrderItems` <-> `LatePredictions` on `item_key`.
- `DimSeller` <-> `AhpRanking` / `DiscoveryQuadrants` / `SellerRecommendations`
  on `seller_id`.

## Labelling requirement

Every Power BI page or visual touching `Clickstream` or
`CallSentiment`/`CallTranscripts` must note the underlying data is
synthetic — Olist has no real clickstream or call-center data (CLAUDE.md
section 6.3).

## Open items / not yet done

- SQL Server load: done by `python -m src.load_sql` (Stage 7, `BUILD_LOG.md`); all ten tables
  and the view verified against the CSVs. The older hand-imported `DimCustomer` and `DimProduct`
  tables were replaced by it.
- Power BI Desktop model build, DAX measures, dashboard pages, paginated
  report, row-level security (T4-T15): not started.
