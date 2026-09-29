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

**Total: 10 tables, 361,860 rows exported.**

## Import steps (SSMS, manual)

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

- SSMS import itself: not run (manual, outside Claude Code).
- Power BI Desktop model build, DAX measures, dashboard pages, paginated
  report, row-level security (T4-T15): not started.
