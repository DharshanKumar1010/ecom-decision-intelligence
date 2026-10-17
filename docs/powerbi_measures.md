# Power BI measures and model checks

The seven measures to create in Power BI Desktop (Home > New measure) once the ten tables are
loaded (`python -m src.load_sql`), the value each should show on a card with no filters, and the
relationships to expect in Model view. Table and column names are checked against
`sql/schema.sql`; the expected values are computed from the exported files by the same code that
writes `reports/RESULTS.md`. This file is documentation only: the report itself is built by hand.

## Measures

| Measure | DAX | Format | Expected (no filters) |
|---|---|---|---|
| Delivered Orders | `Delivered Orders = DISTINCTCOUNT(FactOrderItems[order_id])` | Whole number, thousands separator (`#,0`) | 96,470 |
| Items Sold | `Items Sold = COUNTROWS(FactOrderItems)` | `#,0` | 110,189 |
| Revenue | `Revenue = SUM(FactOrderItems[price])` | Currency, 2 decimals (`"R$" #,0.00`) | R$ 13,220,248.93 |
| Late Rate | `Late Rate = DIVIDE(SUM(FactOrderItems[is_late]), COUNTROWS(FactOrderItems))` | Percentage, 2 decimals (`0.00%`) | 7.91% |
| Avg Late Risk Score | `Avg Late Risk Score = AVERAGE(LatePredictions[late_risk])` | Decimal, 3 places (`0.000`) | 0.361 |
| Hidden Gems | `Hidden Gems = CALCULATE(COUNTROWS(DiscoveryQuadrants), DiscoveryQuadrants[quadrant] = "Hidden Gem")` | `#,0` | 437 |
| Sellers Ranked | `Sellers Ranked = COUNTROWS(AhpRanking)` | `#,0` | 681 |

Notes:
- **Items Sold counts order lines** (an order with three products is three items); **Delivered
  Orders counts each order once**. Label them that way on the report.
- **Avg Late Risk Score is a ranking score, not a probability.** It is the average of the
  out-of-fold scores over items, so it sits far above the real late rate (Late Rate). Do not
  format it as a percentage or call it a chance of lateness.
- Revenue excludes freight (`FactOrderItems[freight_value]` is a separate column).
- Hidden Gems rest on few items sold; show the Stronger evidence / Early signal split
  (`DiscoveryQuadrants[confidence]`: `normal` is Stronger evidence, `low` is Early signal) next to
  the number.
- Clickstream and call-sentiment visuals must say the data is synthetic.

## Relationships to expect in Model view

These are the eight foreign keys declared in `sql/schema.sql`, which Power BI should detect on load. In
each, the first column is the "many" or child side.

| From (column) | To (column) | Cardinality | Cross-filter direction |
|---|---|---|---|
| `FactOrderItems[seller_id]` | `DimSeller[seller_id]` | many to one | Single (dimension filters the fact) |
| `FactOrderItems[product_id]` | `DimProduct[product_id]` | many to one | Single |
| `FactOrderItems[customer_id]` | `DimCustomer[customer_id]` | many to one | Single |
| `LatePredictions[item_key]` | `FactOrderItems[item_key]` | one to one | Both |
| `AhpRanking[seller_id]` | `DimSeller[seller_id]` | one to one | Both |
| `DiscoveryQuadrants[seller_id]` | `DimSeller[seller_id]` | one to one | Both |
| `SellerRecommendations[seller_id]` | `DimSeller[seller_id]` | one to one | Both |
| `Clickstream[product_id]` | `DimProduct[product_id]` | many to one | Single |

Power BI makes a one-to-one relationship filter in both directions; that is expected. Seller
slicers should use `DimSeller`. `AhpRanking` covers only the 30+ items sellers and
`DiscoveryQuadrants` only the 5+ items sellers, so a visual that mixes them with all of
`DimSeller` shows blanks for the others; the view `dbo.vw_SellerScorecard` (one row per seller,
left joins) is an alternative source when a single flat table is easier.

## After a database reload

`python -m src.load_sql` drops and recreates the tables. Open the report in Power BI Desktop and
press **Refresh** (Home ribbon). The model needs no change unless a column was added or renamed;
if a relationship disappears, recreate it from the table above. The cards should show the
expected values above again.
