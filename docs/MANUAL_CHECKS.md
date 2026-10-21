# Manual checks

Things only a person can judge, plus ten queries to run in SSMS. Automated checks are
`.\tasks.ps1 verify`; numbers are in `reports/RESULTS.md`. Start the app with
`.\tasks.ps1 app` (or `streamlit run app/Home.py`).

## In the browser, page by page

Tick a row once you have looked at it. The automated tests already cover the numbers and the
text; these rows are what only your eyes can judge (layout, colour, readability, how it feels).
Start the app with `.\tasks.ps1 app`, resize the window to a laptop width, and use a dark room
and a bright room at least once.

| Page | What to look at | Tick |
|---|---|---|
| Home | The title asks which sellers to back and which are being overlooked; the sidebar lists six pages; a sentence says the clickstream and call data are synthetic. | [ ] |
| 1 Intelligence | Four figures: delivered orders, item revenue, late-delivery rate, average review score. They equal `reports/RESULTS.md` and the SQL sums below (late rate about 7.9%). The review score is the mean of one review per seller-order pair (about 4.14, the definition behind every seller's average review); its tooltip also gives the distinct-order and per-item figures, which are deliberately not used. | [ ] |
| 1 Intelligence | The funnel and call-sentiment charts carry a visible **Synthetic data** tag (five headings in all, including the ones inside Technical details). Funnel: roughly 21 of every 100 visits add to the cart, 13 reach checkout, 9 buy. Sentiment is described as coming from transcripts only. | [ ] |
| 1 Intelligence | Technical details: steps read "Viewed a page", "Added to cart", "Reached checkout", "Purchased"; counts have thousands separators and shares two decimals. The device chart uses three clearly different colours (lavender, aqua, sand), none of them an action or group colour. The call-sentiment caption says bubble size is the number of calls and colour is mean sentiment; hovering a bubble shows the agent id. | [ ] |
| 2 Design | Headline "Late-order ranking score" of about 74% (hover its tooltip: the chance a late order is ranked above an on-time one, 50% is a coin flip, not accuracy) and a late-delivery rate shown with two decimals, the same as Intelligence. The page says the score *ranks* orders and never says "forecast", "predicts future deliveries" or "probability". The out-of-fold note and the note that the model has little skill on future months are present. | [ ] |
| 2 Design | The bars show what the model leans on (month of purchase first) and a sentence says this is not what causes lateness. The "What if an input changes?" chart moves when you change the selection; red bars raise late risk and blue bars lower it. Two plain sentences sit under the trade-off sentence (strongest input with a caution when it is the month; the forward-looking result). | [ ] |
| 2 Design | Technical details: the split note says order-grouped, fold 0 of 5, and that precision, recall and F1 use the default 0.5 threshold; one combined confusion table with nothing clipped; permutation importance with friendly names; "Class 1 = late, class 0 = on time" above the tree; the linear-weights chart and the sensitivity heatmap use the same red and blue; the heatmap shows no "-0.000". | [ ] |
| 3 Choice | 681 sellers ranked, minimum 30 items sold, consistency ratio 0.019. In Technical details, change one comparison box: the bars move. Make the judgments contradict each other: the sentence says the weights should not be trusted. | [ ] |
| 3 Choice | The score column is headed "AHP score (30+ items pool)". The full table lists raw review and on-time rate per order, reviewed and delivered orders, the shrunk values (headed "shrunk") and the four contributions; a note explains that the same seller scores slightly differently on Discovery because each page scales within its own pool. | [ ] |
| 4 Discovery | The quadrant counts line equals `reports/RESULTS.md`. Tier labels read **Stronger evidence** and **Early signal** everywhere, including the downloaded CSV (never "Actionable" or "Watchlist"). Early-signal gems are hollow markers; the four quadrant colours can be told apart, also in a greyscale screenshot. | [ ] |
| 4 Discovery | A caption under "Hidden gems by confidence" says both lists are ranked by quality score. Wording says "items sold", never "orders", for a seller's volume. Nothing mentions food or nutrition except to say "healthier" means business quality. | [ ] |
| 5 Implementation | Opens on a seller with at least 5 reviewed orders where a real rule fired (not the default rule R12; currently a Feature seller). The grey line says what the recommendation means, once; the big word is the recommendation, with a tick/cross list of the rule's conditions. "Try an example" switches to the first seller with each outcome. | [ ] |
| 5 Implementation | Technical details: the full rule table, the "what would change the outcome" box, items sold, delivered orders and reviewed orders, the rule-engine explanation (wrapped, no decimal noise), a search box with a hint, and the line comparing the live result with the stored batch result (they must agree). Paste the id of a seller with fewer than 5 items sold (or fewer than 3 delivered orders): Keep from the default rule, "pending more data". | [ ] |
| 6 DSS Architecture | Data files 16, loaded into SQL Server 10, expert rules 12; the four boxes (data, models, knowledge, interface) read in order; the catalog row counts equal the pipeline files; the module table lists `src/load_sql.py`, `sql/schema.sql` and `src/explain.py`, each marked present. | [ ] |
| 4 Discovery | The two list headings read "top 10 of N" with N the tier's total; the download button works and the CSV's tier column reads Stronger evidence / Early signal (never Actionable / Watchlist). In Technical details the top-gems table has one evidence column, not two. | [ ] |
| 5 Implementation | Thin-evidence seller: paste `010da0602d7774602cd1b3f5fb7b709e`. The headline reads "Keep for now: too few orders to warn or suspend", the grey line gives its items sold, delivered orders and reviewed orders against the minimums, and a "Thin evidence" note appears. A seller with enough history who is Keep still reads "No change is needed for now". Four numbers sit under the headline (items sold, delivered orders, average review, late-delivery rate). | [ ] |
| Home | Each of the six page names is an underlined link that opens that page. | [ ] |
| All pages | Tab icon: a small dark square with three gold bars on every page. Hovering a heading shows no link icon. Headings step down in order (page title, then section titles). | [ ] |
| All pages | Narrow window and zoom: drag the window to a phone-like width and set the browser zoom to 150 percent; text wraps, the four boxes on DSS Architecture grow instead of clipping, nothing sits off-screen to the right. | [ ] |
| All pages | Colours: the five action colours and four group colours stay the same on every page; Keep and Promote, and Hidden Gem and Overlooked-Low-Quality, are the pairs closest under red-green colour blindness (about 6 to 7 on a 0 to 100 scale), so check each is also named in words or a label. No white boxes, one sans-serif font, nothing clipped at laptop width. | [ ] |

## Validate in SSMS

Connect to `localhost\SQLEXPRESS`, open a new query on the `EcomDecisionIntelligence` database
and run each query. The expected results come from the exported CSVs (`reports/RESULTS.md` has
the same counts); they change only when the pipeline is re-run, after which `python -m src.load_sql`
reloads the database. All queries only read.

**1. Row count of every table** (expected: DimSeller 3,095; DimProduct 32,951; DimCustomer
99,441; FactOrderItems 110,189; LatePredictions 110,189; CallSentiment 300; Clickstream 100,000;
AhpRanking 681; DiscoveryQuadrants 1,864; SellerRecommendations 2,970; total 461,680)

```sql
SELECT 'DimSeller' AS tbl, COUNT_BIG(*) AS n FROM dbo.DimSeller
UNION ALL SELECT 'DimProduct', COUNT_BIG(*) FROM dbo.DimProduct
UNION ALL SELECT 'DimCustomer', COUNT_BIG(*) FROM dbo.DimCustomer
UNION ALL SELECT 'FactOrderItems', COUNT_BIG(*) FROM dbo.FactOrderItems
UNION ALL SELECT 'LatePredictions', COUNT_BIG(*) FROM dbo.LatePredictions
UNION ALL SELECT 'CallSentiment', COUNT_BIG(*) FROM dbo.CallSentiment
UNION ALL SELECT 'Clickstream', COUNT_BIG(*) FROM dbo.Clickstream
UNION ALL SELECT 'AhpRanking', COUNT_BIG(*) FROM dbo.AhpRanking
UNION ALL SELECT 'DiscoveryQuadrants', COUNT_BIG(*) FROM dbo.DiscoveryQuadrants
UNION ALL SELECT 'SellerRecommendations', COUNT_BIG(*) FROM dbo.SellerRecommendations;
```

**2. Orphan rows per relationship** (expected: 0 in every row; the foreign keys already
enforce this, so a non-zero result means the constraints were disabled)

```sql
SELECT 'Fact -> DimSeller' AS relationship, COUNT_BIG(*) AS orphans
  FROM dbo.FactOrderItems f LEFT JOIN dbo.DimSeller s ON s.seller_id = f.seller_id WHERE s.seller_id IS NULL
UNION ALL SELECT 'Fact -> DimProduct', COUNT_BIG(*)
  FROM dbo.FactOrderItems f LEFT JOIN dbo.DimProduct p ON p.product_id = f.product_id WHERE p.product_id IS NULL
UNION ALL SELECT 'Fact -> DimCustomer', COUNT_BIG(*)
  FROM dbo.FactOrderItems f LEFT JOIN dbo.DimCustomer c ON c.customer_id = f.customer_id WHERE c.customer_id IS NULL
UNION ALL SELECT 'LatePredictions -> Fact', COUNT_BIG(*)
  FROM dbo.LatePredictions l LEFT JOIN dbo.FactOrderItems f ON f.item_key = l.item_key WHERE f.item_key IS NULL
UNION ALL SELECT 'AhpRanking -> DimSeller', COUNT_BIG(*)
  FROM dbo.AhpRanking a LEFT JOIN dbo.DimSeller s ON s.seller_id = a.seller_id WHERE s.seller_id IS NULL
UNION ALL SELECT 'DiscoveryQuadrants -> DimSeller', COUNT_BIG(*)
  FROM dbo.DiscoveryQuadrants q LEFT JOIN dbo.DimSeller s ON s.seller_id = q.seller_id WHERE s.seller_id IS NULL
UNION ALL SELECT 'SellerRecommendations -> DimSeller', COUNT_BIG(*)
  FROM dbo.SellerRecommendations r LEFT JOIN dbo.DimSeller s ON s.seller_id = r.seller_id WHERE s.seller_id IS NULL
UNION ALL SELECT 'Clickstream -> DimProduct', COUNT_BIG(*)
  FROM dbo.Clickstream k LEFT JOIN dbo.DimProduct p ON p.product_id = k.product_id WHERE p.product_id IS NULL;
```

**3. NULLs, and the em dash** (expected: product_category_name 610; product_weight_g 2;
review_score 827; review_creation_date 827; key columns 0; reasons containing an em dash 141,
which shows the text survived as UTF-8)

```sql
SELECT 'DimProduct.product_category_name' AS what, COUNT_BIG(*) AS n FROM dbo.DimProduct WHERE product_category_name IS NULL
UNION ALL SELECT 'DimProduct.product_weight_g', COUNT_BIG(*) FROM dbo.DimProduct WHERE product_weight_g IS NULL
UNION ALL SELECT 'FactOrderItems.review_score', COUNT_BIG(*) FROM dbo.FactOrderItems WHERE review_score IS NULL
UNION ALL SELECT 'FactOrderItems.review_creation_date', COUNT_BIG(*) FROM dbo.FactOrderItems WHERE review_creation_date IS NULL
UNION ALL SELECT 'FactOrderItems key columns NULL', COUNT_BIG(*) FROM dbo.FactOrderItems
  WHERE item_key IS NULL OR order_id IS NULL OR seller_id IS NULL OR product_id IS NULL OR customer_id IS NULL
UNION ALL SELECT 'because with an em dash', COUNT_BIG(*) FROM dbo.SellerRecommendations
  WHERE because LIKE N'%' + NCHAR(8212) + N'%';
```

**4. Order totals** (expected: items 110,189; distinct orders 96,470; sum of price
13,220,248.93; sum of freight 2,198,145.90; late items 8,714; first purchase 2016-09-15 12:16:38,
last 2018-08-29 15:00:37)

```sql
SELECT COUNT_BIG(*) AS items, COUNT(DISTINCT order_id) AS orders,
       SUM(price) AS sum_price, SUM(freight_value) AS sum_freight,
       SUM(CAST(is_late AS int)) AS late_items,
       MIN(order_purchase_timestamp) AS first_purchase, MAX(order_purchase_timestamp) AS last_purchase
FROM dbo.FactOrderItems;
```

**5. Late-risk scores** (expected: sum 39,763.554743; average 0.360867; minimum 0.023306;
maximum 0.911814; average logistic regression 0.481552; average decision tree 0.449317. These
are ranking scores, not probabilities: the average is far above the 7.9% late rate.)

```sql
SELECT SUM(late_risk) AS sum_late_risk, AVG(late_risk) AS avg_late_risk,
       MIN(late_risk) AS min_late_risk, MAX(late_risk) AS max_late_risk,
       AVG(late_risk_lr) AS avg_lr, AVG(late_risk_dt) AS avg_dt
FROM dbo.LatePredictions;
```

**6. Recommendations by action** (expected: Keep 2,277; Warn 234; Promote 210; Feature 174;
Suspend 75; total 2,970)

```sql
SELECT action, COUNT_BIG(*) AS sellers FROM dbo.SellerRecommendations GROUP BY action ORDER BY sellers DESC;
```

**7. Recommendations by rule** (expected: R01 69; R02 6; R03 144; R04 59; R05 31; R06 137;
R07 37; R08 106; R09 104; R10 171; R11 706; R12 1,400)

```sql
SELECT rule_id, action, COUNT_BIG(*) AS sellers FROM dbo.SellerRecommendations GROUP BY rule_id, action ORDER BY rule_id;
```

**8. Quadrants and hidden-gem evidence** (expected: Star 495; Hidden Gem 437; Overrated 439;
Overlooked-Low-Quality 493; of the 437 hidden gems 44 with confidence `normal` and 393 `low`)

```sql
SELECT quadrant, confidence, COUNT_BIG(*) AS sellers FROM dbo.DiscoveryQuadrants GROUP BY quadrant, confidence ORDER BY quadrant, confidence;
```

**9. Minimum evidence** (expected: 1,106 sellers with fewer than 5 items sold; 1,123 with fewer
than 5 items sold or fewer than 3 delivered orders; 0 of them warned or suspended)

```sql
WITH s AS (
    SELECT seller_id, COUNT_BIG(*) AS n_items, COUNT_BIG(DISTINCT order_id) AS n_orders
    FROM dbo.FactOrderItems GROUP BY seller_id)
SELECT SUM(CASE WHEN s.n_items < 5 THEN 1 ELSE 0 END) AS sellers_under_5_items,
       SUM(CASE WHEN s.n_items < 5 OR s.n_orders < 3 THEN 1 ELSE 0 END) AS sellers_thin_evidence,
       SUM(CASE WHEN (s.n_items < 5 OR s.n_orders < 3) AND r.action IN ('Warn', 'Suspend')
                THEN 1 ELSE 0 END) AS warned_or_suspended_thin
FROM s JOIN dbo.SellerRecommendations r ON r.seller_id = s.seller_id;
```

**10. The scorecard view keeps every seller** (expected: 3,095 rows; 681 with an AHP score;
1,864 with a quadrant; 2,970 with a recommendation, so 125 sellers with no delivered items have
none; `items_sold` is never empty, 125 rows have 0, and it sums to 110,189)

```sql
SELECT COUNT_BIG(*) AS sellers,
       COUNT(ahp_score_established) AS with_ahp_score,
       COUNT(quadrant) AS with_quadrant,
       COUNT(recommended_action) AS with_recommendation,
       COUNT(items_sold) AS with_items_sold,
       SUM(CAST(items_sold AS bigint)) AS total_items_sold
FROM dbo.vw_SellerScorecard;
```

## Power BI refresh

After `python -m src.load_sql` (a full reload drops and recreates the tables), open the report
in Power BI Desktop and press **Refresh** on the Home ribbon. Nothing in the `.pbix` file needs
to change unless a column was added or renamed. The report is kept outside the repository
(`powerbi/`, ignored by git).
