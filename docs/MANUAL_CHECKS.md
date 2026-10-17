# Manual checks

Things only a person can judge, plus ten queries to run in SSMS. Automated checks are
`.\tasks.ps1 verify`; numbers are in `reports/RESULTS.md`. Start the app with
`.\tasks.ps1 app` (or `streamlit run app/Home.py`).

## In the browser, page by page

**Home**
- The title asks which sellers to back and which are being overlooked; the page list in the
  sidebar has six pages.
- A sentence says the clickstream and call data are synthetic.

**1 Intelligence** ("What is happening in the marketplace?")
- Four figures: delivered orders, item revenue, late-delivery rate, average review score. They
  should equal `reports/RESULTS.md` and the SQL sums below (late rate about 7.9%).
- Funnel and call-sentiment charts carry a visible **SYNTHETIC** tag. Funnel: roughly 21 of
  every 100 visits add to the cart, 13 reach checkout, 9 buy.
- Sentiment is described as coming from transcripts only (no audio).

**2 Design** ("How likely is an order to arrive late, and why?")
- Headline "Ranking score" of about 74% with a sentence saying 50% is a coin flip.
- Wording: the page says the score *ranks* orders. It must not say "forecast", "predicts
  future deliveries" or "probability". The out-of-fold note (every score comes from a model
  that never saw that order) is present, and so is the note that the model has little skill on
  future months.
- The bars show what the model leans on (month of purchase first), and a sentence says this
  is not what causes lateness. The "What if an input changes?" chart moves when you change the
  selection.

**3 Choice** ("Which established sellers are best overall?")
- 681 sellers ranked, minimum 30 items sold, consistency ratio 0.019.
- In Technical details, change one comparison box: the bars move. Make the judgments
  contradict each other and the sentence says the weights should not be trusted.

**4 Discovery** ("Which good sellers are we overlooking?")
- The four quadrant counts line equals `reports/RESULTS.md` (Star, Hidden Gem, Overrated,
  Overlooked-Low-Quality).
- Tier labels read **Stronger evidence** and **Early signal**; Early-signal gems are drawn as
  hollow markers; the four quadrant colours can be told apart (also in a greyscale screenshot).
- Wording says "items sold", never "orders", for a seller's volume. Nothing mentions food or
  nutrition except to say "healthier" means business quality.

**5 Implementation** ("What should we do about this seller?")
- Opens on a seller with at least 5 items sold where a real rule fired (not the default
  rule R12). The recommendation is one word, with a tick/cross list of the rule's conditions.
- "Try an example" switches the seller to the first one with each outcome.
- Technical details holds the full rule table, the "what would change the outcome" near-miss
  box, and the comparison with the stored batch result (they must agree).
- Paste the id of any seller with fewer than 5 items sold: the outcome is Keep from the default
  rule, "pending more data". No seller with fewer than 5 items sold is ever Warn or Suspend.

**6 DSS Architecture** ("How does the system fit together?")
- Data tables 16, expert rules 12; the four boxes (data, models, knowledge, interface) read in
  order; the data catalog's row counts equal the pipeline files.

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
review_score 827; review_creation_date 827; key columns 0; reasons containing an em dash 1,149,
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

**6. Recommendations by action** (expected: Keep 2,269; Warn 241; Promote 210; Feature 174;
Suspend 76; total 2,970)

```sql
SELECT action, COUNT_BIG(*) AS sellers FROM dbo.SellerRecommendations GROUP BY action ORDER BY sellers DESC;
```

**7. Recommendations by rule** (expected: R01 70; R02 6; R03 144; R04 66; R05 31; R06 137;
R07 37; R08 106; R09 104; R10 170; R11 701; R12 1,398)

```sql
SELECT rule_id, action, COUNT_BIG(*) AS sellers FROM dbo.SellerRecommendations GROUP BY rule_id, action ORDER BY rule_id;
```

**8. Quadrants and hidden-gem evidence** (expected: Star 495; Hidden Gem 437; Overrated 439;
Overlooked-Low-Quality 493; of the 437 hidden gems 44 with confidence `normal` and 393 `low`)

```sql
SELECT quadrant, confidence, COUNT_BIG(*) AS sellers FROM dbo.DiscoveryQuadrants GROUP BY quadrant, confidence ORDER BY quadrant, confidence;
```

**9. Minimum evidence** (expected: 1,106 sellers with fewer than 5 items sold; 0 of them
warned or suspended)

```sql
WITH items AS (SELECT seller_id, COUNT_BIG(*) AS n FROM dbo.FactOrderItems GROUP BY seller_id)
SELECT SUM(CASE WHEN i.n < 5 THEN 1 ELSE 0 END) AS sellers_under_5_items,
       SUM(CASE WHEN i.n < 5 AND r.action IN ('Warn', 'Suspend') THEN 1 ELSE 0 END) AS warned_or_suspended_under_5
FROM items i JOIN dbo.SellerRecommendations r ON r.seller_id = i.seller_id;
```

**10. The scorecard view keeps every seller** (expected: 3,095 rows; 681 with an AHP score;
1,864 with a quadrant; 2,970 with a recommendation, so 125 sellers with no delivered items have
none)

```sql
SELECT COUNT_BIG(*) AS sellers,
       COUNT(ahp_score_established) AS with_ahp_score,
       COUNT(quadrant) AS with_quadrant,
       COUNT(recommended_action) AS with_recommendation
FROM dbo.vw_SellerScorecard;
```

## Power BI refresh

After `python -m src.load_sql` (a full reload drops and recreates the tables), open the report
in Power BI Desktop and press **Refresh** on the Home ribbon. Nothing in the `.pbix` file needs
to change unless a column was added or renamed. The report is kept outside the repository
(`powerbi/`, ignored by git).
