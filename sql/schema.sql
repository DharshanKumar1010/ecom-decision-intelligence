/*
================================================================================
 schema.sql : tables, keys, indexes and one view for the E-Commerce DSS export
================================================================================
 Database : EcomDecisionIntelligence   (must already exist; this file never
            creates or drops a database)
 Loaded by: python -m src.load_sql      (it parses this file; keep the markers
            "-- @drops", "-- @table: <Name>" and "-- @view: <Name>" and the
            "GO" batch separators exactly as they are)
 Also runs on its own in SSMS: open it, connect to the database above and
 press Execute. It drops and recreates ONLY the 10 tables and the view named
 below, in dependency order, and leaves every other object alone.

 Source : the 10 flat files in data/export/ written by python -m src.export_sql
          (UTF-8, comma delimited, ISO datetimes YYYY-MM-DDTHH:MM:SS, a blank
          cell is NULL, floats written with 6 decimals).

 Type choices (from a profile of the real files; see BUILD_LOG.md, Stage 7):
   * text    : nvarchar, about 2x the longest value seen, rounded up. No
               nvarchar(max): nothing needs it, and it cannot be indexed and
               slows pyodbc bulk inserts.
   * numbers : int / smallint / tinyint where every value is a whole number,
               decimal(p,s) where the data has a fixed number of decimals
               (price and freight 2, scores and risks 6). No float, so sums
               are exact and a re-export that adds precision fails loudly in
               the loader instead of rounding silently.
   * NOT NULL only where the real data has no blank cell.

 Foreign keys: every candidate was checked on the real data (orphan child rows):
     FactOrderItems.seller_id         -> DimSeller        0 of 110,189
     FactOrderItems.product_id        -> DimProduct       0 of 110,189
     FactOrderItems.customer_id       -> DimCustomer      0 of 110,189
     LatePredictions.item_key         -> FactOrderItems   0 of 110,189
     AhpRanking.seller_id             -> DimSeller        0 of 681
     DiscoveryQuadrants.seller_id     -> DimSeller        0 of 1,864
     SellerRecommendations.seller_id  -> DimSeller        0 of 2,970
     Clickstream.product_id           -> DimProduct       0 of 100,000
   All eight passed and are declared below. Nothing was skipped.

 Collation: the database default (case-insensitive). All ids are lower-case
 hex, so the primary keys stay unique; the loader would fail loudly otherwise.
================================================================================
*/

USE [EcomDecisionIntelligence];
GO

-- @drops
-- Children first, parents last (reverse of the create order below).
DROP VIEW  IF EXISTS dbo.vw_SellerScorecard;
DROP TABLE IF EXISTS dbo.SellerRecommendations;
DROP TABLE IF EXISTS dbo.DiscoveryQuadrants;
DROP TABLE IF EXISTS dbo.AhpRanking;
DROP TABLE IF EXISTS dbo.Clickstream;
DROP TABLE IF EXISTS dbo.CallSentiment;
DROP TABLE IF EXISTS dbo.LatePredictions;
DROP TABLE IF EXISTS dbo.FactOrderItems;
DROP TABLE IF EXISTS dbo.DimCustomer;
DROP TABLE IF EXISTS dbo.DimProduct;
DROP TABLE IF EXISTS dbo.DimSeller;
GO

-- ============================================================================
-- Dimensions
-- ============================================================================

-- @table: DimSeller
CREATE TABLE dbo.DimSeller (
    seller_id     nvarchar(64) NOT NULL,   -- 32-char hex id
    seller_state  nvarchar(8)  NOT NULL,   -- 2-letter Brazilian state (row-level security key)
    CONSTRAINT PK_DimSeller PRIMARY KEY (seller_id)
);
CREATE INDEX IX_DimSeller_state ON dbo.DimSeller (seller_state);
GO

-- @table: DimProduct
CREATE TABLE dbo.DimProduct (
    product_id                     nvarchar(64)  NOT NULL,
    product_category_name          nvarchar(128) NULL,      -- 610 products have no category
    product_weight_g               int           NULL,      -- whole grams; 2 products have none
    product_category_name_english  nvarchar(128) NOT NULL,  -- translation, filled for every product
    CONSTRAINT PK_DimProduct PRIMARY KEY (product_id)
);
GO

-- @table: DimCustomer
CREATE TABLE dbo.DimCustomer (
    customer_id     nvarchar(64) NOT NULL,
    customer_state  nvarchar(8)  NOT NULL,
    CONSTRAINT PK_DimCustomer PRIMARY KEY (customer_id)
);
GO

-- ============================================================================
-- Facts
-- ============================================================================

-- @table: FactOrderItems
CREATE TABLE dbo.FactOrderItems (
    item_key                       nvarchar(80)  NOT NULL,   -- order_id + '_' + order_item_id
    order_id                       nvarchar(64)  NOT NULL,
    order_item_id                  smallint      NOT NULL,   -- 1..21 today
    product_id                     nvarchar(64)  NOT NULL,
    seller_id                      nvarchar(64)  NOT NULL,
    customer_id                    nvarchar(64)  NOT NULL,
    price                          decimal(12,2) NOT NULL,   -- BRL
    freight_value                  decimal(12,2) NOT NULL,   -- BRL
    order_status                   nvarchar(32)  NOT NULL,   -- always 'delivered' in this export
    order_purchase_timestamp       datetime2(0)  NOT NULL,
    order_delivered_customer_date  datetime2(0)  NOT NULL,
    order_estimated_delivery_date  datetime2(0)  NOT NULL,
    order_date                     datetime2(0)  NOT NULL,
    delivery_days                  int           NOT NULL,
    is_late                        tinyint       NOT NULL,   -- 1 = delivered after the estimate
    review_score                   tinyint       NULL,       -- 1..5; 827 items have no review
    review_creation_date           datetime2(0)  NULL,
    CONSTRAINT PK_FactOrderItems PRIMARY KEY (item_key),
    CONSTRAINT CK_FactOrderItems_is_late CHECK (is_late IN (0, 1)),
    CONSTRAINT FK_FactOrderItems_DimSeller   FOREIGN KEY (seller_id)   REFERENCES dbo.DimSeller (seller_id),
    CONSTRAINT FK_FactOrderItems_DimProduct  FOREIGN KEY (product_id)  REFERENCES dbo.DimProduct (product_id),
    CONSTRAINT FK_FactOrderItems_DimCustomer FOREIGN KEY (customer_id) REFERENCES dbo.DimCustomer (customer_id)
);
-- Join and slicer paths used by the Power BI model.
CREATE INDEX IX_FactOrderItems_seller_id   ON dbo.FactOrderItems (seller_id);
CREATE INDEX IX_FactOrderItems_product_id  ON dbo.FactOrderItems (product_id);
CREATE INDEX IX_FactOrderItems_customer_id ON dbo.FactOrderItems (customer_id);
CREATE INDEX IX_FactOrderItems_order_id    ON dbo.FactOrderItems (order_id);
CREATE INDEX IX_FactOrderItems_order_date  ON dbo.FactOrderItems (order_date);
GO

-- @table: LatePredictions
-- late_risk is an out-of-fold RANKING score (random forest), not a calibrated
-- probability: its median is about 0.34 against a 7.9 % late rate.
CREATE TABLE dbo.LatePredictions (
    item_key      nvarchar(80)  NOT NULL,
    late_risk     decimal(9,6)  NOT NULL,   -- primary model (random forest)
    late_risk_lr  decimal(9,6)  NOT NULL,   -- logistic regression, kept for interpretability
    late_risk_dt  decimal(9,6)  NOT NULL,   -- decision tree, kept for interpretability
    CONSTRAINT PK_LatePredictions PRIMARY KEY (item_key),
    CONSTRAINT FK_LatePredictions_FactOrderItems FOREIGN KEY (item_key) REFERENCES dbo.FactOrderItems (item_key)
);
GO

-- @table: CallSentiment
-- SYNTHETIC data (Olist has no call-centre data). Label it as such in Power BI.
CREATE TABLE dbo.CallSentiment (
    call_id          nvarchar(16)  NOT NULL,
    agent_id         nvarchar(16)  NOT NULL,
    transcript       nvarchar(256) NOT NULL,   -- longest seen: 107 characters, contains commas
    hold_seconds     decimal(9,1)  NOT NULL,
    silence_pct      decimal(7,4)  NOT NULL,   -- share of the call that was silent, 0..1
    compound         decimal(7,4)  NOT NULL,   -- VADER compound score, -1..1
    sentiment_label  nvarchar(32)  NOT NULL,
    CONSTRAINT PK_CallSentiment PRIMARY KEY (call_id)
);
GO

-- @table: Clickstream
-- SYNTHETIC data (Olist has no clickstream). 70,000 sessions over 100,000 events,
-- so session_id repeats: a clustered index, not a primary key.
CREATE TABLE dbo.Clickstream (
    session_id  nvarchar(32)  NOT NULL,
    event_type  nvarchar(32)  NOT NULL,   -- page_view / add_to_cart / checkout / purchase
    product_id  nvarchar(64)  NOT NULL,
    device      nvarchar(32)  NOT NULL,
    source      nvarchar(32)  NOT NULL,
    event_time  datetime2(0)  NOT NULL,
    CONSTRAINT FK_Clickstream_DimProduct FOREIGN KEY (product_id) REFERENCES dbo.DimProduct (product_id)
);
CREATE CLUSTERED INDEX CIX_Clickstream_session_id ON dbo.Clickstream (session_id);
CREATE INDEX IX_Clickstream_product_id ON dbo.Clickstream (product_id);
GO

-- ============================================================================
-- Seller-level model outputs (one row per seller, join to DimSeller on seller_id)
-- ============================================================================

-- @table: AhpRanking
-- Established sellers only (at least 30 items sold). avg_review_score / on_time_rate
-- are the shrunk values used for scoring; the *_raw columns are the plain averages.
CREATE TABLE dbo.AhpRanking (
    seller_id                      nvarchar(64)  NOT NULL,
    seller_state                   nvarchar(8)   NOT NULL,
    avg_review_score               decimal(9,6)  NOT NULL,
    on_time_rate                   decimal(9,6)  NOT NULL,
    avg_price                      decimal(14,6) NOT NULL,
    order_volume                   int           NOT NULL,   -- items sold (order lines)
    avg_review_raw                 decimal(9,6)  NOT NULL,
    on_time_rate_raw               decimal(9,6)  NOT NULL,
    n_reviewed                     int           NOT NULL,   -- distinct reviewed orders
    n_orders                       int           NOT NULL,   -- distinct delivered orders
    norm_avg_review_score          decimal(9,6)  NOT NULL,
    norm_on_time_rate              decimal(9,6)  NOT NULL,
    norm_avg_price                 decimal(9,6)  NOT NULL,
    norm_order_volume              decimal(9,6)  NOT NULL,
    contribution_avg_review_score  decimal(9,6)  NOT NULL,
    contribution_on_time_rate      decimal(9,6)  NOT NULL,
    contribution_avg_price         decimal(9,6)  NOT NULL,
    contribution_order_volume      decimal(9,6)  NOT NULL,
    ahp_score                      decimal(9,6)  NOT NULL,
    CONSTRAINT PK_AhpRanking PRIMARY KEY (seller_id),
    CONSTRAINT FK_AhpRanking_DimSeller FOREIGN KEY (seller_id) REFERENCES dbo.DimSeller (seller_id)
);
GO

-- @table: DiscoveryQuadrants
-- Sellers with at least 5 items sold. Low-confidence rows (fewer than 15 items)
-- must not be presented with the certainty of the main ranking.
CREATE TABLE dbo.DiscoveryQuadrants (
    seller_id     nvarchar(64)  NOT NULL,
    ahp_score     decimal(9,6)  NOT NULL,
    order_volume  int           NOT NULL,   -- items sold
    quadrant      nvarchar(64)  NOT NULL,   -- Star / Hidden Gem / Overrated / Overlooked-Low-Quality
    confidence    nvarchar(32)  NOT NULL,   -- normal / low
    CONSTRAINT PK_DiscoveryQuadrants PRIMARY KEY (seller_id),
    CONSTRAINT FK_DiscoveryQuadrants_DimSeller FOREIGN KEY (seller_id) REFERENCES dbo.DimSeller (seller_id)
);
CREATE INDEX IX_DiscoveryQuadrants_quadrant ON dbo.DiscoveryQuadrants (quadrant);
GO

-- @table: SellerRecommendations
CREATE TABLE dbo.SellerRecommendations (
    seller_id  nvarchar(64)  NOT NULL,
    action     nvarchar(32)  NOT NULL,   -- Keep / Warn / Suspend / Feature / Promote
    rule_id    nvarchar(16)  NOT NULL,   -- R01..R12
    because    nvarchar(512) NOT NULL,   -- plain-language reason; longest seen 227, has commas and an em dash
    CONSTRAINT PK_SellerRecommendations PRIMARY KEY (seller_id),
    CONSTRAINT FK_SellerRecommendations_DimSeller FOREIGN KEY (seller_id) REFERENCES dbo.DimSeller (seller_id)
);
CREATE INDEX IX_SellerRecommendations_action ON dbo.SellerRecommendations (action);
GO

-- ============================================================================
-- View for Power BI: one row per seller, nothing dropped (LEFT JOINs)
-- ============================================================================

-- @view: vw_SellerScorecard
-- Sellers missing from a model table (for example the 2,414 sellers below the
-- 30-item AHP threshold) keep their row with NULLs in that table's columns.
-- The two score columns are different scores (30+ items vs 5+ items): keep both.
CREATE VIEW dbo.vw_SellerScorecard AS
SELECT
    s.seller_id,
    s.seller_state,
    a.ahp_score          AS ahp_score_established,
    a.avg_review_score   AS ahp_avg_review_score,
    a.on_time_rate       AS ahp_on_time_rate,
    a.avg_review_raw,
    a.on_time_rate_raw,
    a.n_orders           AS established_n_orders,
    q.ahp_score          AS ahp_score_discovery,
    q.order_volume       AS items_sold,
    q.quadrant,
    q.confidence,
    r.action             AS recommended_action,
    r.rule_id            AS recommendation_rule_id,
    r.because            AS recommendation_reason
FROM dbo.DimSeller AS s
LEFT JOIN dbo.AhpRanking            AS a ON a.seller_id = s.seller_id
LEFT JOIN dbo.DiscoveryQuadrants    AS q ON q.seller_id = s.seller_id
LEFT JOIN dbo.SellerRecommendations AS r ON r.seller_id = s.seller_id;
GO
