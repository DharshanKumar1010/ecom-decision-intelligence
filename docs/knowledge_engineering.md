# Knowledge Engineering: Threshold Justifications

Every threshold below was checked against the real output of `python run_all.py`
on this project's actual `data/processed/` tables (2,970 sellers total, 110,189
fact rows), not invented. Percentiles are computed from `SellerFacts.parquet`
(written by `src.discovery.build_seller_facts`) and `FactOrderItems.parquet`
unless stated otherwise. If the pipeline is rerun on a different data
snapshot, these numbers should be recomputed and this file updated —
CLAUDE.md section 1 forbids quoting a number that wasn't actually measured.

## 1. AHP established-sellers threshold: `AHP_MIN_ORDERS = 30`

Per-seller order counts across all 2,970 sellers:

| Percentile | order_volume |
|---|---|
| 25th | 2 |
| 50th (median) | 8 |
| 75th | 26 |
| 90th | 83 |
| 95th | 151 |

30 orders sits at the **77.9th percentile** of all sellers — i.e. it selects
the top ~22.9% (681 of 2,970) as "established" for the main AHP ranking.
This was chosen to keep the established-seller ranking restricted to sellers
with enough of a track record that an average-review/on-time-rate estimate
is meaningful, while still leaving a majority of the marketplace (the other
77%) for the lower-threshold discovery pass. 30 orders is also comfortably
above the point (order_volume >= ~10-15) where an empirical rate like
`on_time_rate` stops being dominated by single-order noise.

Resulting AHP run: 681 established sellers, weights `avg_review_score=0.473,
on_time_rate=0.284, avg_price=0.073, order_volume=0.170` (from the pairwise
matrix in `config.AHP_PAIRWISE_MATRIX`), consistency ratio **CR = 0.0189**,
well under the `AHP_CR_THRESHOLD = 0.10` flag.

## 2. Discovery AHP-eligibility threshold: `DISCOVERY_MIN_ORDERS = 5`

5 orders sits at the **41.9th percentile** — it makes 1,864 of 2,970 sellers
(62.8%) eligible for a quality (AHP) score, versus 22.9% at the 30-order
established-seller threshold. This is the whole point of the discovery
feature (CLAUDE.md section 7.7.1): a seller needs *some* orders for an
average review score and on-time rate to mean anything at all, but 5 is low
enough that small/newer sellers — the ones a "sort by volume" view would
bury — are actually included, rather than only re-running the same
established-seller list at a different name.

## 3. Discovery confidence cutoff: `DISCOVERY_CONFIDENCE_MIN_ORDERS = 15`

15 orders sits at the **66.1st percentile** of all sellers. Among the 1,864
AHP-eligible (>=5 order) sellers specifically, the median order count is
**18** — meaning the 15-order confidence cutoff sits just below the middle
of the eligible population, splitting it into **818 "low" confidence rows
(43.9%) and 1,046 "normal" (56.1%)** on the last real run. This threshold
was picked so "low confidence" genuinely means "below-typical order count
for an eligible seller" (not an arbitrary round number): a seller just past
the 5-order eligibility floor is treated with reduced confidence until they
clear roughly the eligible population's own midpoint.

## 4. Discovery quadrant thresholds: median split (both axes)

`config.DISCOVERY_QUALITY_THRESHOLD` / `DISCOVERY_POPULARITY_THRESHOLD` are
both `None` by default, meaning `compute_quadrants` splits each axis at its
own median among AHP-eligible sellers — the simplest split that guarantees
roughly balanced quadrants without hand-picking a magic number for either
axis. On the last real run this produced: **Star 454, Hidden Gem 478,
Overrated 480, Overlooked-Low-Quality 452** — a near-even four-way split,
confirming the median default doesn't collapse the classification onto one
dominant label.

## 5. Promote late-risk cutoffs (rules R08, R09)

`avg_late_risk` (mean predicted `late_risk` per seller from `src.predict`'s
logistic regression) across all 2,970 sellers:

| Percentile | avg_late_risk |
|---|---|
| min | 0.414 |
| 10th | 0.447 |
| 25th | 0.474 |
| 50th (median) | 0.503 |
| 75th | 0.527 |
| 90th | 0.542 |
| max | 0.837 |

**R08** uses `avg_late_risk < 0.3`, the exact value CLAUDE.md section 7.7.2
specifies for the Promote rule. Checked against the real distribution above,
**no observed seller currently falls below 0.3** (the minimum is 0.414):
the deliberately small, non-date feature set (`price`, `freight_value`,
`product_weight_g`, `same_state`; ROC-AUC ~0.57) only weakly separates risk,
so predicted probabilities stay in a fairly narrow band around 0.41-0.57
rather than spanning the full [0, 1] range a stronger model might. R08 is
kept verbatim because CLAUDE.md specifies it exactly, it is exercised by a
constructed fact set in `tests/test_expert.py`, and it documents an honest
limitation (CLAUDE.md section 1: never claim more separation than the model
actually has) rather than silently dropping the mandated rule.

**R09** is the second, realistic Promote variant: `avg_late_risk < 0.474`,
the **25th percentile** of the real distribution — "safer quartile of
predicted risk." This threshold does fire on real data (132 of the 478
Hidden Gem sellers on the last run) and is what actually drives the
`Promote` action in `reports/seller_recommendations.csv` today.

## 6. Remaining expert-system rule thresholds (`rules/seller_rules.yaml`)

All checked against the same `SellerFacts.parquet` run (2,970 sellers;
`avg_review` percentiles: min 1.0, 10th 3.2, 25th 3.90, 50th 4.27, 75th 4.70,
90th-95th 5.0; `order_volume` percentiles as in section 1;
`late_rate` percentiles: 50th 0.0, 75th 0.10, 90th 0.222, 95th 0.337;
`ahp_score` percentiles as in section 1's established-seller pool extended
to the 5-order-eligible pool: 10th 0.604, 25th 0.669, 50th 0.716, 75th
0.755, 90th 0.789).

| Rule | Threshold | Justification | Real matches |
|---|---|---|---|
| R01 Suspend | `avg_late_risk > 0.53` (~75th pct) AND `avg_review < 3.5` (between 10th/25th pct) | Combines a top-quartile risk signal with a clearly-below-average review score — CLAUDE.md's own example rule, re-thresholded to this data's real percentiles instead of its illustrative 0.6/3.0. | 94 sellers |
| R02 Suspend | `late_rate > 0.4` AND `order_volume >= 10` | An empirical (not predicted) late rate above 40% is a severe, directly-observed failure; requiring >=10 orders rules out a fluke from 1-2 late shipments. | 7 sellers |
| R06 Feature | `ahp_score >= 0.789` (90th pct) AND `avg_late_risk < 0.503` (median) | Top-decile AHP quality with at-or-below-median predicted risk — a proven, low-risk top performer. | 107 sellers |
| R07 Feature | `avg_review >= 4.3` AND `order_volume >= 83` (90th pct) | High-volume sellers (90th percentile+) have a *lower* mean review (4.08) than the overall population (4.27 median) in this data, so requiring 4.3 specifically picks out sellers sustaining quality at scale rather than everyone at that volume. | 69 sellers |
| R03 Warn | `avg_late_risk > 0.55` (~90th pct) | A single, wide risk-only caution signal, deliberately set higher than R01's combined-condition threshold since it fires alone. | 226 sellers |
| R04 Warn | `avg_review < 3.5` AND `order_volume >= 5` | Same review cutoff as R01 (10th-25th pct band) without the risk condition, gated on order_volume >= 5 so it reflects a pattern, not one bad review. | 177 sellers |
| R05 Warn | `late_rate > 0.25` (90th pct) AND `order_volume >= 5` | A softer, single-condition version of R02's empirical-lateness signal. | 105 sellers |
| R10 Keep | `avg_late_risk <= 0.503` AND `avg_review >= 4.27` (both median) AND `order_volume >= 5` | At-or-better-than-median on both risk and reviews, with a real order history — no signal to act on. | 446 sellers |
| R11 Keep | `ahp_score >= 0.604` (10th pct) AND `late_rate <= 0.10` (75th pct) | Excludes only the bottom decile of AHP quality and the worst quartile of empirical lateness — a wide "no red flags" net. | 1,264 sellers |
| R12 (default) | none (`if: []`) | Every seller must get a recommendation (CLAUDE.md section 7.5); catches sellers with too little data to trigger any of the above (frequently order_volume < 5, so no `ahp_score`). | 1,228 sellers (41.4% of all sellers) |

**Priority ladder** (rationale for the ordering, not just the numbers):
Suspend (100, 98) > Feature (90, 88) > Promote (80, 78) > Warn (70, 68, 66)
> Keep (50, 48) > default (1). Suspend and Feature intentionally outrank
Promote and Warn so that a severe risk/review problem or a proven top
performer is never overridden by a milder signal. One known edge case,
documented rather than hidden: because `is_hidden_gem` sellers are chosen
via the AHP composite (which weights `avg_review_score` at 0.473, the
largest single weight), a seller matching R09 (Promote) with `avg_review <
3.5` is mathematically possible but very unlikely in practice — the AHP
score already penalizes poor reviews heavily before a seller can qualify as
a Hidden Gem.

## 7. Sentiment thresholds

`SENTIMENT_POSITIVE_THRESHOLD = 0.05` / `SENTIMENT_NEGATIVE_THRESHOLD =
-0.05` are VADER's own published default compound-score cutoffs, not tuned
against this project's synthetic call transcripts — `CallTranscripts` has no
persisted ground-truth sentiment label to tune against (see
`src.synthetic`'s docstring and `src.sentiment`'s module docstring), so
using the library's validated defaults rather than an untuned guess is the
honest choice here.
