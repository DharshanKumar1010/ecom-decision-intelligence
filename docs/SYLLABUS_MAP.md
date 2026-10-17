# Syllabus map

Which part of the BIA 21CSE421T syllabus each piece of the project covers. Every path below
exists; `tests/test_results.py` fails if one is renamed or a listed function disappears.

## Topics implemented in the Python layer

| Unit | Syllabus topic | Where it lives | Dashboard page | Output |
|---|---|---|---|---|
| 1 | Clickstream analysis (funnel, bounce, cart abandonment) | `src/clickstream.py::funnel_conversion`, `src/clickstream.py::bounce_rate`, `src/clickstream.py::cart_abandonment_rate`, `src/clickstream.py::funnel_by_device` (synthetic input: `src/synthetic.py::generate_clickstream`) | `app/pages/1_Intelligence.py` | `data/processed/Clickstream.parquet` |
| 2 | Phases of decision making (Simon) and DSS components | `app/pages/6_DSS_Architecture.py`; the four phase pages `app/pages/1_Intelligence.py`, `app/pages/2_Design.py`, `app/pages/3_Choice.py`, `app/pages/5_Implementation.py` | `app/pages/6_DSS_Architecture.py` | the data catalog on that page |
| 3 | Predictive modeling and sensitivity analysis | `src/predict.py::build_feature_table`, `src/predict.py::grouped_stratified_folds`, `src/predict.py::out_of_fold_predictions`, `src/predict.py::_sensitivity_analysis` | `app/pages/2_Design.py` | `reports/model_metrics.json`, `reports/oof_metrics.json`, `reports/sensitivity.csv`, `data/processed/LatePredictions.parquet` |
| 3 | Sentiment analysis process and speech analytics | `src/sentiment.py::preprocess`, `src/sentiment.py::extract_features`, `src/sentiment.py::classify`, `src/sentiment.py::score`, `src/sentiment.py::aggregate_by_agent` (synthetic input: `src/synthetic.py::generate_call_transcripts`) | `app/pages/1_Intelligence.py` | `data/processed/CallSentiment.parquet`, `reports/sentiment_agent_summary.csv` |
| 4 | Multi-criteria decision making, pairwise comparison (AHP) | `src/ahp.py::compute_weights`, `src/ahp.py::build_seller_criteria`, `src/ahp.py::shrink_toward_mean`, `src/ahp.py::score_sellers` | `app/pages/3_Choice.py` | `reports/ahp_ranking.csv`, `reports/ahp_prior_strength.csv` |
| 5 | Expert systems: rules, inference, explanation | `src/expert.py::KnowledgeBase`, `src/expert.py::InferenceEngine`, `src/expert.py::ExplanationFacility`, `rules/seller_rules.yaml`, plain-language wording in `src/explain.py::near_miss` | `app/pages/5_Implementation.py` | `reports/seller_recommendations.csv` |
| 4 and 5 | Quality-versus-popularity long-tail analysis ("hidden gems") | `src/discovery.py::compute_quadrants`, `src/discovery.py::confidence_flag`, `src/discovery.py::top_hidden_gems`, `src/discovery.py::build_seller_facts` | `app/pages/4_Discovery.py` | `reports/discovery_quadrants.csv`, `reports/discovery_summary.md` |

"Healthier" in the hidden-gems discussion means sound business quality (good reviews, reliable
delivery). Olist has no food or nutrition data.

## DSS subsystems

| Subsystem | Files |
|---|---|
| Data management | `src/build_tables.py`, `src/synthetic.py`, `data/processed/`, SQL Server (`sql/schema.sql`, `src/load_sql.py`) |
| Model management | `src/predict.py`, `src/sentiment.py`, `src/ahp.py`, `src/discovery.py` |
| Knowledge base | `src/expert.py`, `rules/seller_rules.yaml`, `docs/knowledge_engineering.md` |
| User interface | `app/` (Streamlit) and the Power BI report (built by hand from `data/export/`) |

## SQL Server and Power BI layer (tutorials T1 to T15)

Course tooling practice, separate from the topics above and done by hand in SSMS and Power BI
Desktop: load with `python -m src.load_sql` (`src/load_sql.py`, schema in `sql/schema.sql`), then
the star-schema model, DAX, report pages, what-if parameter, paginated report and row-level
security. Setup notes: `docs/powerbi_setup.md`; checks to run in SSMS: `docs/MANUAL_CHECKS.md`.
