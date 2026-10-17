# Documentation index

| Document | What it is for |
|---|---|
| [`../README.md`](../README.md) | What the project does, how to set it up and run it, its limitations. Start here. |
| [`../reports/RESULTS.md`](../reports/RESULTS.md) | Every number (model scores, row counts, quadrant, rule and action counts), generated from the pipeline outputs. Quote numbers from here only. |
| [`knowledge_engineering.md`](knowledge_engineering.md) | Why each threshold has the value it has: AHP and discovery cut-offs, expert-system rules (section 6), shrinkage, the order-grouped evaluation, and the minimum-evidence rule (section 10). |
| [`SYLLABUS_MAP.md`](SYLLABUS_MAP.md) | Which syllabus topic lives in which file, function and dashboard page. |
| [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | The five-minute walkthrough and the twelve questions a reviewer is likely to ask. |
| [`powerbi_setup.md`](powerbi_setup.md) | The ten exported tables, how to load them into SQL Server (`python -m src.load_sql`), and the Power BI model relationships. |
| [`powerbi_measures.md`](powerbi_measures.md) | The seven DAX measures with the value each card should show, and the eight relationships to expect in Model view. |
| [`MANUAL_CHECKS.md`](MANUAL_CHECKS.md) | What to look at in the browser, page by page, and ten read-only queries to run in SSMS with their expected results. |
| [`../sql/schema.sql`](../sql/schema.sql) | The SQL Server tables, keys, indexes and the seller scorecard view; the source of truth for table definitions. |
| [`archive/BUILD_LOG_full.md`](archive/BUILD_LOG_full.md) | The complete stage-by-stage build log as it stood before Stage 8 (661 lines). The short [`../BUILD_LOG.md`](../BUILD_LOG.md) summarises it. |
| [`archive/AUDIT_REPORT.md`](archive/AUDIT_REPORT.md) | The Stage 7 audit of the repository (23 findings). Its DDL block is superseded by `sql/schema.sql`. |
| [`archive/CLEANUP_REPORT.md`](archive/CLEANUP_REPORT.md) | What Stage 8 verified, changed and left for decision, with before/after numbers. |
| [`archive/FINAL_POLISH_REPORT.md`](archive/FINAL_POLISH_REPORT.md) | The last polish round before Power BI: page and Technical-details fixes, the 5-item gate measurement, the checks run and the commit commands. |
| [`../reports/diagnostics/`](../reports/diagnostics/) | One-off experiments behind past decisions; each report starts with a line saying whether it is current or historical. |
