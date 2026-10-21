"""Tests for src.load_sql: schema parsing, CSV/DDL fit, safety rules, dry run, rollback.

None of these need a database except `test_loaded_database_matches_csvs`, which is marked
`sql_server` and skips itself when no SQL Server (or no loaded database) is reachable. That
test only reads; it never drops or loads anything.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from src import config, load_sql
from src.load_sql import (
    CHECKSUMS,
    CSV_FILES,
    TABLES,
    Column,
    LoaderError,
    SafetyError,
    assert_database_allowed,
    assert_statement_allowed,
    assert_table_allowed,
    build_connection_string,
    convert_column,
    load_schema,
    main,
    parse_docs_counts,
    parse_schema,
    pick_driver,
    read_raw_csv,
    select_tables,
)

EXPORTS_AVAILABLE = all((config.EXPORT_DATA_DIR / f).exists() for f in CSV_FILES.values())
needs_exports = pytest.mark.skipif(
    not EXPORTS_AVAILABLE, reason="data/export CSVs not available; run run_all.py --export"
)


@pytest.fixture(scope="module")
def schema() -> load_sql.Schema:
    return load_schema()


@pytest.fixture(scope="module")
def raw_tables() -> dict[str, pd.DataFrame]:
    if not EXPORTS_AVAILABLE:
        pytest.skip("data/export CSVs not available")
    return {t: read_raw_csv(t) for t in TABLES}


# --------------------------------------------------------------------------
# schema.sql
# --------------------------------------------------------------------------


def test_schema_parses_into_the_ten_tables_and_the_view(schema: load_sql.Schema) -> None:
    assert set(schema.tables) == set(TABLES)
    assert len(TABLES) == 10
    assert schema.create_order == TABLES
    assert schema.drop_order == tuple(reversed(TABLES))
    assert "CREATE VIEW dbo.vw_SellerScorecard" in schema.view_batch


def test_the_view_counts_every_sellers_items_from_the_fact_table(schema: load_sql.Schema) -> None:
    """items_sold must not come from the 5+ items pool (it was NULL for 1,231 sellers)."""
    view = " ".join(schema.view_batch.split())
    assert "COALESCE(i.items, 0) AS items_sold" in view
    assert "q.order_volume" not in view
    assert "FROM dbo.FactOrderItems GROUP BY seller_id" in view
    assert "FactOrderItems" in load_sql.VIEW_SOURCE_TABLES
    load_sql.assert_statement_allowed(schema.view_batch)  # the allow-list accepts the new join


def test_every_table_has_columns_and_a_key(schema: load_sql.Schema) -> None:
    for name, ddl in schema.tables.items():
        assert ddl.columns, name
        code = load_sql.strip_comments(ddl.batch).upper()
        if name == "Clickstream":
            assert "PRIMARY KEY" not in code
            assert "CREATE CLUSTERED INDEX" in code
        else:
            assert "PRIMARY KEY" in code, name


def test_foreign_keys_point_to_tables_created_earlier(schema: load_sql.Schema) -> None:
    for name, ddl in schema.tables.items():
        for parent in ddl.references:
            assert parent in TABLES
            assert TABLES.index(parent) < TABLES.index(name), f"{name} -> {parent}"
    assert "DimSeller" in schema.tables["FactOrderItems"].references
    assert "FactOrderItems" in schema.tables["LatePredictions"].references


def test_schema_text_contains_no_forbidden_statement() -> None:
    text = (config.PROJECT_ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
    code = load_sql.strip_comments(text)
    assert not re.search(r"DROP\s+DATABASE|CREATE\s+DATABASE|ALTER|xp_|sp_|EXEC", code, re.I)
    for batch in re.split(r"^\s*GO\s*$", code, flags=re.M | re.I):
        assert_statement_allowed(batch, batch=True)  # schema batches hold several statements


def test_schema_with_a_forbidden_batch_is_rejected() -> None:
    good = (config.PROJECT_ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
    with pytest.raises(SafetyError):
        parse_schema(good + "\nDROP DATABASE EcomDecisionIntelligence;\nGO\n")
    with pytest.raises(SafetyError):
        parse_schema(good + "\nEXEC sp_configure 'show advanced options', 1;\nGO\n")
    with pytest.raises(SafetyError):
        parse_schema(good.replace("USE [EcomDecisionIntelligence]", "USE [master]"))


# --------------------------------------------------------------------------
# CSV <-> DDL
# --------------------------------------------------------------------------


@needs_exports
@pytest.mark.parametrize("table", TABLES)
def test_every_csv_column_maps_to_a_ddl_column(table: str, schema: load_sql.Schema) -> None:
    header = pd.read_csv(load_sql.csv_path(table), nrows=0)
    ddl_names = [c.name for c in schema.tables[table].columns]
    assert sorted(header.columns) == sorted(ddl_names)


_INT = re.compile(r"^-?\d+$")
_DEC = re.compile(r"^-?\d+\.\d+$")
_DT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


def _data_kind(values: pd.Series) -> str:
    filled = values[values != ""]
    if filled.empty:
        return "empty"
    if filled.map(lambda x: bool(_DT.match(x))).all():
        return "datetime"
    if filled.map(lambda x: bool(_INT.match(x) or _DEC.match(x))).all():
        return "number"
    return "text"


@needs_exports
@pytest.mark.parametrize("table", TABLES)
def test_ddl_types_fit_and_match_the_data(
    table: str, schema: load_sql.Schema, raw_tables: dict[str, pd.DataFrame]
) -> None:
    raw = raw_tables[table]
    for col in schema.tables[table].columns:
        kind = _data_kind(raw[col.name])
        family = load_sql.type_family(col.sql_type)
        allowed = {"datetime": {"datetime"}, "number": {"int", "decimal"}, "text": {"text"}}
        if kind != "empty":
            assert family in allowed[kind], f"{table}.{col.name}: {kind} data in {col.sql_type}"
        blanks = int((raw[col.name] == "").sum())
        if blanks:
            assert col.nullable, f"{table}.{col.name} has {blanks} blanks but is NOT NULL"
        # Raises if any value is too long, rounded, out of range or the wrong shape.
        convert_column(raw[col.name], col, table)


@needs_exports
def test_text_columns_are_at_least_as_wide_as_the_data(
    schema: load_sql.Schema, raw_tables: dict[str, pd.DataFrame]
) -> None:
    for table, ddl in schema.tables.items():
        for col in ddl.columns:
            if load_sql.type_family(col.sql_type) == "text":
                longest = int(raw_tables[table][col.name].str.len().max())
                width = load_sql.parse_type(col.sql_type)[1][0]
                assert width >= longest, f"{table}.{col.name}"


@needs_exports
def test_docs_row_counts_match_the_csvs(raw_tables: dict[str, pd.DataFrame]) -> None:
    docs = parse_docs_counts()
    assert set(docs) == set(TABLES)
    for table, frame in raw_tables.items():
        assert docs[table] == len(frame), table


# --------------------------------------------------------------------------
# convert_column
# --------------------------------------------------------------------------


def _series(*values: str) -> pd.Series:
    return pd.Series(list(values), dtype=str)


def test_convert_column_good_values() -> None:
    assert convert_column(_series("1", "5.000000", ""), Column("x", "tinyint", True), "T") == [
        1,
        5,
        None,
    ]
    assert convert_column(_series("58.900000"), Column("x", "decimal(12,2)", False), "T") == [
        Decimal("58.90")
    ]
    assert convert_column(_series("ab", ""), Column("x", "nvarchar(8)", True), "T") == ["ab", None]
    stamps = convert_column(
        _series("2018-01-02T03:04:05", ""), Column("x", "datetime2(0)", True), "T"
    )
    assert stamps[0].year == 2018 and stamps[1] is None


@pytest.mark.parametrize(
    ("values", "column"),
    [
        (("toolongvalue",), Column("x", "nvarchar(8)", False)),
        (("",), Column("x", "nvarchar(8)", False)),
        (("1.5",), Column("x", "int", False)),
        (("300",), Column("x", "tinyint", False)),
        (("1.234567",), Column("x", "decimal(9,2)", False)),
        (("12345678.00",), Column("x", "decimal(9,2)", False)),
        (("2018-13-45T00:00:00",), Column("x", "datetime2(0)", False)),
    ],
)
def test_convert_column_never_truncates_or_rounds_silently(
    values: tuple[str, ...], column: Column
) -> None:
    with pytest.raises((LoaderError, ValueError)):
        convert_column(_series(*values), column, "T")


# --------------------------------------------------------------------------
# Safety rules
# --------------------------------------------------------------------------


def test_only_the_allowed_database_name_passes() -> None:
    assert_database_allowed("EcomDecisionIntelligence")
    for name in ("master", "tempdb", "ecomdecisionintelligence", "EcomDecisionIntelligence2", ""):
        with pytest.raises(SafetyError):
            assert_database_allowed(name)
        with pytest.raises(SafetyError):
            build_connection_string(r"localhost\SQLEXPRESS", name, "ODBC Driver 18 for SQL Server")


def test_a_table_outside_the_list_is_refused() -> None:
    for table in TABLES:
        assert_table_allowed(table)
    for bad in ("sys.objects", "dbo.DimSeller", "Orders", "dimseller", "DimSeller; DROP TABLE x"):
        with pytest.raises(SafetyError):
            assert_table_allowed(bad)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP DATABASE EcomDecisionIntelligence",
        "ALTER TABLE dbo.DimSeller ADD x int",
        "EXEC xp_cmdshell 'dir'",
        "EXEC sp_configure 'show advanced options', 1",
        "RECONFIGURE",
        "CREATE LOGIN bob WITH PASSWORD = 'x'",
        "CREATE DATABASE Other",
        "DROP TABLE dbo.DimSeller",
        "DROP TABLE IF EXISTS dbo.SomethingElse",
        "DROP TABLE IF EXISTS sys.objects",
        "DELETE FROM dbo.DimSeller",
        "TRUNCATE TABLE dbo.DimSeller",
        "UPDATE dbo.DimSeller SET seller_state = 'XX'",
        "USE master",
        "SELECT name FROM sys.databases",
        "SELECT @@VERSION",
        "SELECT 1",
        "SELECT * FROM dbo.DimSeller, sys.objects",
        "INSERT INTO dbo.Other (a) VALUES (1)",
        "GRANT CONTROL TO bob",
    ],
)
def test_statements_outside_the_allow_list_are_refused(sql: str) -> None:
    with pytest.raises(SafetyError):
        assert_statement_allowed(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * INTO dbo.DimSellerCopy FROM dbo.DimSeller",
        "SELECT seller_id INTO #temp FROM dbo.DimSeller",
        "SELECT seller_id INTO dbo.FactOrderItems FROM dbo.DimSeller",
        "SELECT seller_id FROM dbo.DimSeller INTO dbo.AhpRanking",
        "SELECT COUNT_BIG(*) FROM dbo.DimSeller WHERE seller_id IN (SELECT seller_id FROM #t)",
        "INSERT INTO #temp (a) VALUES (1)",
        "DROP TABLE IF EXISTS #temp",
        "CREATE TABLE #t (a int)",
        "EXEC('SELECT 1')",
        "EXECUTE sp_who",
        "exec ('DROP TABLE dbo.DimSeller')",
        "SELECT COUNT_BIG(*) FROM dbo.DimSeller; SELECT COUNT_BIG(*) FROM dbo.DimProduct",
        "SELECT COUNT_BIG(*) FROM dbo.DimSeller; DROP TABLE IF EXISTS dbo.DimSeller",
        "DROP TABLE IF EXISTS dbo.DimSeller; DROP TABLE IF EXISTS dbo.DimProduct",
        "INSERT INTO dbo.CallSentiment ([call_id]) VALUES (?); "
        "INSERT INTO dbo.Clickstream ([a]) VALUES (?)",
    ],
)
def test_into_temp_tables_exec_and_multiple_statements_are_refused(sql: str) -> None:
    with pytest.raises(SafetyError):
        assert_statement_allowed(sql)


def test_a_single_trailing_semicolon_is_still_one_statement() -> None:
    assert_statement_allowed("SELECT COUNT_BIG(*) FROM dbo.FactOrderItems;")
    assert_statement_allowed("DROP TABLE IF EXISTS dbo.DimSeller ;")
    with pytest.raises(SafetyError):
        assert_statement_allowed(
            "SELECT COUNT_BIG(*) FROM dbo.FactOrderItems;;SELECT 1 FROM dbo.DimSeller"
        )


def test_a_ddl_batch_may_hold_several_statements_only_when_declared_a_batch() -> None:
    batch = "CREATE TABLE dbo.DimSeller (a int); CREATE INDEX IX_a ON dbo.DimSeller (a);"
    assert_statement_allowed(batch, batch=True)
    with pytest.raises(SafetyError):
        assert_statement_allowed(batch)
    with pytest.raises(SafetyError):  # a batch is still checked statement by statement
        assert_statement_allowed(batch + " SELECT a INTO #t FROM dbo.DimSeller", batch=True)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE IF EXISTS dbo.DimSeller",
        "DROP VIEW IF EXISTS dbo.vw_SellerScorecard",
        "SELECT COUNT_BIG(*) FROM dbo.FactOrderItems",
        "INSERT INTO dbo.CallSentiment ([call_id]) VALUES (?)",
        "CREATE INDEX IX_a ON dbo.DimSeller (seller_state)",
        "USE [EcomDecisionIntelligence]",
    ],
)
def test_the_loaders_own_statements_are_allowed(sql: str) -> None:
    assert_statement_allowed(sql)


def test_connection_string_uses_windows_authentication_and_holds_no_secret() -> None:
    conn_str = build_connection_string(
        r"localhost\SQLEXPRESS", "EcomDecisionIntelligence", "ODBC Driver 18 for SQL Server"
    )
    assert "Trusted_Connection=yes" in conn_str
    assert not re.search(r"\b(UID|PWD|Password|User ?Id)\b", conn_str, re.I)
    assert r"SERVER=localhost\SQLEXPRESS" in conn_str


def test_pick_driver_prefers_18_then_17() -> None:
    both = ["SQL Server", "ODBC Driver 17 for SQL Server", "ODBC Driver 18 for SQL Server"]
    assert pick_driver(both) == "ODBC Driver 18 for SQL Server"
    assert pick_driver(["ODBC Driver 17 for SQL Server"]) == "ODBC Driver 17 for SQL Server"
    with pytest.raises(LoaderError):
        pick_driver(["SQL Server"])


class _NoConnection:
    """Replaces the connection functions: any attempt to connect fails the test."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a database connection was attempted")


@pytest.fixture
def no_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(load_sql, "connect", _NoConnection())
    monkeypatch.setattr(load_sql.pyodbc, "connect", _NoConnection())


def test_wrong_database_is_refused_before_any_connection(no_connection: None) -> None:
    assert main(["--database", "master"]) == load_sql.EXIT_REFUSED
    assert main(["--database", "master", "--dry-run"]) == load_sql.EXIT_REFUSED


def test_table_outside_the_list_is_refused_by_the_cli(no_connection: None) -> None:
    assert main(["--only", "sys.objects", "--dry-run"]) == load_sql.EXIT_REFUSED


@needs_exports
def test_dry_run_makes_no_connection_and_no_writes(
    no_connection: None, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--dry-run", "--only", "CallSentiment", "--only", "AhpRanking"])
    out = capsys.readouterr().out
    assert code == load_sql.EXIT_OK
    assert "DRY RUN: no connection opened" in out
    assert "dbo.CallSentiment" in out and "dbo.AhpRanking" in out
    assert "FactOrderItems" not in out


@needs_exports
def test_only_refuses_a_table_that_others_depend_on(schema: load_sql.Schema) -> None:
    with pytest.raises(LoaderError, match="LatePredictions"):
        select_tables(schema, ["FactOrderItems"])
    with pytest.raises(LoaderError, match="FactOrderItems"):
        select_tables(schema, ["DimSeller"])
    assert select_tables(schema, ["LatePredictions", "FactOrderItems"]) == [
        "FactOrderItems",
        "LatePredictions",
    ]
    assert select_tables(schema, ["CallSentiment"]) == ["CallSentiment"]


# --------------------------------------------------------------------------
# Load and rollback with a fake connection (no server)
# --------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn
        self.fast_executemany = False
        self._answer = 0

    def execute(self, sql: str, *params: Any) -> FakeCursor:
        # DDL batches from schema.sql hold several statements; everything else must be one.
        code = load_sql.strip_comments(sql).lstrip()
        is_ddl_batch = code.startswith(("CREATE TABLE", "CREATE VIEW", "USE"))
        load_sql.assert_statement_allowed(sql, batch=is_ddl_batch)
        self.conn.executed.append(sql)
        if sql.startswith("SELECT"):
            self._answer = self.conn.answers.get(sql, 0)
        return self

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        self.conn.inserted.append((sql, len(rows)))

    def setinputsizes(self, sizes: Any) -> None:
        self.conn.sizes = list(sizes)

    def fetchone(self) -> tuple[int]:
        return (self._answer,)


class FakeConnection:
    def __init__(self, answers: dict[str, int]) -> None:
        self.answers = answers
        self.executed: list[str] = []
        self.inserted: list[tuple[str, int]] = []
        self.sizes: list[Any] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def _answers_for(table: str, rows: int, checksum_delta: int = 0) -> dict[str, int]:
    raw = read_raw_csv(table)
    return {
        f"SELECT COUNT_BIG(*) FROM dbo.{table}": rows,
        CHECKSUMS[table].sql: CHECKSUMS[table].expected(raw) + checksum_delta,
    }


@needs_exports
def test_a_good_load_commits_and_prints_ok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeConnection(_answers_for("CallSentiment", 300))
    monkeypatch.setattr(load_sql, "connect", lambda server, database: fake)
    assert main(["--only", "CallSentiment"]) == load_sql.EXIT_OK
    assert fake.executed[0] == "DROP TABLE IF EXISTS dbo.CallSentiment"
    assert not any("DROP VIEW" in s for s in fake.executed)  # partial load leaves the view alone
    assert fake.inserted and fake.inserted[0][1] == 300
    assert fake.commits == 2 and fake.rollbacks == 0 and fake.closed
    out = capsys.readouterr().out
    assert "CallSentiment" in out and "OK" in out and "300" in out


@needs_exports
@pytest.mark.parametrize(("rows", "delta"), [(299, 0), (300, 1)])
def test_a_row_count_or_checksum_mismatch_rolls_back_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, rows: int, delta: int
) -> None:
    fake = FakeConnection(_answers_for("CallSentiment", rows, delta))
    monkeypatch.setattr(load_sql, "connect", lambda server, database: fake)
    assert main(["--only", "CallSentiment"]) == load_sql.EXIT_FAILED
    assert fake.rollbacks == 1
    assert fake.commits == 1  # only the drop phase committed; the table was rolled back


@needs_exports
def test_only_the_selected_tables_are_ever_touched(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {**_answers_for("CallSentiment", 300), **_answers_for("Clickstream", 100_000)}
    fake = FakeConnection(answers)
    monkeypatch.setattr(load_sql, "connect", lambda server, database: fake)
    assert main(["--only", "CallSentiment", "--only", "Clickstream"]) == load_sql.EXIT_OK
    acted_on = r"(?:DROP (?:TABLE|VIEW) IF EXISTS|CREATE TABLE|INSERT INTO|FROM|ON) dbo\.(\w+)"
    touched = set(re.findall(acted_on, " ".join(fake.executed)))
    assert touched == {"CallSentiment", "Clickstream"}


@needs_exports
def test_a_connection_failure_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(server: str, database: str) -> Any:
        raise LoaderError("cannot connect")

    monkeypatch.setattr(load_sql, "connect", refuse)
    assert main(["--only", "CallSentiment"]) == load_sql.EXIT_NO_CONNECTION


# --------------------------------------------------------------------------
# Integration: only if a SQL Server with the loaded database is reachable
# --------------------------------------------------------------------------


@pytest.mark.sql_server
@needs_exports
def test_loaded_database_matches_csvs() -> None:
    try:
        conn = load_sql.connect(load_sql.DEFAULT_SERVER, load_sql.ALLOWED_DATABASE)
    except (LoaderError, SafetyError) as exc:
        pytest.skip(f"no reachable SQL Server: {exc}")
    try:
        plan = load_sql.prepare(None, convert=False)
        results = load_sql.run_verify(conn, plan)
    finally:
        conn.close()
    if any("cannot read table" in r.detail for r in results):
        pytest.skip("database reachable but not all tables are loaded yet")
    bad = [(r.table, r.detail) for r in results if r.status != "OK"]
    assert not bad, bad


@pytest.mark.sql_server
@needs_exports
def test_scorecard_view_has_an_items_sold_for_every_seller() -> None:
    try:
        conn = load_sql.connect(load_sql.DEFAULT_SERVER, load_sql.ALLOWED_DATABASE)
    except (LoaderError, SafetyError) as exc:
        pytest.skip(f"no reachable SQL Server: {exc}")
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT COUNT_BIG(*) FROM dbo.vw_SellerScorecard WHERE items_sold IS NULL"
            )
            nulls = int(cursor.fetchone()[0])
            cursor.execute(
                "SELECT COUNT_BIG(*) FROM dbo.vw_SellerScorecard v JOIN (SELECT seller_id, "
                "COUNT_BIG(*) AS n FROM dbo.FactOrderItems GROUP BY seller_id) f "
                "ON f.seller_id = v.seller_id WHERE v.items_sold <> f.n"
            )
            wrong = int(cursor.fetchone()[0])
        except Exception as exc:  # view not (re)loaded yet
            pytest.skip(f"view not readable: {exc}")
    finally:
        conn.close()
    assert nulls == 0 and wrong == 0
