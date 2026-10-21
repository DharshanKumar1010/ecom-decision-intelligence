"""SQL Server loader for the 10 exported CSVs (owner-authorised exception).

CLAUDE.md sections 1 and 10 forbid writing SQL or opening a database connection
from Claude Code. The owner lifted that rule for exactly three files: this
module, `sql/schema.sql` and `tests/test_load_sql.py`. Nothing else in the
project may do this.

What it does: reads the CSVs in `data/export/`, recreates the 10 tables from
`sql/schema.sql` in dependency order, bulk loads them with pyodbc
`fast_executemany` (one transaction per table) and verifies every table: the SQL
row count must equal the CSV row count and the count in `docs/powerbi_setup.md`,
and one exact checksum column must match. Any mismatch rolls that table back and
the process exits non-zero.

Safety rules (enforced in code, covered by tests):
- Windows Authentication only (`Trusted_Connection=yes`); no password, user name
  or secret exists anywhere. The default server is `localhost\\SQLEXPRESS`.
- Only the database `EcomDecisionIntelligence` is accepted; any other name is
  refused before a connection is attempted. The database must already exist; this
  module never creates or drops one.
- Only the 10 named tables (and the view `vw_SellerScorecard`) are ever dropped or
  created. Every statement is checked against an allow-list first, so
  `DROP DATABASE`, `ALTER`, `EXEC`, `sp_configure`, `xp_cmdshell`, logins and the
  like cannot be sent even if `schema.sql` were edited to contain them.
- `--dry-run` opens no connection. `--verify-only` runs SELECTs only.

Usage:
    python -m src.load_sql --dry-run
    python -m src.load_sql                    # load everything
    python -m src.load_sql --only CallSentiment --only Clickstream
    python -m src.load_sql --verify-only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pyodbc

from src.config import DOCS_DIR, EXPORT_DATA_DIR, PROJECT_ROOT, get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Constants (kept here, not in config.py, so the exemption stays in three files)
# --------------------------------------------------------------------------

ALLOWED_DATABASE = "EcomDecisionIntelligence"
DEFAULT_SERVER = r"localhost\SQLEXPRESS"
SERVER_ENV_VAR = "ECOM_SQL_SERVER"
PREFERRED_DRIVERS: tuple[str, ...] = (
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 17 for SQL Server",
)

# Dependency order: parents before children. Drops run in the reverse order.
TABLES: tuple[str, ...] = (
    "DimSeller",
    "DimProduct",
    "DimCustomer",
    "FactOrderItems",
    "LatePredictions",
    "CallSentiment",
    "Clickstream",
    "AhpRanking",
    "DiscoveryQuadrants",
    "SellerRecommendations",
)
VIEW_NAME = "vw_SellerScorecard"
VIEW_SOURCE_TABLES: frozenset[str] = frozenset(
    {"DimSeller", "AhpRanking", "DiscoveryQuadrants", "SellerRecommendations", "FactOrderItems"}
)
CSV_FILES: dict[str, str] = {
    "DimSeller": "DimSeller.csv",
    "DimProduct": "DimProduct.csv",
    "DimCustomer": "DimCustomer.csv",
    "FactOrderItems": "FactOrderItems.csv",
    "LatePredictions": "LatePredictions.csv",
    "CallSentiment": "CallSentiment.csv",
    "Clickstream": "Clickstream.csv",
    "AhpRanking": "ahp_ranking.csv",
    "DiscoveryQuadrants": "discovery_quadrants.csv",
    "SellerRecommendations": "seller_recommendations.csv",
}
SCHEMA_PATH: Path = PROJECT_ROOT / "sql" / "schema.sql"
DOCS_PATH: Path = DOCS_DIR / "powerbi_setup.md"
BATCH_SIZE = 10_000
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_NO_CONNECTION = 3


class LoaderError(Exception):
    """A load or verification problem (data does not fit the schema, count mismatch...)."""


class SafetyError(LoaderError):
    """A request or statement outside what this loader is allowed to do."""


# --------------------------------------------------------------------------
# Safety guards
# --------------------------------------------------------------------------

_ALLOWED_OBJECTS: frozenset[str] = frozenset(TABLES) | {VIEW_NAME}
_FORBIDDEN = re.compile(
    r"\b(ALTER|EXEC(?:UTE)?|GRANT|REVOKE|DENY|TRUNCATE|DELETE|UPDATE|MERGE|BACKUP|RESTORE|"
    r"SHUTDOWN|RECONFIGURE|OPENROWSET|OPENQUERY|BULK|"
    r"DROP\s+DATABASE|CREATE\s+(?:DATABASE|LOGIN|USER|ROLE|SCHEMA|PROCEDURE|FUNCTION|TRIGGER)|"
    r"sp_\w+|xp_\w+)\b",
    re.IGNORECASE,
)
_ALLOWED_STATEMENTS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^CREATE\s+TABLE\s+dbo\.(\w+)\s*\(", re.IGNORECASE),
    re.compile(
        r"^CREATE\s+(?:UNIQUE\s+)?(?:CLUSTERED\s+|NONCLUSTERED\s+)?INDEX\s+\w+\s+ON\s+dbo\.(\w+)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^CREATE\s+VIEW\s+dbo\.(\w+)\s+AS\b", re.IGNORECASE),
    re.compile(r"^DROP\s+(?:TABLE|VIEW)\s+IF\s+EXISTS\s+dbo\.(\w+)$", re.IGNORECASE),
    re.compile(r"^INSERT\s+INTO\s+dbo\.(\w+)\s*\(", re.IGNORECASE),
    re.compile(r"^SELECT\b", re.IGNORECASE),
)
_OTHER_DATABASE_OBJECT = re.compile(
    r"\b(?:sys|INFORMATION_SCHEMA|master|msdb|tempdb|model)\s*\.|@@", re.IGNORECASE
)
_DBO_REFERENCE = re.compile(r"\bdbo\.(\w+)", re.IGNORECASE)
_USE_STATEMENT = re.compile(r"^USE\s+\[?(\w+)\]?$", re.IGNORECASE)


def assert_database_allowed(name: str) -> None:
    """Refuse any database other than `ALLOWED_DATABASE` (exact, case-sensitive match).

    Raises:
        SafetyError: if `name` differs from the allowed database name.
    """
    if name != ALLOWED_DATABASE:
        raise SafetyError(
            f"refusing to run against database {name!r}: only {ALLOWED_DATABASE!r} is allowed"
        )


def assert_table_allowed(name: str) -> None:
    """Refuse any table that is not one of the 10 named tables (bare name, exact match).

    Raises:
        SafetyError: if `name` is not in `TABLES`.
    """
    if name not in TABLES:
        raise SafetyError(f"table {name!r} is not one of the 10 loader tables: {', '.join(TABLES)}")


def strip_comments(sql: str) -> str:
    """Remove `/* ... */` and `-- ...` comments from a SQL batch."""
    no_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", no_block)


def split_statements(sql: str) -> list[str]:
    """Split a comment-free batch on `;` into stripped, non-empty statements."""
    return [part.strip() for part in strip_comments(sql).split(";") if part.strip()]


_INTO = re.compile(r"\bINTO\b", re.IGNORECASE)
_INSERT_INTO = re.compile(r"^INSERT\s+INTO\s+dbo\.\w+\s*\(", re.IGNORECASE)


def assert_statement_allowed(sql: str, *, batch: bool = False) -> None:
    """Check every statement in `sql` against the allow-list.

    Only CREATE TABLE/INDEX/VIEW, DROP TABLE/VIEW IF EXISTS, INSERT and SELECT are
    permitted, and only on the 10 tables and the one view. Keywords that could change
    anything outside those objects are rejected wherever they appear. Also rejected: any
    temp-table marker (`#`), `EXEC`/`EXECUTE`, `SELECT ... INTO` (it creates a table), and,
    unless `batch=True`, more than one statement (a single trailing semicolon is fine).
    `batch=True` is for the DDL batches read from schema.sql, which hold several statements.

    Raises:
        SafetyError: on the first statement that is not allowed.
    """
    statements = split_statements(sql)
    if not batch and len(statements) > 1:
        raise SafetyError(f"only one statement per call is allowed, got {len(statements)}")
    for statement in statements:
        flat = " ".join(statement.split())
        if "#" in flat:
            raise SafetyError(f"temp-table marker '#' is not allowed: {flat[:80]}")
        if _INTO.search(flat) and not _INSERT_INTO.match(flat):
            raise SafetyError(f"INTO is only allowed in the loader's own INSERT: {flat[:80]}")
        use = _USE_STATEMENT.match(flat)
        if use:
            if use.group(1) != ALLOWED_DATABASE:
                raise SafetyError(f"USE of database {use.group(1)!r} is not allowed")
            continue
        forbidden = _FORBIDDEN.search(flat)
        if forbidden:
            raise SafetyError(f"forbidden keyword {forbidden.group(0)!r} in: {flat[:80]}")
        if not any(p.match(flat) for p in _ALLOWED_STATEMENTS):
            raise SafetyError(f"statement type not allowed: {flat[:80]}")
        if _OTHER_DATABASE_OBJECT.search(flat):
            raise SafetyError(f"system or cross-database object referenced: {flat[:80]}")
        if flat.upper().startswith("SELECT") and not re.search(r"\bFROM\s+dbo\.\w+", flat, re.I):
            raise SafetyError(f"SELECT must read one of the loader's objects: {flat[:80]}")
        for obj in _DBO_REFERENCE.findall(flat):
            if obj not in _ALLOWED_OBJECTS:
                raise SafetyError(f"statement touches {obj!r}, not one of the loader's objects")


# --------------------------------------------------------------------------
# Schema parsing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """One column of a DDL table."""

    name: str
    sql_type: str
    nullable: bool


@dataclass(frozen=True)
class TableDdl:
    """A table block from schema.sql."""

    name: str
    columns: tuple[Column, ...]
    batch: str
    references: tuple[str, ...]


@dataclass(frozen=True)
class Schema:
    """Everything parsed from schema.sql."""

    tables: dict[str, TableDdl]
    view_batch: str
    create_order: tuple[str, ...]
    drop_order: tuple[str, ...]


_MARKER = re.compile(r"^--\s*@(drops|table|view)(?::\s*(\w+))?\s*$", re.MULTILINE)
_GO = re.compile(r"^\s*GO\s*$", re.MULTILINE | re.IGNORECASE)
_COLUMN = re.compile(
    r"^(?P<name>\w+)\s+(?P<type>[A-Za-z0-9]+(?:\s*\(\s*[\w,\s]+\s*\))?)\s*(?P<null>NOT\s+NULL|NULL)?",
    re.IGNORECASE,
)


def _split_top_level(body: str) -> list[str]:
    """Split a column list on commas that are not inside parentheses."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_create_table(name: str, batch: str) -> tuple[tuple[Column, ...], tuple[str, ...]]:
    """Extract columns and referenced tables from a table block."""
    clean = strip_comments(batch)
    match = re.search(rf"CREATE\s+TABLE\s+dbo\.{name}\s*\(", clean, re.IGNORECASE)
    if not match:
        raise LoaderError(f"no CREATE TABLE dbo.{name} in its schema.sql block")
    start = match.end()
    depth = 1
    end = start
    while depth and end < len(clean):
        depth += {"(": 1, ")": -1}.get(clean[end], 0)
        end += 1
    columns: list[Column] = []
    references: list[str] = []
    for part in _split_top_level(clean[start : end - 1]):
        if re.match(r"^CONSTRAINT\b", part, re.IGNORECASE):
            ref = re.search(r"REFERENCES\s+dbo\.(\w+)", part, re.IGNORECASE)
            if ref:
                references.append(ref.group(1))
            continue
        col = _COLUMN.match(part)
        if not col:
            raise LoaderError(f"cannot parse column definition in {name}: {part[:60]}")
        sql_type = re.sub(r"\s+", "", col.group("type")).lower()
        nullable = not (col.group("null") and col.group("null").upper().startswith("NOT"))
        columns.append(Column(col.group("name"), sql_type, nullable))
    return tuple(columns), tuple(references)


def parse_schema(text: str) -> Schema:
    """Parse schema.sql into table blocks, the view block and the drop order.

    Every batch is run through `assert_statement_allowed`; a batch with no marker may
    contain only comments and `USE [EcomDecisionIntelligence]`.

    Raises:
        SafetyError: if a batch holds a statement that is not allowed.
        LoaderError: if the file does not define exactly the expected tables and view.
    """
    tables: dict[str, TableDdl] = {}
    view_batch = ""
    drops: list[str] = []
    for batch in _GO.split(text):
        marker = _MARKER.search(batch)
        assert_statement_allowed(batch, batch=True)
        if marker is None:
            continue
        kind, name = marker.group(1), marker.group(2)
        if kind == "table":
            assert_table_allowed(name)
            columns, references = _parse_create_table(name, batch)
            tables[name] = TableDdl(name, columns, batch, references)
        elif kind == "view":
            if name != VIEW_NAME:
                raise SafetyError(f"unexpected view {name!r}")
            view_batch = batch
        else:
            drops = re.findall(r"DROP\s+TABLE\s+IF\s+EXISTS\s+dbo\.(\w+)", batch, re.IGNORECASE)
    if set(tables) != set(TABLES) or not view_batch:
        raise LoaderError(f"schema.sql must define exactly {sorted(TABLES)} and {VIEW_NAME}")
    create_order = tuple(tables)
    if create_order != TABLES or tuple(drops) != tuple(reversed(TABLES)):
        raise LoaderError("schema.sql create/drop order does not follow the dependency order")
    return Schema(tables, view_batch, create_order, tuple(drops))


def load_schema(path: Path = SCHEMA_PATH) -> Schema:
    """Read and parse `sql/schema.sql`."""
    return parse_schema(path.read_text(encoding="utf-8"))


def dependents(schema: Schema, table: str) -> list[str]:
    """Tables whose foreign keys reference `table`."""
    return [t.name for t in schema.tables.values() if table in t.references]


# --------------------------------------------------------------------------
# Selecting tables
# --------------------------------------------------------------------------


def select_tables(schema: Schema, only: Sequence[str] | None) -> list[str]:
    """Validate `--only` names and return the selected tables in dependency order.

    Raises:
        SafetyError: for a name outside the 10 tables.
        LoaderError: if a selected table is referenced by an unselected table (it could
            not be dropped without breaking that table's foreign key).
    """
    if not only:
        return list(TABLES)
    for name in only:
        assert_table_allowed(name)
    selected = [t for t in TABLES if t in set(only)]
    for name in selected:
        blocking = [d for d in dependents(schema, name) if d not in selected]
        if blocking:
            raise LoaderError(
                f"{name} is referenced by {', '.join(blocking)}; load those too "
                f"(--only {name} --only {' --only '.join(blocking)}) or run a full load"
            )
    return selected


# --------------------------------------------------------------------------
# Reading the CSVs and converting them to typed Python values
# --------------------------------------------------------------------------

_TYPE = re.compile(r"^([a-z0-9]+)(?:\(([^)]*)\))?$")
_INT_RANGES: dict[str, tuple[int, int]] = {
    "tinyint": (0, 255),
    "smallint": (-32768, 32767),
    "int": (-(2**31), 2**31 - 1),
    "bigint": (-(2**63), 2**63 - 1),
}


def parse_type(sql_type: str) -> tuple[str, tuple[int, ...]]:
    """Split `decimal(12,2)` into (`decimal`, (12, 2))."""
    match = _TYPE.match(sql_type)
    if not match:
        raise LoaderError(f"unsupported SQL type {sql_type!r}")
    args = tuple(int(a) for a in match.group(2).split(",")) if match.group(2) else ()
    return match.group(1), args


def type_family(sql_type: str) -> str:
    """Return `text`, `int`, `decimal` or `datetime` for a DDL type."""
    base, _ = parse_type(sql_type)
    if base in ("nvarchar", "varchar", "nchar", "char"):
        return "text"
    if base in _INT_RANGES:
        return "int"
    if base == "decimal":
        return "decimal"
    if base == "datetime2":
        return "datetime"
    raise LoaderError(f"unsupported SQL type {sql_type!r}")


def csv_path(table: str, export_dir: Path = EXPORT_DATA_DIR) -> Path:
    """Path of the export CSV for `table`."""
    assert_table_allowed(table)
    return export_dir / CSV_FILES[table]


def read_raw_csv(table: str, export_dir: Path = EXPORT_DATA_DIR) -> pd.DataFrame:
    """Read a CSV with every cell as text and blanks kept as empty strings."""
    path = csv_path(table, export_dir)
    if not path.exists():
        raise LoaderError(f"{path} not found; run python -m src.export_sql first")
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
    )


def convert_column(values: pd.Series, col: Column, table: str) -> list[Any]:
    """Convert one text column to Python values matching the DDL type.

    Nothing is rounded or truncated silently: a value that does not fit the declared type
    (too long, not a whole number, more decimals than the scale, out of range) raises.

    Raises:
        LoaderError: on a blank in a NOT NULL column or a value that does not fit.
    """
    where = f"{table}.{col.name}"
    raw: list[str] = values.tolist()
    if not col.nullable and any(v == "" for v in raw):
        raise LoaderError(f"{where} is NOT NULL but the CSV has blank cells")
    base, args = parse_type(col.sql_type)
    family = type_family(col.sql_type)
    if family == "text":
        longest = max((len(v) for v in raw), default=0)
        if longest > args[0]:
            raise LoaderError(f"{where}: value of {longest} characters does not fit {col.sql_type}")
        return [v if v != "" else None for v in raw]
    if family == "int":
        low, high = _INT_RANGES[base]
        ints: list[Any] = []
        for v in raw:
            if v == "":
                ints.append(None)
                continue
            number = Decimal(v)
            if number != number.to_integral_value() or not low <= number <= high:
                raise LoaderError(f"{where}: {v!r} is not a whole number within {col.sql_type}")
            ints.append(int(number))
        return ints
    if family == "decimal":
        precision, scale = args
        step = Decimal(1).scaleb(-scale)
        limit = Decimal(10) ** (precision - scale)
        decimals: list[Any] = []
        for v in raw:
            if v == "":
                decimals.append(None)
                continue
            number = Decimal(v)
            fitted = number.quantize(step, rounding=ROUND_HALF_UP)
            if fitted != number or abs(fitted) >= limit:
                raise LoaderError(f"{where}: {v!r} does not fit {col.sql_type} without rounding")
            decimals.append(fitted)
        return decimals
    parsed = pd.to_datetime(values.mask(values == ""), format=DATETIME_FORMAT, errors="raise")
    return [None if pd.isna(t) else t.to_pydatetime() for t in parsed]


def convert_table(raw: pd.DataFrame, ddl: TableDdl) -> list[tuple[Any, ...]]:
    """Convert a whole CSV to a list of row tuples in DDL column order.

    Columns are matched by name, so their order in the CSV does not matter.

    Raises:
        LoaderError: if the CSV columns and the DDL columns differ.
    """
    csv_cols = list(raw.columns)
    ddl_cols = [c.name for c in ddl.columns]
    if sorted(csv_cols) != sorted(ddl_cols):
        raise LoaderError(f"{ddl.name}: CSV columns {csv_cols} differ from DDL columns {ddl_cols}")
    converted = [convert_column(raw[c.name], c, ddl.name) for c in ddl.columns]
    return list(zip(*converted, strict=True))


# --------------------------------------------------------------------------
# Verification: counts and one exact checksum per table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Checksum:
    """A SQL aggregate and the same number computed from the CSV text."""

    description: str
    sql: str
    expected: Callable[[pd.DataFrame], int]


def _scaled_sum(column: str, factor: int) -> Callable[[pd.DataFrame], int]:
    def compute(df: pd.DataFrame) -> int:
        values = (Decimal(v) * factor for v in df[column] if v != "")
        return int(sum((x.to_integral_value() for x in values), Decimal(0)))

    return compute


def _length_sum(column: str) -> Callable[[pd.DataFrame], int]:
    def compute(df: pd.DataFrame) -> int:
        return sum(len(v.rstrip(" ").encode("utf-16-le")) // 2 for v in df[column])

    return compute


def _seconds_sum(column: str) -> Callable[[pd.DataFrame], int]:
    def compute(df: pd.DataFrame) -> int:
        stamps = pd.to_datetime(df[column], format=DATETIME_FORMAT)
        return int((stamps - pd.Timestamp("2000-01-01")).dt.total_seconds().sum())

    return compute


CHECKSUMS: dict[str, Checksum] = {
    "DimSeller": Checksum(
        "SUM(LEN(seller_state))",
        "SELECT COALESCE(SUM(CAST(LEN(seller_state) AS bigint)), 0) FROM dbo.DimSeller",
        _length_sum("seller_state"),
    ),
    "DimProduct": Checksum(
        "SUM(product_weight_g)",
        "SELECT COALESCE(SUM(CAST(product_weight_g AS bigint)), 0) FROM dbo.DimProduct",
        _scaled_sum("product_weight_g", 1),
    ),
    "DimCustomer": Checksum(
        "SUM(LEN(customer_state))",
        "SELECT COALESCE(SUM(CAST(LEN(customer_state) AS bigint)), 0) FROM dbo.DimCustomer",
        _length_sum("customer_state"),
    ),
    "FactOrderItems": Checksum(
        "SUM(price x 100)",
        "SELECT COALESCE(SUM(CAST(price * 100 AS bigint)), 0) FROM dbo.FactOrderItems",
        _scaled_sum("price", 100),
    ),
    "LatePredictions": Checksum(
        "SUM(late_risk x 1e6)",
        "SELECT COALESCE(SUM(CAST(late_risk * 1000000 AS bigint)), 0) FROM dbo.LatePredictions",
        _scaled_sum("late_risk", 1_000_000),
    ),
    "CallSentiment": Checksum(
        "SUM(hold_seconds x 10)",
        "SELECT COALESCE(SUM(CAST(hold_seconds * 10 AS bigint)), 0) FROM dbo.CallSentiment",
        _scaled_sum("hold_seconds", 10),
    ),
    "Clickstream": Checksum(
        "SUM(seconds since 2000-01-01 of event_time)",
        "SELECT COALESCE(SUM(DATEDIFF_BIG(second, '2000-01-01', event_time)), 0) "
        "FROM dbo.Clickstream",
        _seconds_sum("event_time"),
    ),
    "AhpRanking": Checksum(
        "SUM(ahp_score x 1e6)",
        "SELECT COALESCE(SUM(CAST(ahp_score * 1000000 AS bigint)), 0) FROM dbo.AhpRanking",
        _scaled_sum("ahp_score", 1_000_000),
    ),
    "DiscoveryQuadrants": Checksum(
        "SUM(ahp_score x 1e6)",
        "SELECT COALESCE(SUM(CAST(ahp_score * 1000000 AS bigint)), 0) FROM dbo.DiscoveryQuadrants",
        _scaled_sum("ahp_score", 1_000_000),
    ),
    "SellerRecommendations": Checksum(
        "SUM(LEN(because))",
        "SELECT COALESCE(SUM(CAST(LEN(because) AS bigint)), 0) FROM dbo.SellerRecommendations",
        _length_sum("because"),
    ),
}

_DOCS_ROW = re.compile(r"^\|\s*`[^`]+\.csv`\s*\|\s*`(\w+)`\s*\|.*\|\s*([\d,]+)\s*\|\s*$")


def parse_docs_counts(path: Path = DOCS_PATH) -> dict[str, int]:
    """Row counts per SQL Server table from the table in docs/powerbi_setup.md."""
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _DOCS_ROW.match(line)
        if match:
            counts[match.group(1)] = int(match.group(2).replace(",", ""))
    return counts


def _scalar(cursor: Any, sql: str) -> int:
    assert_statement_allowed(sql)
    cursor.execute(sql)
    return int(cursor.fetchone()[0])


def verify_table(cursor: Any, table: str, raw: pd.DataFrame, docs_count: int) -> str:
    """Compare SQL against the CSV and the docs; return "" if all agree, else the problem."""
    csv_rows = len(raw)
    sql_rows = _scalar(cursor, f"SELECT COUNT_BIG(*) FROM dbo.{table}")
    if not sql_rows == csv_rows == docs_count:
        return f"row count: SQL {sql_rows:,}, CSV {csv_rows:,}, docs {docs_count:,}"
    check = CHECKSUMS[table]
    want = check.expected(raw)
    got = _scalar(cursor, check.sql)
    if got != want:
        return f"checksum {check.description}: SQL {got}, CSV {want}"
    return ""


# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------


def pick_driver(available: Sequence[str]) -> str:
    """Prefer ODBC Driver 18, then 17 for SQL Server.

    Raises:
        LoaderError: if neither is installed.
    """
    for driver in PREFERRED_DRIVERS:
        if driver in available:
            return driver
    raise LoaderError(f"no SQL Server ODBC driver found; installed: {list(available)}")


def build_connection_string(server: str, database: str, driver: str) -> str:
    """ODBC connection string using Windows Authentication; no secret of any kind.

    `TrustServerCertificate=yes` is a client-side setting for the local Express instance's
    self-signed certificate; it changes nothing on the server.
    """
    assert_database_allowed(database)
    return (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
        "Trusted_Connection=yes;TrustServerCertificate=yes;Connection Timeout=15"
    )


def connect(server: str, database: str) -> Any:
    """Open a Windows-authenticated connection with autocommit off.

    Raises:
        SafetyError: for a database other than the allowed one (before any connection).
        LoaderError: if the server cannot be reached or the login is rejected.
    """
    assert_database_allowed(database)
    conn_str = build_connection_string(server, database, pick_driver(pyodbc.drivers()))
    try:
        return pyodbc.connect(conn_str, autocommit=False, timeout=15)
    except pyodbc.Error as exc:
        raise LoaderError(f"cannot connect to {server} / {database}: {exc}") from exc


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class TableResult:
    """One row of the final table."""

    table: str
    loaded: int = 0
    expected: int = 0
    status: str = "SKIPPED"
    seconds: float = 0.0
    detail: str = ""


@dataclass
class Plan:
    """Everything prepared before the database is touched."""

    schema: Schema
    selected: list[str]
    raw: dict[str, pd.DataFrame]
    docs: dict[str, int]
    rows: dict[str, list[tuple[Any, ...]]] = field(default_factory=dict)


def _input_sizes(columns: Sequence[Column]) -> list[tuple[int, int, int]]:
    """pyodbc parameter types for fast_executemany, one per column."""
    sizes: list[tuple[int, int, int]] = []
    for col in columns:
        family = type_family(col.sql_type)
        _, args = parse_type(col.sql_type)
        if family == "text":
            sizes.append((pyodbc.SQL_WVARCHAR, args[0], 0))
        elif family == "int":
            sizes.append((pyodbc.SQL_INTEGER, 0, 0))
        elif family == "decimal":
            sizes.append((pyodbc.SQL_DECIMAL, args[0], args[1]))
        else:
            sizes.append((pyodbc.SQL_TYPE_TIMESTAMP, 19, 0))
    return sizes


def insert_statement(ddl: TableDdl) -> str:
    """Parameterised INSERT for a table."""
    names = ", ".join(f"[{c.name}]" for c in ddl.columns)
    marks = ", ".join("?" for _ in ddl.columns)
    return f"INSERT INTO dbo.{ddl.name} ({names}) VALUES ({marks})"


def prepare(only: Sequence[str] | None, *, convert: bool = True) -> Plan:
    """Read and validate everything that does not need the database.

    Checks the schema, the CSV columns and values against the DDL, and each CSV's row
    count against docs/powerbi_setup.md, so a problem is found before any table is dropped.

    Raises:
        LoaderError / SafetyError: on any problem.
    """
    schema = load_schema()
    selected = select_tables(schema, only)
    docs = parse_docs_counts()
    plan = Plan(schema, selected, {}, docs)
    for table in selected:
        raw = read_raw_csv(table)
        plan.raw[table] = raw
        if table not in docs:
            raise LoaderError(f"{table} has no row count in {DOCS_PATH.name}")
        if len(raw) != docs[table]:
            raise LoaderError(
                f"{table}: CSV has {len(raw):,} rows but {DOCS_PATH.name} says {docs[table]:,}"
            )
        if convert:
            plan.rows[table] = convert_table(raw, schema.tables[table])
    return plan


def drop_phase(conn: Any, plan: Plan, full: bool) -> None:
    """Drop the selected tables (children first) in one transaction."""
    cursor = conn.cursor()
    try:
        statements = [f"DROP VIEW IF EXISTS dbo.{VIEW_NAME}"] if full else []
        statements += [f"DROP TABLE IF EXISTS dbo.{t}" for t in reversed(plan.selected)]
        for statement in statements:
            assert_statement_allowed(statement)
            cursor.execute(statement)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def load_table(conn: Any, plan: Plan, table: str) -> TableResult:
    """Create, fill and verify one table inside a single transaction.

    Any exception or verification mismatch rolls the transaction back (SQL Server DDL is
    transactional, so the table does not exist afterwards) and is reported in the result.
    """
    result = TableResult(table, expected=len(plan.raw[table]))
    started = time.perf_counter()
    ddl = plan.schema.tables[table]
    try:
        cursor = conn.cursor()
        assert_statement_allowed(ddl.batch, batch=True)
        cursor.execute(ddl.batch)
        insert = insert_statement(ddl)
        assert_statement_allowed(insert)
        cursor.fast_executemany = True
        rows = plan.rows[table]
        sizes = _input_sizes(ddl.columns)
        for start in range(0, len(rows), BATCH_SIZE):
            cursor.setinputsizes(sizes)
            cursor.executemany(insert, rows[start : start + BATCH_SIZE])
        result.loaded = _scalar(cursor, f"SELECT COUNT_BIG(*) FROM dbo.{table}")
        problem = verify_table(cursor, table, plan.raw[table], plan.docs[table])
        if problem:
            raise LoaderError(problem)
        conn.commit()
        result.status = "OK"
    except Exception as exc:
        conn.rollback()
        result.status = "FAILED"
        result.detail = str(exc)[:300]
        logger.error("%s rolled back: %s", table, result.detail)
    result.seconds = time.perf_counter() - started
    return result


def recreate_view(conn: Any, plan: Plan) -> TableResult:
    """Drop and recreate the scorecard view, then check it keeps every seller."""
    result = TableResult(VIEW_NAME)
    started = time.perf_counter()
    try:
        cursor = conn.cursor()
        for statement in (f"DROP VIEW IF EXISTS dbo.{VIEW_NAME}", plan.schema.view_batch):
            assert_statement_allowed(statement)
            cursor.execute(statement)
        result.loaded = _scalar(cursor, f"SELECT COUNT_BIG(*) FROM dbo.{VIEW_NAME}")
        sellers = _scalar(cursor, "SELECT COUNT_BIG(*) FROM dbo.DimSeller")
        result.expected = sellers
        if result.loaded != sellers:
            raise LoaderError(f"view has {result.loaded:,} rows but DimSeller has {sellers:,}")
        conn.commit()
        result.status = "OK"
    except Exception as exc:
        conn.rollback()
        result.status = "FAILED"
        result.detail = str(exc)[:300]
        logger.error("%s rolled back: %s", VIEW_NAME, result.detail)
    result.seconds = time.perf_counter() - started
    return result


def run_load(conn: Any, plan: Plan, full: bool) -> list[TableResult]:
    """Drop, then create and load each selected table; stop at the first failure."""
    results = [TableResult(t, expected=len(plan.raw[t])) for t in plan.selected]
    drop_phase(conn, plan, full)
    failed = False
    for index, table in enumerate(plan.selected):
        if failed:
            continue
        results[index] = load_table(conn, plan, table)
        failed = results[index].status != "OK"
    if not failed and VIEW_SOURCE_TABLES & set(plan.selected):
        results.append(recreate_view(conn, plan))
    return results


def run_verify(conn: Any, plan: Plan) -> list[TableResult]:
    """Read-only check of the tables already in the database."""
    results: list[TableResult] = []
    cursor = conn.cursor()
    for table in plan.selected:
        result = TableResult(table, expected=len(plan.raw[table]))
        started = time.perf_counter()
        try:
            result.loaded = _scalar(cursor, f"SELECT COUNT_BIG(*) FROM dbo.{table}")
            problem = verify_table(cursor, table, plan.raw[table], plan.docs[table])
            result.status = "FAILED" if problem else "OK"
            result.detail = problem
        except pyodbc.Error as exc:
            result.status = "FAILED"
            result.detail = f"cannot read table: {str(exc)[:200]}"
        result.seconds = time.perf_counter() - started
        results.append(result)
    return results


# --------------------------------------------------------------------------
# Output and CLI
# --------------------------------------------------------------------------


def render_results(results: Sequence[TableResult]) -> str:
    """The final table: table, rows loaded, expected, status, seconds."""
    lines = [
        f"{'table':<24}{'rows loaded':>13}{'expected':>12}  {'status':<8}{'seconds':>9}",
        "-" * 68,
    ]
    for r in results:
        lines.append(
            f"{r.table:<24}{r.loaded:>13,}{r.expected:>12,}  {r.status:<8}{r.seconds:>9.2f}"
        )
        if r.detail:
            lines.append(f"    {r.detail}")
    lines.append("-" * 68)
    lines.append(f"{'total':<24}{'':>13}{'':>12}  {'':<8}{sum(r.seconds for r in results):>9.2f}")
    return "\n".join(lines) + "\n"


def render_dry_run(plan: Plan, full: bool, server: str) -> str:
    """What a real run would do, as text (no connection is opened)."""
    out = [f"DRY RUN: no connection opened. Target {server} / {ALLOWED_DATABASE}"]
    out.append("1. drop (one transaction, children first):")
    if full:
        out.append(f"     DROP VIEW IF EXISTS dbo.{VIEW_NAME}")
    out += [f"     DROP TABLE IF EXISTS dbo.{t}" for t in reversed(plan.selected)]
    step = 2
    for table in plan.selected:
        ddl = plan.schema.tables[table]
        statements = len(split_statements(ddl.batch))
        out.append(
            f"{step}. {table}: run {statements} DDL statements from sql/schema.sql, insert "
            f"{len(plan.rows[table]):,} rows from {CSV_FILES[table]} in batches of {BATCH_SIZE:,}, "
            f"verify count (CSV = docs = {plan.docs[table]:,}) and "
            f"{CHECKSUMS[table].description}, commit"
        )
        step += 1
    if VIEW_SOURCE_TABLES & set(plan.selected):
        out.append(
            f"{step}. recreate dbo.{VIEW_NAME} and check it returns one row per DimSeller row"
        )
    return "\n".join(out) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Command line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src.load_sql",
        description=f"Load the exported CSVs into SQL Server database {ALLOWED_DATABASE}.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="validate and print the plan; no connection"
    )
    mode.add_argument(
        "--verify-only", action="store_true", help="only compare the database with the CSVs"
    )
    parser.add_argument(
        "--only", action="append", metavar="TABLE", help="load only this table (repeatable)"
    )
    parser.add_argument(
        "--server",
        default=os.environ.get(SERVER_ENV_VAR, DEFAULT_SERVER),
        help=f"SQL Server instance (default {DEFAULT_SERVER}, or ${SERVER_ENV_VAR})",
    )
    parser.add_argument("--database", default=ALLOWED_DATABASE, help=f"must be {ALLOWED_DATABASE}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the loader; returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        assert_database_allowed(args.database)
        plan = prepare(args.only, convert=not args.verify_only)
        full = args.only is None
        if args.dry_run:
            sys.stdout.write(render_dry_run(plan, full, args.server))
            return EXIT_OK
        try:
            conn = connect(args.server, args.database)
        except LoaderError as exc:
            logger.error("%s", exc)
            return EXIT_NO_CONNECTION
        try:
            results = run_verify(conn, plan) if args.verify_only else run_load(conn, plan, full)
        finally:
            conn.close()
    except SafetyError as exc:
        logger.error("refused: %s", exc)
        return EXIT_REFUSED
    except pyodbc.Error as exc:
        logger.error("database error: %s", exc)
        return EXIT_FAILED
    except LoaderError as exc:
        logger.error("%s", exc)
        return EXIT_FAILED
    sys.stdout.write(render_results(results))
    return EXIT_OK if all(r.status == "OK" for r in results) else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
