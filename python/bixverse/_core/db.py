"""The obs / var DuckDB of a single cell directory.

A port of the bixverse R6 class `SingleCellDuckDB` as plain functions over a
`duckdb.DuckDBPyConnection`. The file format is shared with R, so the on-disk
contract is R's: tables `obs` (`cell_idx`, `cell_id`, `to_keep`, ...), `var`
(`gene_idx`, `gene_id`, ...) and optionally `var_adt` (`feature_idx`,
`feature_id`, ...), the index columns 1-based, categoricals as ENUM.

The index columns are the only place 1-based indices exist on the Python side.
Every function here takes and returns 0-based indices and converts at the SQL
boundary.

Frames cross into DuckDB through the Arrow PyCapsule interface and come back
the same way, so none of this needs pyarrow. Identifiers are checked against
the table schema before they are templated into SQL; values always go in as
parameters or registered frames.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import duckdb
import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse

###########
# Globals #
###########

DB_NAME: str = "sc_duckdb.db"

Table = Literal["obs", "var", "var_adt"]

# index column, identifier column
TABLE_KEYS: dict[str, tuple[str, str]] = {
    "obs": ("cell_idx", "cell_id"),
    "var": ("gene_idx", "gene_id"),
    "var_adt": ("feature_idx", "feature_id"),
}

PROTECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "obs": ("cell_idx", "cell_id", "to_keep"),
    "var": ("gene_idx", "gene_id"),
}

# Up to this many levels a categorical is cast with an inline ENUM literal;
# above it the levels go in through a registered frame and a named type,
# since parsing the literal dominates for millions of levels. Not tuned; the
# same constant as R's `SC_ENUM_LITERAL_MAX`.
SC_ENUM_LITERAL_MAX: int = 10_000

PARQUET_TYPES: tuple[str, ...] = (
    "integer",
    "double",
    "boolean",
    "string",
    "categorical",
)


class TableNotFoundError(Exception):
    """A table the call needs is missing from the DuckDB."""


###########
# Records #
###########


@dataclass(frozen=True)
class ParquetFrameInfo:
    """Description of a Parquet metadata table, as the Rust writer returns it.

    Attributes:
        names: Column names in file order.
        types: One of `"integer"`, `"double"`, `"boolean"`, `"string"` or
            `"categorical"` per column.
        categories: The levels in order for a categorical column, `None`
            otherwise.
        skipped: Source columns the writer left out.
        n_rows: Number of rows written.
    """

    names: tuple[str, ...]
    types: tuple[str, ...]
    categories: tuple[tuple[str, ...] | None, ...]
    skipped: tuple[str, ...] = ()
    n_rows: int = 0

    @beartype
    def __post_init__(self) -> None:
        """Check the three per-column tuples line up.

        Raises:
            ValueError: On a length mismatch or an unknown type.
        """
        if not len(self.names) == len(self.types) == len(self.categories):
            raise ValueError("names, types and categories must have equal lengths")
        bad = set(self.types) - set(PARQUET_TYPES)
        if bad:
            raise ValueError(f"unknown Parquet column type(s): {sorted(bad)}")

    @beartype
    def levels(self, name: str) -> tuple[str, ...] | None:
        """The ENUM levels of a column, or `None` when it is not categorical."""
        i = self.names.index(name)
        return self.categories[i] if self.types[i] == "categorical" else None


@dataclass(frozen=True)
class ParquetPart:
    """One file of a multi-file load.

    Attributes:
        path: The Parquet file.
        info: Its writer output.
        exp_id: Experiment id, prefixed onto the identifier column.
    """

    path: Path
    info: ParquetFrameInfo
    exp_id: str


##############
# Connection #
##############


@beartype
def db_path(directory: Path) -> Path:
    """Path of the DuckDB inside a single cell directory."""
    return directory / DB_NAME


@contextmanager
@beartype
def connect(
    path: Path, *, read_only: bool = False
) -> Generator[duckdb.DuckDBPyConnection]:
    """Open the DuckDB for one call and close it afterwards, as R does.

    Args:
        path: The `.db` file.
        read_only: Open read-only.

    Yields:
        The open connection.
    """
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


class _ArrowStream:
    """A polars frame seen only through the Arrow PyCapsule stream protocol.

    DuckDB scans a polars frame through pyarrow, but any object exposing just
    `__arrow_c_stream__` is read natively.
    """

    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        return self._frame.__arrow_c_stream__(requested_schema)


@contextmanager
@beartype
def _registered(
    con: duckdb.DuckDBPyConnection, name: str, frame: pl.DataFrame
) -> Generator[None]:
    """Register a polars frame as a view for the duration of a block."""
    con.register(name, _ArrowStream(frame))
    try:
        yield
    finally:
        con.unregister(name)


###########
# Helpers #
###########


@beartype
def _qi(name: str) -> str:
    """Quote an SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


@beartype
def _qs(value: str) -> str:
    """Quote an SQL string literal, for the places DuckDB cannot bind one."""
    return "'" + value.replace("'", "''") + "'"


@beartype
def _enum_literal(levels: list[str] | tuple[str, ...]) -> str:
    """An inline `ENUM(...)` type for the given levels."""
    return "ENUM(" + ", ".join(_qs(v) for v in levels) + ")"


@beartype
def has_table(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    """Whether a table exists in the open database."""
    rows = con.execute(
        "SELECT 1 FROM duckdb_tables() "
        "WHERE database_name = current_database() AND table_name = ?",
        [table],
    ).fetchall()
    return len(rows) > 0


@beartype
def check_table(con: duckdb.DuckDBPyConnection, table: Table) -> None:
    """Raise unless the table exists.

    Raises:
        TableNotFoundError: If it does not.
    """
    if not has_table(con, table):
        raise TableNotFoundError(f"The {table} table was not found in the DB.")


@beartype
def columns(con: duckdb.DuckDBPyConnection, table: Table) -> list[str]:
    """Column names of a table, in table order.

    Raises:
        TableNotFoundError: If the table does not exist.
    """
    check_table(con, table)
    rows = con.execute(
        "SELECT column_name FROM duckdb_columns() "
        "WHERE database_name = current_database() AND table_name = ? "
        "ORDER BY column_index",
        [table],
    ).fetchall()
    return [r[0] for r in rows]


@beartype
def _check_columns(
    con: duckdb.DuckDBPyConnection, table: Table, cols: list[str]
) -> None:
    """Raise unless every column exists in the table.

    Raises:
        KeyError: Naming the missing columns.
    """
    missing = [c for c in cols if c not in set(columns(con, table))]
    if missing:
        raise KeyError(f"column(s) not found in '{table}': {missing}")


@beartype
def n_rows(con: duckdb.DuckDBPyConnection, table: Table) -> int:
    """Number of rows in a table."""
    check_table(con, table)
    # fetchall, not fetchone: DuckDB 2 refuses further work on a connection
    # while a result is only partly fetched
    return int(con.execute(f"SELECT COUNT(*) FROM {_qi(table)}").fetchall()[0][0])


@beartype
def _fetch(
    con: duckdb.DuckDBPyConnection, sql: str, params: list[object] | None = None
) -> pl.DataFrame:
    """Run a query and return it as polars, ENUM columns as `pl.Enum`.

    Polars reads a DuckDB ENUM as a plain categorical, which loses the level
    order and the unused levels; the cast restores both from the column type.
    """
    rel = con.sql(sql, params=params)
    frame = pl.DataFrame(rel)
    casts = [
        pl.col(name).cast(pl.Enum(cast(list[str], dict(dtype.children)["values"])))
        for name, dtype in zip(rel.columns, rel.types, strict=True)
        if dtype.id == "enum"
    ]
    return frame.with_columns(casts) if casts else frame


@beartype
def _to_zero_based(frame: pl.DataFrame, key: str) -> pl.DataFrame:
    """Shift a 1-based index column, if selected, to 0-based `UInt32`."""
    if key not in frame.columns:
        return frame
    return frame.with_columns((pl.col(key) - 1).cast(pl.UInt32))


@beartype
def _key_frame(key: str, indices: np.ndarray) -> pl.DataFrame:
    """A one-column frame of 1-based keys from 0-based indices."""
    return pl.DataFrame({key: (indices.astype(np.int64) + 1).astype(np.int32)})


###########
# Getters #
###########


@beartype
def get_table(
    con: duckdb.DuckDBPyConnection,
    table: Table,
    *,
    indices: np.ndarray | None = None,
    cols: list[str] | None = None,
    filtered: bool = False,
    where_equal: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Read a table, optionally restricted to rows and columns.

    Rows come back in index order.

    Args:
        con: Open connection.
        table: `"obs"`, `"var"` or `"var_adt"`.
        indices: Optional 0-based row indices to keep.
        cols: Optional columns to return. `None` returns all.
        filtered: Keep only `to_keep` cells. Only valid for `obs`.
        where_equal: Keep rows whose column equals the value. The value is
            cast to the column type, not the column to a string, as R does,
            so `"1"` matches a numeric 1 whose string form would be `"1.0"`.

    Returns:
        The table with its index column 0-based.

    Raises:
        TableNotFoundError: If the table does not exist.
        KeyError: If a requested column does not exist.
        ValueError: If `filtered` is asked of a table other than `obs`.
    """
    key, _ = TABLE_KEYS[table]
    check_table(con, table)
    if cols is not None:
        _check_columns(con, table, cols)
    if filtered and table != "obs":
        raise ValueError("filtered only applies to the obs table")

    col_part = "*" if cols is None else ", ".join(_qi(c) for c in cols)
    where = []
    if filtered:
        where.append("t.to_keep = TRUE")
    if indices is not None:
        where.append(f"t.{_qi(key)} IN (SELECT {_qi(key)} FROM bixverse_keys)")
    params: list[object] = []
    if where_equal:
        _check_columns(con, table, list(where_equal))
        types = dict(
            con.execute(
                "SELECT column_name, data_type FROM duckdb_columns() "
                "WHERE database_name = current_database() AND table_name = ?",
                [table],
            ).fetchall()
        )
        for col, value in where_equal.items():
            where.append(f"t.{_qi(col)} = TRY_CAST(? AS {types[col]})")
            params.append(value)
    where_part = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"SELECT {col_part} FROM {_qi(table)} t {where_part} ORDER BY t.{_qi(key)}"

    if indices is None:
        frame = _fetch(con, sql, params)
    else:
        with _registered(con, "bixverse_keys", _key_frame(key, indices)):
            frame = _fetch(con, sql, params)
    return _to_zero_based(frame, key)


@beartype
def get_ids(con: duckdb.DuckDBPyConnection, table: Table) -> np.ndarray:
    """The identifier column, positioned by 0-based index.

    The R index maps (`get_obs_index_map`, `get_var_index_map`) as an array:
    element `i` is the id of row `i`.

    Raises:
        ValueError: If the index column is not exactly `1..n`, which would
            leave the ids misaligned with the binary stores.
    """
    key, id_col = TABLE_KEYS[table]
    check_table(con, table)
    rows = con.execute(
        f"SELECT {_qi(key)}, {_qi(id_col)} FROM {_qi(table)} ORDER BY {_qi(key)}"
    ).fetchall()
    idx = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    if not np.array_equal(idx, np.arange(1, len(rows) + 1)):
        raise ValueError(f"{table}.{key} is not a contiguous 1..n index")
    return np.array([r[1] for r in rows], dtype=object)


@beartype
def get_cells_to_keep(con: duckdb.DuckDBPyConnection) -> np.ndarray:
    """0-based indices of the cells with `to_keep = TRUE`, ascending."""
    check_table(con, "obs")
    rows = con.execute(
        "SELECT cell_idx FROM obs WHERE to_keep ORDER BY cell_idx"
    ).fetchall()
    return np.fromiter((r[0] - 1 for r in rows), dtype=np.uint32, count=len(rows))


#####################
# Setters / filters #
#####################


@beartype
def set_cells_to_keep(con: duckdb.DuckDBPyConnection, cells: np.ndarray) -> None:
    """Flag exactly the given cells as `to_keep`, every other one as not.

    Args:
        con: Open connection.
        cells: 0-based cell indices to keep.
    """
    check_table(con, "obs")
    with _registered(con, "bixverse_keys", _key_frame("cell_idx", cells)):
        con.execute(
            "UPDATE obs SET to_keep = cell_idx IN (SELECT cell_idx FROM bixverse_keys)"
        )


@beartype
def reset_cells_to_keep(con: duckdb.DuckDBPyConnection) -> None:
    """Flag every cell as `to_keep`."""
    check_table(con, "obs")
    con.execute("UPDATE obs SET to_keep = TRUE")


@beartype
def set_to_keep_column(con: duckdb.DuckDBPyConnection) -> None:
    """Add `to_keep` if missing and set it to `TRUE` wherever it is null."""
    check_table(con, "obs")
    con.execute("ALTER TABLE obs ADD COLUMN IF NOT EXISTS to_keep BOOLEAN")
    con.execute("UPDATE obs SET to_keep = TRUE WHERE to_keep IS NULL")


@beartype
def _column_types(
    con: duckdb.DuckDBPyConnection, view: str, frame: pl.DataFrame
) -> dict[str, str]:
    """SQL type of every column of a registered frame.

    `pl.Enum` and `pl.Categorical` columns become an ENUM, as a factor does
    when R writes it, with the Enum's levels in order and a Categorical's
    distinct values sorted (what `factor()` would make of them). DuckDB itself
    would read either as VARCHAR.
    """
    types = {r[0]: r[1] for r in con.execute(f"DESCRIBE {_qi(view)}").fetchall()}
    for name, dtype in frame.schema.items():
        if isinstance(dtype, pl.Enum):
            levels = dtype.categories.to_list()
        elif isinstance(dtype, pl.Categorical):
            levels = sorted(frame[name].drop_nulls().unique().cast(pl.String))
        else:
            continue
        if levels:
            types[name] = _enum_literal(levels)
    return types


@beartype
def _write_columns(
    con: duckdb.DuckDBPyConnection, table: Table, key: str, frame: pl.DataFrame
) -> None:
    """Write every non-key column of `frame` into `table`, matched on `key`.

    R's `write_columns`: each column is dropped if present, added, and filled
    with one UPDATE. Rows without a match in `frame` end up NULL. `frame[key]`
    must already be 1-based.
    """
    with _registered(con, "bixverse_new_data", frame):
        types = _column_types(con, "bixverse_new_data", frame)
        new_cols = [c for c in frame.columns if c != key]
        for col in new_cols:
            con.execute(f"ALTER TABLE {_qi(table)} DROP COLUMN IF EXISTS {_qi(col)}")
            con.execute(f"ALTER TABLE {_qi(table)} ADD COLUMN {_qi(col)} {types[col]}")
        if new_cols:
            sets = ", ".join(f"{_qi(c)} = n.{_qi(c)}" for c in new_cols)
            con.execute(
                f"UPDATE {_qi(table)} SET {sets} FROM bixverse_new_data n "
                f"WHERE {_qi(table)}.{_qi(key)} = n.{_qi(key)}"
            )


@beartype
def add_data(
    con: duckdb.DuckDBPyConnection, table: Literal["obs", "var"], frame: pl.DataFrame
) -> None:
    """Add or replace columns in `obs` or `var`.

    With a key column (`cell_idx` / `gene_idx`, 0-based) the frame is joined
    on it, as R's `join_data_obs`. Without one it must hold one row per table
    row in index order, as R's `add_data_obs` / `add_data_var`.

    Args:
        con: Open connection.
        table: `"obs"` or `"var"`.
        frame: The new columns.

    Raises:
        ValueError: If a keyless frame has the wrong number of rows, or a key
            is out of range or repeated.
    """
    key, _ = TABLE_KEYS[table]
    n = n_rows(con, table)
    if key in frame.columns:
        idx = frame[key].to_numpy()
        if idx.dtype.kind not in "iu":
            raise ValueError(f"{key} must hold integers, got {frame[key].dtype}")
        if len(idx) and (idx.min() < 0 or idx.max() >= n):
            raise ValueError(f"{key} out of range for a {table} table of {n} rows")
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"{key} holds repeated values")
        frame = frame.with_columns((pl.col(key).cast(pl.Int64) + 1).cast(pl.Int32))
    else:
        if frame.height != n:
            raise ValueError(
                f"frame has {frame.height} rows; the {table} table has {n}"
            )
        frame = frame.with_columns(pl.int_range(1, n + 1, dtype=pl.Int32).alias(key))
    _write_columns(con, table, key, frame)


##########
# Rename #
##########


@beartype
def rename_column(
    con: duckdb.DuckDBPyConnection, table: Literal["obs", "var"], old: str, new: str
) -> None:
    """Rename a column. A missing column warns and does nothing, as in R."""
    if old not in columns(con, table):
        warnings.warn(
            f"Column '{old}' not found in table '{table}'. No renaming done.",
            stacklevel=2,
        )
        return
    con.execute(f"ALTER TABLE {_qi(table)} RENAME COLUMN {_qi(old)} TO {_qi(new)}")


@beartype
def drop_columns(
    con: duckdb.DuckDBPyConnection, table: Literal["obs", "var"], cols: list[str]
) -> None:
    """Drop columns, refusing the identifier and bookkeeping ones.

    Protected (`cell_idx`, `cell_id`, `to_keep`; `gene_idx`, `gene_id`) and
    missing columns warn and are skipped, as in R.
    """
    existing = columns(con, table)
    protected = PROTECTED_COLUMNS[table]

    refused = [c for c in cols if c in protected]
    if refused:
        warnings.warn(
            f"Refusing to drop protected column(s) in '{table}': {', '.join(refused)}",
            stacklevel=2,
        )
    missing = [c for c in cols if c not in existing and c not in protected]
    if missing:
        warnings.warn(
            f"Column(s) not found in '{table}' (skipping): {', '.join(missing)}",
            stacklevel=2,
        )
    for col in cols:
        if col in existing and col not in protected:
            con.execute(f"ALTER TABLE {_qi(table)} DROP COLUMN {_qi(col)}")


############
# Populate #
############


@beartype
def populate_var_minimal(
    con: duckdb.DuckDBPyConnection, gene_ids: list[str] | np.ndarray
) -> None:
    """Write a `var` table of `gene_idx` and `gene_id` only.

    Args:
        con: Open connection.
        gene_ids: Gene ids in store order.

    Raises:
        ValueError: If `gene_ids` is empty.
    """
    if len(gene_ids) == 0:
        raise ValueError("gene_ids must be non-empty")
    frame = pl.DataFrame({"gene_id": pl.Series(list(gene_ids), dtype=pl.String)})
    populate_from_frame(con, "var", frame)


@beartype
def populate_from_frame(
    con: duckdb.DuckDBPyConnection, table: Table, frame: pl.DataFrame
) -> None:
    """Replace a table with the rows of a polars frame.

    The index column is (re)numbered `1..n` in row order and put first,
    followed by the identifier column and then the rest in frame order. Enum
    and Categorical columns become ENUMs. Column names are taken as they are;
    R's `populate_obs_from_data.table` additionally snake_cases them and
    renames the first column to the identifier.

    Args:
        con: Open connection.
        table: The table to write.
        frame: The rows. Must hold the identifier column (`cell_id`,
            `gene_id` or `feature_id`).

    Raises:
        KeyError: If the identifier column is missing.
    """
    key, id_col = TABLE_KEYS[table]
    if id_col not in frame.columns:
        raise KeyError(f"frame must hold the '{id_col}' column")
    rest = [c for c in frame.columns if c not in (key, id_col)]
    frame = frame.select(
        pl.int_range(1, frame.height + 1, dtype=pl.Int32).alias(key), id_col, *rest
    )

    drop_table_and_enums(con, table)
    with _registered(con, "bixverse_new_data", frame):
        types = _column_types(con, "bixverse_new_data", frame)
        select = ", ".join(
            f"CAST({_qi(c)} AS {types[c]}) AS {_qi(c)}" for c in frame.columns
        )
        con.execute(
            f"CREATE OR REPLACE TABLE {_qi(table)} AS "
            f"SELECT {select} FROM bixverse_new_data"
        )


###########
# Parquet #
###########


@beartype
def drop_table_and_enums(con: duckdb.DuckDBPyConnection, table: Table) -> None:
    """Drop a table and the named ENUM types its loads created.

    The named types of `parquet_select_col` are called
    `<table>_<column>_enum`. Only names built from the table's current
    columns are dropped, so `var` never touches the types of `var_adt`. The
    table goes first, since a type cannot be dropped while a column uses it.
    """
    cols = columns(con, table) if has_table(con, table) else []
    con.execute(f"DROP TABLE IF EXISTS {_qi(table)}")
    for col in cols:
        con.execute(f"DROP TYPE IF EXISTS {_qi(f'{table}_{col}_enum')}")


@beartype
def parquet_select_col(
    con: duckdb.DuckDBPyConnection,
    table: Table,
    name: str,
    levels: list[str] | tuple[str, ...] | None,
    *,
    enum_literal_max: int = SC_ENUM_LITERAL_MAX,
) -> str:
    """SQL select expression for one Parquet column.

    A categorical with at least one level is cast to an ENUM with its levels
    in order; with none it stays VARCHAR, since `ENUM()` is invalid. Up to
    `enum_literal_max` levels the ENUM is inline, above it a named type
    `<table>_<column>_enum` is created from a registered frame.

    Args:
        con: Open connection.
        table: The table being built, used to name the type.
        name: Column name.
        levels: The ENUM levels, or `None` for a non-categorical column.
        enum_literal_max: Most levels written inline.

    Returns:
        The select expression. Creates the named type where one is needed.
    """
    col = _qi(name)
    if not levels:
        return col
    if len(levels) <= enum_literal_max:
        return f"CAST({col} AS {_enum_literal(levels)}) AS {col}"

    type_name = _qi(f"{table}_{name}_enum")
    lv = pl.DataFrame({"pos": range(len(levels)), "lvl": list(levels)})
    with _registered(con, "bixverse_enum_levels", lv):
        con.execute(f"DROP TYPE IF EXISTS {type_name}")
        con.execute(
            f"CREATE TYPE {type_name} AS ENUM "
            "(SELECT lvl FROM bixverse_enum_levels ORDER BY pos)"
        )
    return f"CAST({col} AS {type_name}) AS {col}"


@beartype
def _parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    """Column names of a Parquet file."""
    desc = con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(path)]).description
    return [d[0] for d in desc]


@beartype
def _check_parquet_info(
    con: duckdb.DuckDBPyConnection, path: Path, info: ParquetFrameInfo
) -> None:
    """Raise unless every column `info` names is in the file.

    Raises:
        KeyError: Naming the absent columns.
    """
    missing = sorted(set(info.names) - set(_parquet_columns(con, path)))
    if missing:
        raise KeyError(f"{path} lacks the column(s) its info names: {missing}")


@beartype
def parquet_to_duckdb(
    con: duckdb.DuckDBPyConnection,
    table: Table,
    parquet_path: Path,
    info: ParquetFrameInfo,
    *,
    enum_literal_max: int = SC_ENUM_LITERAL_MAX,
) -> None:
    """Load a Parquet metadata table into DuckDB, casting categoricals to ENUM.

    Args:
        con: Open connection.
        table: Target table.
        parquet_path: The Parquet file.
        info: Its writer output.
        enum_literal_max: Most levels cast with an inline ENUM.

    Raises:
        KeyError: If `info` names a column the file does not hold.
    """
    _check_parquet_info(con, parquet_path, info)
    drop_table_and_enums(con, table)
    cols = [
        parquet_select_col(
            con, table, name, info.levels(name), enum_literal_max=enum_literal_max
        )
        for name in info.names
    ]
    con.execute(
        f"CREATE OR REPLACE TABLE {_qi(table)} AS "
        f"SELECT {', '.join(cols)} FROM read_parquet(?)",
        [str(parquet_path)],
    )


@beartype
def _parquet_parts_levels(
    con: duckdb.DuckDBPyConnection, name: str, parts: list[ParquetPart], idx_col: str
) -> list[str] | None:
    """ENUM levels of a column across several Parquet parts.

    Reproduces what `data.table::rbindlist()` makes of the column when at
    least one part holds it as a factor: per part in order, a categorical adds
    its levels and a string column its values in row order, each only when not
    yet seen.

    Returns:
        The levels, or `None` if no part holds the column as categorical, or
        some part holds it as something other than a string.
    """
    types = [
        p.info.types[p.info.names.index(name)] if name in p.info.names else None
        for p in parts
    ]
    present = [t for t in types if t is not None]
    if "categorical" not in present or not set(present) <= {"categorical", "string"}:
        return None

    levels: list[str] = []
    seen: set[str] = set()
    for part, kind in zip(parts, types, strict=True):
        match kind:
            case "categorical":
                new = list(part.info.levels(name) or ())
            case "string":
                rows = con.execute(
                    f"SELECT {_qi(name)} AS v FROM read_parquet(?) "
                    f"WHERE {_qi(name)} IS NOT NULL "
                    f"GROUP BY {_qi(name)} ORDER BY min({_qi(idx_col)})",
                    [str(part.path)],
                ).fetchall()
                new = [r[0] for r in rows]
            case None:
                new = []
            case _:
                raise AssertionError(f"unexpected column type {kind}")
        for v in new:
            if v not in seen:
                seen.add(v)
                levels.append(v)
    return levels


@beartype
def parquet_parts_to_duckdb(
    con: duckdb.DuckDBPyConnection,
    table: Table,
    parts: list[ParquetPart],
    *,
    enum_literal_max: int = SC_ENUM_LITERAL_MAX,
) -> None:
    """Load several Parquet metadata tables into one DuckDB table.

    Stacks the parts in order with `UNION ALL BY NAME` (columns a part lacks
    become NULL), adds `exp_id`, prefixes the identifier with it and
    renumbers the index. Column order is the index, the identifier, `exp_id`,
    then the rest in order of first appearance.

    Args:
        con: Open connection.
        table: Target table.
        parts: One entry per file.
        enum_literal_max: Most levels cast with an inline ENUM.

    Raises:
        ValueError: If `parts` is empty.
        KeyError: If a part's info names a column its file does not hold.
    """
    if not parts:
        raise ValueError("parts must be non-empty")
    idx_col, id_col = TABLE_KEYS[table]
    for p in parts:
        _check_parquet_info(con, p.path, p.info)

    drop_table_and_enums(con, table)

    other_cols: list[str] = []
    for p in parts:
        for name in p.info.names:
            if name not in (idx_col, id_col, "exp_id") and name not in other_cols:
                other_cols.append(name)

    stacked = " UNION ALL BY NAME ".join(
        f"SELECT *, {k} AS _part, ? AS exp_id FROM read_parquet(?)"
        for k in range(1, len(parts) + 1)
    )
    params: list[object] = []
    for p in parts:
        params.extend([p.exp_id, str(p.path)])

    cols = [
        f"CAST(row_number() OVER (ORDER BY _part, {_qi(idx_col)}) AS INTEGER) "
        f"AS {_qi(idx_col)}",
        f"exp_id || '_' || coalesce({_qi(id_col)}, 'NA') AS {_qi(id_col)}",
        "exp_id",
        *(
            parquet_select_col(
                con,
                table,
                col,
                _parquet_parts_levels(con, col, parts, idx_col),
                enum_literal_max=enum_literal_max,
            )
            for col in other_cols
        ),
    ]
    con.execute(
        f"CREATE OR REPLACE TABLE {_qi(table)} AS SELECT {', '.join(cols)} "
        f"FROM ({stacked}) ORDER BY _part, {_qi(idx_col)}",
        params,
    )


##############
# Plain text #
##############


@beartype
def plain_text_delimiter(path: Path) -> str:
    """Tab for a `.tsv` / `.tsv.gz`, comma otherwise, as R decides it."""
    return "\t" if re.search(r"\.tsv(\.gz)?$", path.name, re.IGNORECASE) else ","


@beartype
def plain_text_to_duckdb(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    *,
    has_hdr: bool,
    keep: np.ndarray | None,
    table: Literal["obs", "var"],
) -> None:
    """Load a CSV / TSV metadata file into a table, as R's `.plain_text_to_duckdb`.

    The file goes into a temporary table with insertion order preserved, so
    its `rowid` is the 0-based file row. The first column becomes the
    identifier, the others are snake_cased when there is a header, and the
    kept rows are numbered in file order.

    Args:
        con: Open connection.
        path: The file.
        has_hdr: Whether the file has a header.
        keep: Optional 0-based file rows to keep.
        table: `"obs"` or `"var"`.
    """
    idx_col, id_col = TABLE_KEYS[table]
    raw = "bixverse_plain_text_raw"
    con.execute("SET preserve_insertion_order = true")
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE {raw} AS SELECT * FROM read_csv(?, "
        f"delim = {_qs(plain_text_delimiter(path))}, header = {str(has_hdr).lower()})",
        [str(path)],
    )
    try:
        desc = con.execute(f"SELECT * FROM {raw} LIMIT 0").description
        original = [d[0] for d in desc]
        sanitised = _bixverse.snake_case(original) if has_hdr else list(original)
        sanitised[0] = id_col
        mapping = ", ".join(
            f"t.{_qi(o)} AS {_qi(s)}" for o, s in zip(original, sanitised, strict=True)
        )
        sql = (
            f"CREATE OR REPLACE TABLE {_qi(table)} AS SELECT ROW_NUMBER() OVER "
            f"(ORDER BY t.rowid) AS {_qi(idx_col)}, {mapping} FROM {raw} t "
            "{where} ORDER BY t.rowid"
        )
        if keep is None:
            con.execute(sql.format(where=""))
        else:
            rows = pl.DataFrame({"row_num": keep.astype(np.int64)})
            with _registered(con, "bixverse_plain_text_keep", rows):
                con.execute(
                    sql.format(
                        where="WHERE t.rowid IN "
                        "(SELECT row_num FROM bixverse_plain_text_keep)"
                    )
                )
    finally:
        con.execute(f"DROP TABLE IF EXISTS {raw}")


#################
# Multi sources #
#################


@beartype
def _int_like_r(frame: pl.DataFrame) -> pl.DataFrame:
    """Narrow `Int64` columns to `Int32` where they fit, as fread reads them."""
    casts = [
        pl.col(name).cast(pl.Int32)
        for name, dtype in frame.schema.items()
        if dtype == pl.Int64
        and bool(frame[name].drop_nulls().is_between(-(2**31) + 1, 2**31 - 1).all())
    ]
    return frame.with_columns(casts) if casts else frame


@beartype
def rbind_fill(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Stack frames by column name, as `data.table::rbindlist(fill = TRUE)`.

    Columns appear in order of first appearance and are null where a frame
    lacks them. A column that is categorical in any frame comes out as an
    `Enum` whose levels are, frame by frame, its levels or (for a string
    column) its values in row order, each only when not yet seen.

    Args:
        frames: The frames, in stacking order.

    Returns:
        The stacked frame.
    """
    names: list[str] = []
    for f in frames:
        names.extend(c for c in f.columns if c not in names)

    levels: dict[str, list[str]] = {}
    for name in names:
        dtypes = [f.schema[name] for f in frames if name in f.columns]
        if not any(isinstance(d, (pl.Enum, pl.Categorical)) for d in dtypes):
            continue
        seen: dict[str, None] = {}
        for f in frames:
            if name not in f.columns:
                continue
            dtype = f.schema[name]
            values = (
                dtype.categories.to_list()
                if isinstance(dtype, pl.Enum)
                else f[name].drop_nulls().cast(pl.String).unique(maintain_order=True)
            )
            seen.update(dict.fromkeys(values))
        levels[name] = list(seen)

    cast = [
        f.with_columns(pl.col(c).cast(pl.String) for c in levels if c in f.columns)
        for f in frames
    ]
    out = pl.concat(cast, how="diagonal_relaxed").select(names)
    return out.with_columns(pl.col(c).cast(pl.Enum(lv)) for c, lv in levels.items())


@beartype
def _populate_stacked(con: duckdb.DuckDBPyConnection, frame: pl.DataFrame) -> None:
    """Write a stacked obs frame as `cell_idx`, `cell_id`, `exp_id`, the rest."""
    rest = [c for c in frame.columns if c not in ("cell_idx", "cell_id", "exp_id")]
    populate_from_frame(con, "obs", frame.select("cell_id", "exp_id", *rest))


@beartype
def obs_from_multi_plain_text(
    con: duckdb.DuckDBPyConnection,
    parts: list[tuple[Path, str, bool, np.ndarray]],
) -> None:
    """Build `obs` from several barcode files.

    R's `populate_obs_from_multi_plain_text`: per file the kept rows, the
    first column as `cell_id` prefixed with the `exp_id`, stacked with the
    union of columns.

    Args:
        con: Open connection.
        parts: Per file the path, `exp_id`, whether it has a header and the
            0-based rows to keep.

    Raises:
        ValueError: If a barcodes file already has an `exp_id` column.
    """
    frames = []
    for path, exp_id, has_hdr, keep in parts:
        frame = pl.read_csv(
            path, separator=plain_text_delimiter(path), has_header=has_hdr
        )
        names = (
            _bixverse.snake_case(frame.columns)
            if has_hdr
            else [f"V{i + 1}" for i in range(frame.width)]
        )
        names[0] = "cell_id"
        frame = _int_like_r(frame.rename(dict(zip(frame.columns, names, strict=True))))
        if "exp_id" in frame.columns:
            raise ValueError(
                f"Input '{exp_id}' barcodes file already has an exp_id column."
            )
        frame = frame[keep.astype(np.int64)].with_columns(
            (pl.lit(f"{exp_id}_") + pl.col("cell_id").cast(pl.String)).alias("cell_id"),
            pl.lit(exp_id).alias("exp_id"),
        )
        frames.append(frame)
    _populate_stacked(con, rbind_fill(frames))


@beartype
def obs_from_multi_duckdb(
    con: duckdb.DuckDBPyConnection, parts: list[tuple[Path, str]]
) -> None:
    """Build `obs` from the kept cells of several DuckDBs.

    R's `populate_obs_from_multi_duckdb`: per source the `to_keep` rows in
    `cell_idx` order, `cell_id` prefixed with the `exp_id`, stacked with the
    union of columns.

    Args:
        con: Open connection to the target.
        parts: Per source its DuckDB and `exp_id`.

    Raises:
        ValueError: If a source already has an `exp_id` column.
    """
    frames = []
    for path, exp_id in parts:
        with connect(path, read_only=True) as src:
            frame = _fetch(
                src, "SELECT * FROM obs WHERE to_keep = TRUE ORDER BY cell_idx"
            )
        if "exp_id" in frame.columns:
            raise ValueError(
                f"Input '{exp_id}' already has an exp_id column in obs. Merging "
                "already-merged objects is not supported."
            )
        frame = frame.drop([c for c in ("cell_idx", "to_keep") if c in frame.columns])
        frames.append(
            frame.with_columns(
                (
                    pl.lit(f"{exp_id}_")
                    + pl.col("cell_id").cast(pl.String).fill_null("NA")
                ).alias("cell_id"),
                pl.lit(exp_id).alias("exp_id"),
            )
        )
    _populate_stacked(con, rbind_fill(frames))


@beartype
def obs_from_duckdb_subset(
    con: duckdb.DuckDBPyConnection, source: Path, cell_idx: np.ndarray
) -> None:
    """Build `obs` from a subset of a source DuckDB's rows, in a given order.

    R's `populate_obs_from_duckdb_subset`: every source column bar
    `cell_idx` and `to_keep`, rows picked by position so the caller's order
    survives, renumbered from one.

    Args:
        con: Open connection to the target.
        source: The source DuckDB.
        cell_idx: 0-based source cell indices, in target order.

    Raises:
        ValueError: If an index is beyond the source obs.
    """
    with connect(source, read_only=True) as src:
        frame = _fetch(src, "SELECT * FROM obs ORDER BY cell_idx")
    if len(cell_idx) and int(cell_idx.max()) >= frame.height:
        raise ValueError(
            f"cell_idx goes up to {int(cell_idx.max())} but the source obs has "
            f"{frame.height} rows."
        )
    frame = frame.drop([c for c in ("cell_idx", "to_keep") if c in frame.columns])
    populate_from_frame(con, "obs", frame[cell_idx.astype(np.int64)])


@beartype
def var_from_duckdb_reordered(
    con: duckdb.DuckDBPyConnection, source: Path, gene_ids: list[str]
) -> None:
    """Build `var` from a source DuckDB, reordered to `gene_ids`.

    R's `populate_vars_from_duckdb_reordered`. A gene the source does not
    hold keeps only its id, with a warning.

    Args:
        con: Open connection to the target.
        source: The source DuckDB.
        gene_ids: Gene ids in the target order.
    """
    with connect(source, read_only=True) as src:
        var = _fetch(src, "SELECT * FROM var ORDER BY gene_idx")
    var = var.drop("gene_idx").unique(
        subset="gene_id", keep="first", maintain_order=True
    )
    frame = pl.DataFrame({"gene_id": gene_ids}).join(
        var, on="gene_id", how="left", maintain_order="left"
    )
    n_missing = len(set(gene_ids) - set(var["gene_id"].to_list()))
    if n_missing:
        warnings.warn(
            f"{n_missing} gene(s) absent from the var reference source; metadata "
            "limited to gene_id (gene_universe = 'union').",
            stacklevel=2,
        )
    populate_from_frame(con, "var", frame)
