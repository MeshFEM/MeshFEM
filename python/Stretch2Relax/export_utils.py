from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Sequence
import polars as pl


DEFAULT_COLUMNS = [
    "frame",
    "newton_step_length",
    "step_length",
    "min_element_eigval",
    "method_name",
    "optimal_alpha",
    "max_energy_reduction",
    "arc_length",
]


def validate_data_list(
    data_list: Sequence[Sequence[Any]],
    expected_num_columns: int = 8,
) -> None:
    """
    Validate that each row in data_list has the expected number of columns.
    """
    for i, row in enumerate(data_list):
        if len(row) != expected_num_columns:
            raise ValueError(
                f"Row {i} has length {len(row)}, expected {expected_num_columns}. "
                f"Row content: {row}"
            )


def data_list_to_polars_df(
    data_list: Sequence[Sequence[Any]],
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """
    Convert a list of rows to a Polars DataFrame.
    """
    if columns is None:
        columns = DEFAULT_COLUMNS

    validate_data_list(data_list, expected_num_columns=len(columns))
    return pl.DataFrame(data_list, schema=list(columns), orient="row")


def make_timestamp_string() -> str:
    """
    Return a filesystem-friendly timestamp string.
    Example: 20260310_153012
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_output_path(
    output_dir: str | Path,
    base_name: str,
    extension: str,
    add_timestamp: bool = False,
) -> Path:
    """
    Build an output path like:
        output_dir / f"{base_name}.csv"
    or
        output_dir / f"{base_name}_20260310_153012.csv"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not extension.startswith("."):
        extension = "." + extension

    if add_timestamp:
        filename = f"{base_name}_{make_timestamp_string()}{extension}"
    else:
        filename = f"{base_name}{extension}"

    return output_dir / filename


def save_polars_df_to_csv(
    df: pl.DataFrame,
    csv_path: str | Path,
) -> Path:
    """
    Save a Polars DataFrame to CSV.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(csv_path)
    return csv_path.resolve()


def save_polars_df_to_parquet(
    df: pl.DataFrame,
    parquet_path: str | Path,
) -> Path:
    """
    Save a Polars DataFrame to Parquet.
    """
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    return parquet_path.resolve()

def read_csv_to_polars_df(
    csv_path: str | Path,
) -> pl.DataFrame:
    """
    Read a CSV file into a Polars DataFrame.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pl.read_csv(csv_path)


def read_parquet_to_polars_df(
    parquet_path: str | Path,
) -> pl.DataFrame:
    """
    Read a Parquet file into a Polars DataFrame.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    return pl.read_parquet(parquet_path)


def read_table_to_polars_df(
    file_path: str | Path,
) -> pl.DataFrame:
    """
    Read a table file into a Polars DataFrame based on file extension.
    Supports .csv and .parquet.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(file_path)
    if suffix == ".parquet":
        return pl.read_parquet(file_path)

    raise ValueError(
        f"Unsupported file extension: {suffix}. "
        f"Supported extensions are '.csv' and '.parquet'."
    )

def find_latest_export_path(
    output_dir: str | Path,
    base_name: str,
    extension: str | None = None,
) -> Path:
    """
    Find the most recent exported file in output_dir matching base_name.

    Supported patterns:
        {base_name}.csv
        {base_name}.parquet
        {base_name}_YYYYMMDD_HHMMSS.csv
        {base_name}_YYYYMMDD_HHMMSS.parquet

    If extension is provided, it should be 'csv', '.csv', 'parquet', or '.parquet'.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Directory not found: {output_dir}")

    if extension is not None:
        if not extension.startswith("."):
            extension = "." + extension
        allowed_suffixes = {extension.lower()}
    else:
        allowed_suffixes = {".csv", ".parquet"}

    candidates = []
    prefix = f"{base_name}"

    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        if not path.stem.startswith(prefix):
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No export files found in {output_dir} for base_name='{base_name}' "
            f"with extension={extension}."
        )

    # Sort by modification time, newest first.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def load_latest_export(
    output_dir: str | Path,
    base_name: str,
    extension: str | None = None,
) -> tuple[pl.DataFrame, Path]:
    """
    Find the latest matching export file and load it as a Polars DataFrame.

    Returns
    -------
    (df, path)
    """
    latest_path = find_latest_export_path(
        output_dir=output_dir,
        base_name=base_name,
        extension=extension,
    )
    df = read_table_to_polars_df(latest_path)
    return df, latest_path

def export_data_list(
    data_list: Sequence[Sequence[Any]],
    output_dir: str | Path,
    base_name: str = "experiment_data",
    columns: Sequence[str] | None = None,
    save_csv: bool = True,
    save_parquet: bool = False,
    add_timestamp: bool = False,
) -> dict[str, Any]:
    """
    Convert data_list to a Polars DataFrame and export it.

    Returns a dictionary containing:
        {
            "df": pl.DataFrame,
            "csv_path": Path | None,
            "parquet_path": Path | None,
        }
    """
    df = data_list_to_polars_df(data_list, columns=columns)

    csv_path = None
    parquet_path = None

    if save_csv:
        csv_path = make_output_path(
            output_dir=output_dir,
            base_name=base_name,
            extension=".csv",
            add_timestamp=add_timestamp,
        )
        df.write_csv(csv_path)

    if save_parquet:
        parquet_path = make_output_path(
            output_dir=output_dir,
            base_name=base_name,
            extension=".parquet",
            add_timestamp=add_timestamp,
        )
        df.write_parquet(parquet_path)

    return {
        "df": df,
        "csv_path": None if csv_path is None else csv_path.resolve(),
        "parquet_path": None if parquet_path is None else parquet_path.resolve(),
    }