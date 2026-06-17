import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Sequence

def plot_methods_from_polars_simple(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    method_col: str = "method_name",
    figsize: tuple[float, float] = (8, 6),
):
    plt.figure(figsize=figsize)

    method_names = df.select(method_col).unique().to_series().sort().to_list()
    for method_name in method_names:
        arr = np.array(
            df.filter(pl.col(method_col) == method_name)
              .select(x_col, y_col)
              .sort(x_col)
        )
        plt.plot(*arr.T, label=str(method_name))

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.grid(True)

def scatter_compare_methods_with_reference_x(
    df: pl.DataFrame,
    method_name_for_x: str,
    x_col: str,
    y_col: str,
    frame_col: str = "frame",
    method_col: str = "method_name",
    figsize: Tuple[float, float] = (8, 6),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    alpha: float = 0.8,
    s: float = 30,
    legend: bool = True,
    grid: bool = True,
    ax=None,
):
    """
    Draw a scatter plot where the x-data comes from one reference method,
    while the y-data comes from every method, matched by frame.
    """
    required_cols = {frame_col, method_col, x_col, y_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x_df = (
        df.filter(pl.col(method_col) == method_name_for_x)
        .select([
            pl.col(frame_col),
            pl.col(x_col).alias("__x_data"),
        ])
    )

    if x_df.height == 0:
        raise ValueError(f"No rows found for method_name_for_x='{method_name_for_x}'")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    method_names = df.select(method_col).unique().to_series().sort().to_list()

    for method_name in method_names:
        y_df = (
            df.filter(pl.col(method_col) == method_name)
            .select([
                pl.col(frame_col),
                pl.col(y_col).alias("__y_data"),
            ])
        )

        joined = x_df.join(y_df, on=frame_col, how="inner").sort(frame_col)

        if joined.height == 0:
            continue

        ax.scatter(
            joined["__x_data"].to_numpy(),
            joined["__y_data"].to_numpy(),
            label=str(method_name),
            alpha=alpha,
            s=s,
        )

    ax.set_xlabel(xlabel if xlabel is not None else f"{x_col} ({method_name_for_x})")
    ax.set_ylabel(ylabel if ylabel is not None else y_col)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"{y_col} vs {x_col} ({method_name_for_x} as x reference)")

    if grid:
        ax.grid(True)

    if legend:
        ax.legend()

    return fig, ax