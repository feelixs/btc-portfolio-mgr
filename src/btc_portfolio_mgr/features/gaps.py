"""Gap detection and complete-hourly reindexing for the feature pipeline."""
from __future__ import annotations

import polars as pl


def reindex_to_hourly(prices: pl.DataFrame) -> pl.DataFrame:
    """Reindex to a complete hourly grid from min to max timestamp.

    Missing hours appear as rows with null price/volume so downstream
    rolling operations naturally produce nulls when their window
    spans a gap.
    """
    if prices.height == 0:
        return prices
    start = prices.select(pl.col("timestamp").min()).item()
    end = prices.select(pl.col("timestamp").max()).item()
    grid = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start, end, interval="1h", time_zone="UTC", eager=True
            )
        }
    )
    return grid.join(prices, on="timestamp", how="left")


def find_gaps(prices: pl.DataFrame) -> pl.DataFrame:
    """Return a frame of (gap_start, gap_end, missing_hours) for each gap.

    A gap is a run of missing hourly timestamps between two present rows.
    gap_start is the first missing hour; gap_end is the last missing hour.
    """
    reindexed = reindex_to_hourly(prices)
    if reindexed.height == 0:
        return pl.DataFrame(
            schema={
                "gap_start": pl.Datetime("us", "UTC"),
                "gap_end": pl.Datetime("us", "UTC"),
                "missing_hours": pl.Int64,
            }
        )
    is_missing = reindexed["price"].is_null()
    if not is_missing.any():
        return pl.DataFrame(
            schema={
                "gap_start": pl.Datetime("us", "UTC"),
                "gap_end": pl.Datetime("us", "UTC"),
                "missing_hours": pl.Int64,
            }
        )
    timestamps = reindexed["timestamp"].to_list()
    missing_flags = is_missing.to_list()
    gaps: list[dict] = []
    i = 0
    n = len(missing_flags)
    while i < n:
        if missing_flags[i]:
            j = i
            while j < n and missing_flags[j]:
                j += 1
            gaps.append(
                {
                    "gap_start": timestamps[i],
                    "gap_end": timestamps[j - 1],
                    "missing_hours": j - i,
                }
            )
            i = j
        else:
            i += 1
    return pl.DataFrame(
        gaps,
        schema={
            "gap_start": pl.Datetime("us", "UTC"),
            "gap_end": pl.Datetime("us", "UTC"),
            "missing_hours": pl.Int64,
        },
    )
