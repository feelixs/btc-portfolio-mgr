"""Gap detection and complete-hourly reindexing for the feature pipeline."""
from __future__ import annotations

import polars as pl

MAX_INTERPOLATION_HOURS = 6


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


def interpolate_short_gaps(
    reindexed: pl.DataFrame, max_gap_hours: int = MAX_INTERPOLATION_HOURS
) -> pl.DataFrame:
    """Linear-interpolate `price` across null gaps of at most `max_gap_hours`.

    Longer gaps remain null. Volume is intentionally NOT interpolated (volume
    is a flow quantity; the missing-hour volume is genuinely unknown). The
    input MUST be the output of `reindex_to_hourly` (a complete hourly grid).

    Rationale: CoinGecko occasionally drops a sample for a given hour. The
    price didn't actually pause — we just don't have a snapshot. A short-gap
    linear interpolation is closer to the unobserved truth than treating the
    hour as missing data forever, which propagates 90 days of nulls into
    `zscore_90d` and similar long-lookback features.
    """
    if max_gap_hours < 0:
        raise ValueError(f"max_gap_hours must be non-negative, got {max_gap_hours}")
    if reindexed.height == 0:
        return reindexed
    df = (
        reindexed.with_columns(_is_null=pl.col("price").is_null())
        .with_columns(_run_id=pl.col("_is_null").rle_id())
        .with_columns(_run_length=pl.col("_run_id").len().over("_run_id"))
        .with_columns(_interpolated=pl.col("price").interpolate())
    )
    df = df.with_columns(
        price=pl.when(
            pl.col("_is_null") & (pl.col("_run_length") <= max_gap_hours)
        )
        .then(pl.col("_interpolated"))
        .otherwise(pl.col("price"))
    )
    return df.drop("_is_null", "_run_id", "_run_length", "_interpolated")
