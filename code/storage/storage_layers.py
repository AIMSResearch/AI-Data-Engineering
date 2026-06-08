"""
Storage planning helpers for AI data engineering examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def load_storage_inventory(path: str | Path) -> pd.DataFrame:
    """Load the sample storage inventory and normalize boolean fields."""
    df = pd.read_csv(path)
    bool_columns = ["contains_pii", "has_embeddings"]
    for column in bool_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .replace({"True": True, "False": False})
                .astype(bool)
            )
    return df


def summarize_storage_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the inventory by storage layer."""
    summary = (
        df.groupby("storage_layer", dropna=False)
        .agg(
            asset_count=("asset_id", "count"),
            total_size_gb=("size_gb", "sum"),
            avg_daily_reads=("daily_reads", "mean"),
            pii_assets=("contains_pii", "sum"),
            embedding_assets=("has_embeddings", "sum"),
        )
        .reset_index()
        .sort_values(["total_size_gb", "asset_count"], ascending=[False, False])
    )
    return summary


def add_access_pattern_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple hot/warm/cold flags used in the Chapter 3 notebook."""
    enriched = df.copy()
    enriched["temperature"] = "warm"
    # Low latency or heavy read volume usually means the asset is functionally
    # hot even if it is not the largest thing in the platform.
    enriched.loc[
        (enriched["latency_class"] == "low") | (enriched["daily_reads"] >= 5000),
        "temperature",
    ] = "hot"
    # Long retention plus low access frequency is the classic archive shape.
    enriched.loc[
        (enriched["daily_reads"] < 250) & (enriched["retention_days"] >= 365),
        "temperature",
    ] = "cold"
    # Immutable and snapshot assets are costly to lose because they anchor
    # replay, audits, or embedding rebuilds later.
    enriched["is_reproducibility_critical"] = enriched["mutability"].isin(
        ["immutable", "snapshot"]
    ) | enriched["has_embeddings"]
    return enriched


def build_retention_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Suggest a storage tier and retention posture for each asset."""
    plan = add_access_pattern_flags(df)

    def choose_tier(row: pd.Series) -> str:
        # Preserve version history when cold data still anchors reproducibility.
        if row["temperature"] == "hot":
            return "hot"
        if row["temperature"] == "cold" and row["is_reproducibility_critical"]:
            return "archive-with-versioning"
        if row["temperature"] == "cold":
            return "archive"
        return "warm"

    plan["recommended_tier"] = plan.apply(choose_tier, axis=1)
    plan["retention_note"] = plan.apply(_retention_note, axis=1)
    columns = [
        "asset_name",
        "storage_layer",
        "temperature",
        "recommended_tier",
        "retention_days",
        "retention_note",
    ]
    return plan[columns].sort_values(["recommended_tier", "asset_name"])


def build_vector_lineage(df: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of assets that create or depend on embeddings."""
    vector_df = df[df["has_embeddings"]].copy()
    columns = [
        "asset_name",
        "storage_layer",
        "embedding_model",
        "lineage_parent",
        "snapshot_date",
    ]
    return vector_df[columns].sort_values("asset_name")


def recommend_storage_layer(
    access_pattern: str,
    latency_class: str,
    data_shape: str,
    mutability: str,
) -> Dict[str, str]:
    """Provide a coarse storage recommendation for common Chapter 3 cases."""
    # Retrieval-first workloads should not be treated like generic analytical
    # tables; the index exists because nearest-neighbor lookup is the contract.
    if data_shape == "vector" or access_pattern == "semantic_search":
        return {
            "recommended_layer": "vector index",
            "why": "Nearest-neighbor retrieval is the primary workload.",
        }
    # Low-latency entity access splits into relational versus NoSQL based on
    # whether strong transactional constraints still dominate.
    if latency_class == "low" and access_pattern in {"entity_lookup", "serving"}:
        if mutability == "transactional":
            return {
                "recommended_layer": "relational database",
                "why": "Low-latency access and strong constraints both matter.",
            }
        return {
            "recommended_layer": "NoSQL / operational store",
            "why": "Fast key-based access matters more than broad analytics.",
        }
    if access_pattern in {"historical_scan", "training", "analytics"}:
        return {
            "recommended_layer": "lakehouse table",
            "why": "The workload needs scan-friendly, version-aware analytical storage.",
        }
    # Default to object storage when flexibility and replay matter more than a
    # tight query model.
    return {
        "recommended_layer": "object storage",
        "why": "Flexible retention and replay matter more than tight query semantics.",
    }


def _retention_note(row: pd.Series) -> str:
    if row["recommended_tier"] == "hot":
        return "Keep in fast storage for active serving or frequent training reads."
    if row["recommended_tier"] == "archive-with-versioning":
        return "Archive carefully, but preserve version history for replay and audits."
    if row["recommended_tier"] == "archive":
        return "Move to low-cost storage once the working window closes."
    return "Keep in a standard tier and review lifecycle policy quarterly."
