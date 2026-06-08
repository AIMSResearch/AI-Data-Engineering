"""Architecture pattern helpers for AI data engineering examples.

The heuristics in this module are intentionally coarse. They are meant to make
trade-offs legible in the notebooks, not to behave like a production planner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def load_architecture_scenarios(path: str | Path) -> pd.DataFrame:
    """Load Chapter 4 architecture scenarios."""
    return pd.read_csv(path)


def summarize_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the scenario inventory by workload shape."""
    summary = (
        df.groupby(["latency_profile", "ownership_model"], dropna=False)
        .agg(
            scenario_count=("scenario_id", "count"),
            avg_team_count=("team_count", "mean"),
            avg_governance_pressure=("governance_pressure", "mean"),
        )
        .reset_index()
        .sort_values(["scenario_count", "avg_team_count"], ascending=[False, False])
    )
    return summary


def recommend_architecture(
    latency_profile: str,
    raw_multimodal: str,
    team_count: int,
    governance_pressure: int,
    online_serving_criticality: int,
    feature_reuse_need: int,
) -> Dict[str, str]:
    """Recommend a coarse architecture shape from a few scenario signals."""
    # Keep the recommendation decomposed so the notebook can discuss storage,
    # processing, ownership, and feature or prompt reuse as separate contracts.
    storage_pattern = _storage_pattern(raw_multimodal, governance_pressure)
    processing_pattern = _processing_pattern(latency_profile, online_serving_criticality)
    ownership_pattern = _ownership_pattern(team_count, governance_pressure)
    feature_pattern = _feature_pattern(feature_reuse_need, online_serving_criticality)
    reference_pattern = _reference_pattern(
        latency_profile=latency_profile,
        raw_multimodal=raw_multimodal,
        team_count=team_count,
        governance_pressure=governance_pressure,
        online_serving_criticality=online_serving_criticality,
        feature_reuse_need=feature_reuse_need,
    )
    required_layers = _required_layers(reference_pattern)

    summary = (
        f"{storage_pattern}; {processing_pattern}; {ownership_pattern}; "
        f"{feature_pattern}"
    )
    return {
        "reference_pattern": reference_pattern,
        "storage_pattern": storage_pattern,
        "processing_pattern": processing_pattern,
        "ownership_pattern": ownership_pattern,
        "feature_pattern": feature_pattern,
        "required_layers": required_layers,
        "reference_architecture": summary,
    }


def build_recommendation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply architecture recommendations to every scenario in the dataset."""
    recommendation_rows = []
    for _, row in df.iterrows():
        recommendation = recommend_architecture(
            latency_profile=row["latency_profile"],
            raw_multimodal=row["raw_multimodal"],
            team_count=int(row["team_count"]),
            governance_pressure=int(row["governance_pressure"]),
            online_serving_criticality=int(row["online_serving_criticality"]),
            feature_reuse_need=int(row["feature_reuse_need"]),
        )
        recommendation_rows.append(
            {
                "scenario_name": row["scenario_name"],
                "industry": row["industry"],
                **recommendation,
            }
        )
    return pd.DataFrame(recommendation_rows)


def _storage_pattern(raw_multimodal: str, governance_pressure: int) -> str:
    # High multimodal pressure usually implies documents, media, or embeddings
    # that benefit from cheaper raw storage plus a curated analytical layer.
    if raw_multimodal == "high":
        if governance_pressure >= 7:
            return "lakehouse-centered platform"
        return "object storage plus curated analytical layer"
    # When multimodal pressure is lower, governance burden becomes the main
    # reason to keep object storage under tighter analytical control.
    if governance_pressure >= 8:
        return "warehouse plus governed object storage"
    return "warehouse-first analytical platform"


def _processing_pattern(latency_profile: str, online_serving_criticality: int) -> str:
    # Critical serving paths usually need an offline plus online split even if
    # the source latency label is not explicitly "real_time".
    if latency_profile == "real_time" or online_serving_criticality >= 8:
        return "hybrid batch-and-streaming architecture"
    if latency_profile == "near_real_time":
        return "stream-informed hybrid architecture"
    return "batch-first architecture"


def _ownership_pattern(team_count: int, governance_pressure: int) -> str:
    # More teams plus non-trivial governance pressure usually means central
    # ownership alone becomes a bottleneck, so shared standards matter.
    if team_count >= 12 and governance_pressure >= 6:
        return "domain-oriented platform with shared standards"
    if team_count >= 8:
        return "central platform with domain-aligned datasets"
    return "centralized platform"


def _feature_pattern(feature_reuse_need: int, online_serving_criticality: int) -> str:
    # High reuse plus critical serving is the point where repeated local logic
    # starts creating the skew problems Chapter 4 warns about.
    if feature_reuse_need >= 8 and online_serving_criticality >= 7:
        return "dedicated feature-store layer"
    if feature_reuse_need >= 6:
        return "shared offline-first feature layer"
    return "pipeline-local feature logic"


def _reference_pattern(
    latency_profile: str,
    raw_multimodal: str,
    team_count: int,
    governance_pressure: int,
    online_serving_criticality: int,
    feature_reuse_need: int,
) -> str:
    # Prompt- and context-centered systems appear when runtime behavior depends
    # on more than retrieval alone: low-latency serving, high multimodal state,
    # and enough reuse pressure that prompt or context logic should not remain
    # trapped in local application code.
    if (
        raw_multimodal == "high"
        and latency_profile in {"near_real_time", "real_time"}
        and online_serving_criticality >= 7
        and feature_reuse_need >= 8
        and governance_pressure <= 6
    ):
        return "prompt_context_centered_ai_platform"

    # Retrieval-centered systems depend on corpus freshness, chunking, and
    # index behavior even when prompt orchestration is less prominent.
    if raw_multimodal == "high" and online_serving_criticality >= 6:
        return "retrieval_centered_ai_platform"

    if team_count >= 12 and governance_pressure >= 6:
        return "domain_oriented_data_platform"

    if latency_profile in {"near_real_time", "real_time"}:
        return "hybrid_lakehouse_platform"

    return "centralized_analytical_core"


def _required_layers(reference_pattern: str) -> str:
    if reference_pattern == "prompt_context_centered_ai_platform":
        return (
            "prompt registry; corpus registry; permission-aware retrieval; "
            "context trace logging; prompt regression tests; deployment manifest"
        )
    if reference_pattern == "retrieval_centered_ai_platform":
        return (
            "corpus registry; chunking and embedding lineage; vector serving "
            "layer; retrieval evaluation"
        )
    if reference_pattern == "domain_oriented_data_platform":
        return "shared standards; domain data products; cross-domain lineage"
    if reference_pattern == "hybrid_lakehouse_platform":
        return "offline and online data paths; governed analytical core"
    return "central analytical storage; shared transformation ownership"
