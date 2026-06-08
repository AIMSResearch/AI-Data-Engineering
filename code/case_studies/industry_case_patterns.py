"""Scenario-driven helpers for Chapter 8 case-study comparisons."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "case_name",
    "industry",
    "dominant_pattern",
    "latency_pressure",
    "governance_pressure",
    "multimodal_pressure",
    "agentic_pressure",
    "feature_reuse_score",
    "current_platform_shape",
    "primary_failure_mode",
    "next_investment",
}


def load_case_profiles(path: str | Path) -> pd.DataFrame:
    """Load Chapter 8 case-study profiles from CSV."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def summarize_case_landscape(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact summary of Chapter 8 pressure signals."""
    metrics = [
        {"metric": "cases", "value": int(len(df))},
        {
            "metric": "high_latency_pressure",
            "value": int(df["latency_pressure"].isin(["high", "critical"]).sum()),
        },
        {
            "metric": "high_governance_pressure",
            "value": int(df["governance_pressure"].isin(["high", "critical"]).sum()),
        },
        {
            "metric": "high_multimodal_pressure",
            "value": int(df["multimodal_pressure"].isin(["high", "critical"]).sum()),
        },
        {
            "metric": "high_agentic_pressure",
            "value": int(df["agentic_pressure"].isin(["high", "critical"]).sum()),
        },
        {
            "metric": "strong_feature_reuse_cases",
            "value": int((df["feature_reuse_score"] >= 7).sum()),
        },
    ]
    return pd.DataFrame(metrics)


def build_case_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Attach emphasis areas and next-step guidance to each case."""
    records = []

    for row in df.to_dict(orient="records"):
        emphasis_area = _emphasis_area(row)
        future_pressure = _future_pressure(row)
        records.append(
            {
                **row,
                "emphasis_area": emphasis_area,
                "future_pressure": future_pressure,
                "recommended_guidance": _recommended_guidance(
                    row, emphasis_area, future_pressure
                ),
            }
        )

    return pd.DataFrame(records)


def find_future_pressure_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Return the cases most exposed to next-wave platform pressure."""
    mask = df["future_pressure"].isin(
        [
            "retrieval and multimodal state pressure is rising",
            "agentic workflow state is becoming a platform concern",
            "control and audit evidence pressure dominates the roadmap",
        ]
    )
    return df.loc[
        mask,
        [
            "case_name",
            "industry",
            "emphasis_area",
            "future_pressure",
            "primary_failure_mode",
            "recommended_guidance",
        ],
    ].reset_index(drop=True)


def _emphasis_area(row: dict) -> str:
    if row["governance_pressure"] in {"high", "critical"}:
        return "control, lineage, and approval discipline"
    if int(row["feature_reuse_score"]) >= 7:
        return "shared feature definitions and platform reuse"
    if row["latency_pressure"] in {"high", "critical"}:
        return "offline and online serving alignment"
    return "workflow reproducibility and platform ergonomics"


def _future_pressure(row: dict) -> str:
    if row["agentic_pressure"] in {"high", "critical"}:
        return "agentic workflow state is becoming a platform concern"
    if row["multimodal_pressure"] in {"high", "critical"}:
        return "retrieval and multimodal state pressure is rising"
    if row["governance_pressure"] in {"high", "critical"}:
        return "control and audit evidence pressure dominates the roadmap"
    return "incremental platform hardening is the likely next step"


def _recommended_guidance(
    row: dict, emphasis_area: str, future_pressure: str
) -> str:
    if "control" in emphasis_area:
        return "strengthen ownership, lineage, and promotion evidence before expanding model scope"
    if "shared feature definitions" in emphasis_area:
        return "invest in reusable feature contracts before more teams duplicate the same logic"
    if "serving alignment" in emphasis_area:
        return "tie model rollout to feature freshness, retrieval state, and rollback readiness"
    if "agentic workflow state" in future_pressure:
        return "treat workflow memory, checkpoints, and tool traces as governed platform data"
    if "retrieval and multimodal" in future_pressure:
        return "version embeddings, indexes, and source snapshots together instead of separately"
    return "preserve reproducible workflow state and add governance only where the risk justifies it"
