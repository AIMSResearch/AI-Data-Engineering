"""Scenario-driven helpers for Chapter 5 processing design exercises."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "pipeline_name",
    "source_pattern",
    "freshness_sla_minutes",
    "rows_per_day_millions",
    "late_data_pct",
    "full_refresh_runtime_minutes",
    "incremental_runtime_minutes",
    "transformation_complexity",
    "stateful_join",
    "online_dependency",
    "unique_key_available",
    "replayable_input",
    "evaluation_asset",
    "chunking_or_embedding_stage",
    "benchmark_versioned",
}


def load_processing_scenarios(path: str | Path) -> pd.DataFrame:
    """Load Chapter 5 processing scenarios from CSV."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def summarize_processing_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact summary of processing pressure across the scenarios."""
    metrics = [
        {
            "metric": "pipelines",
            "value": int(len(df)),
        },
        {
            "metric": "freshness_sla_under_15m",
            "value": int((df["freshness_sla_minutes"] <= 15).sum()),
        },
        {
            "metric": "rows_over_50m_per_day",
            "value": int((df["rows_per_day_millions"] >= 50).sum()),
        },
        {
            "metric": "stateful_joins",
            "value": int((df["stateful_join"] == "yes").sum()),
        },
        {
            "metric": "non_replayable_inputs",
            "value": int((df["replayable_input"] == "no").sum()),
        },
        {
            "metric": "missing_unique_key",
            "value": int((df["unique_key_available"] == "no").sum()),
        },
        {
            "metric": "evaluation_assets",
            "value": int((df["evaluation_asset"] == "yes").sum()),
        },
        {
            "metric": "retrieval_prep_stages",
            "value": int((df["chunking_or_embedding_stage"] == "yes").sum()),
        },
        {
            "metric": "unversioned_benchmarks",
            "value": int(
                ((df["evaluation_asset"] == "yes") & (df["benchmark_versioned"] == "no")).sum()
            ),
        },
    ]
    return pd.DataFrame(metrics)


def build_processing_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Attach coarse processing recommendations and risks to each scenario."""
    records = []

    for row in df.to_dict(orient="records"):
        # Runtime reduction is explanatory rather than authoritative; it helps
        # readers see why incremental logic is tempting before we discuss risk.
        runtime_reduction_pct = round(
            100
            * (1 - (row["incremental_runtime_minutes"] / row["full_refresh_runtime_minutes"])),
            1,
        )
        recommended_mode = _recommend_processing_mode(row)
        recommended_engine = _recommend_engine(row, recommended_mode)
        replay_strategy = _recommend_replay_strategy(row)
        primary_risk = _primary_risk(row, recommended_mode)
        evaluation_guidance = _evaluation_guidance(row)
        retrieval_guidance = _retrieval_guidance(row)

        records.append(
            {
                **row,
                "recommended_mode": recommended_mode,
                "recommended_engine": recommended_engine,
                "replay_strategy": replay_strategy,
                "runtime_reduction_pct": runtime_reduction_pct,
                "primary_risk": primary_risk,
                "evaluation_guidance": evaluation_guidance,
                "retrieval_guidance": retrieval_guidance,
            }
        )

    return pd.DataFrame(records)


def find_high_risk_pipelines(df: pd.DataFrame) -> pd.DataFrame:
    """Return the scenarios with the clearest operational risk."""
    risk_terms = (
        "unsafe incremental merge",
        "non-replayable source",
        "freshness pressure",
        "late-data corrections",
        "stream-state burden",
        "benchmark drift",
        "retrieval behavior",
    )
    mask = df["primary_risk"].str.contains("|".join(risk_terms), case=False, regex=True)
    return df.loc[
        mask,
        [
            "pipeline_name",
            "recommended_mode",
            "recommended_engine",
            "replay_strategy",
            "evaluation_guidance",
            "retrieval_guidance",
            "primary_risk",
        ],
    ].reset_index(drop=True)


def _recommend_processing_mode(row: dict) -> str:
    freshness = int(row["freshness_sla_minutes"])
    online = row["online_dependency"]
    late_data = float(row["late_data_pct"])
    stateful = row["stateful_join"] == "yes"

    if freshness <= 5 or (stateful and online in {"high", "critical"}):
        return "hybrid streaming plus batch replay"
    if freshness <= 60 or late_data >= 4.0:
        return "incremental batch with correction window"
    if row["incremental_runtime_minutes"] <= row["full_refresh_runtime_minutes"] * 0.5:
        return "incremental batch"
    return "scheduled batch"


def _recommend_engine(row: dict, recommended_mode: str) -> str:
    if recommended_mode == "hybrid streaming plus batch replay":
        return "stream processor plus distributed batch engine"
    if row["rows_per_day_millions"] >= 25 or row["transformation_complexity"] == "high":
        return "distributed SQL or DataFrame engine"
    return "warehouse SQL or single-node dataframe"


def _recommend_replay_strategy(row: dict) -> str:
    if row["replayable_input"] == "no":
        return "capture immutable landing copies before trusting incremental logic"
    if float(row["late_data_pct"]) >= 4.0:
        return "recompute bounded recent windows on every run"
    if row["stateful_join"] == "yes":
        return "retain raw event history and schedule periodic state repair"
    return "partition reruns are usually enough"


def _primary_risk(row: dict, recommended_mode: str) -> str:
    # Prioritize the failure the chapter would want the reader to discuss first,
    # not a complete risk register for the pipeline.
    if row["evaluation_asset"] == "yes" and row["benchmark_versioned"] == "no":
        return "benchmark drift risk because evaluation data is not versioned"
    if row["chunking_or_embedding_stage"] == "yes" and row["replayable_input"] == "no":
        return "retrieval behavior can change without a clean replay boundary"
    if recommended_mode != "scheduled batch" and row["unique_key_available"] == "no":
        return "unsafe incremental merge because no stable key is available"
    if row["replayable_input"] == "no":
        return "non-replayable source makes recovery and logic changes expensive"
    if recommended_mode == "hybrid streaming plus batch replay":
        return "stream-state burden can hide drift until online metrics move"
    if float(row["late_data_pct"]) >= 4.0:
        return "late-data corrections can silently age feature tables"
    if row["full_refresh_runtime_minutes"] >= row["freshness_sla_minutes"]:
        return "freshness pressure already exceeds the current full-refresh budget"
    return "moderate operational risk if ownership and tests stay clear"


def _evaluation_guidance(row: dict) -> str:
    if row["evaluation_asset"] != "yes":
        return "not an evaluation-specific pipeline"
    # Evaluation assets need explicit versioning because small rebuild changes
    # can invalidate model comparisons without touching training code.
    if row["benchmark_versioned"] == "no":
        return "treat this output as a versioned benchmark or label set before comparing models"
    if float(row["late_data_pct"]) >= 4.0:
        return "watch for leakage and relabel windows when rebuilding benchmark slices"
    return "keep benchmark lineage, split logic, and relabel history with the pipeline run"


def _retrieval_guidance(row: dict) -> str:
    if row["chunking_or_embedding_stage"] != "yes":
        return "no retrieval-preparation stage in this scenario"
    # Retrieval preparation needs the same reproducibility discipline as feature
    # pipelines because chunking or embedding changes alter runtime behavior.
    if row["replayable_input"] == "no":
        return "capture immutable source copies before chunking or embedding refreshes"
    if row["freshness_sla_minutes"] <= 60:
        return "tie chunking and embedding refreshes to explicit source snapshots and index rebuild windows"
    return "record chunking logic and embedding model version with every rebuild"
