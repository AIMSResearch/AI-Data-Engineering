"""Scenario-driven helpers for Chapter 7 ML integration planning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "scenario_name",
    "model_type",
    "feature_reuse_need",
    "online_latency_ms",
    "training_frequency",
    "registry_discipline",
    "rollback_ready",
    "feedback_logging",
    "serving_criticality",
    "point_in_time_training",
    "evaluation_discipline",
    "benchmark_versioned",
    "owner_clarity",
}


def load_integration_scenarios(path: str | Path) -> pd.DataFrame:
    """Load Chapter 7 integration scenarios from CSV."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def summarize_integration_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact summary of feature-store and serving pressure."""
    metrics = [
        {"metric": "scenarios", "value": int(len(df))},
        {
            "metric": "high_feature_reuse_need",
            "value": int((df["feature_reuse_need"] >= 8).sum()),
        },
        {
            "metric": "sub_100ms_online_latency",
            "value": int((df["online_latency_ms"] <= 100).sum()),
        },
        {
            "metric": "weak_registry_discipline",
            "value": int(df["registry_discipline"].isin(["low", "none"]).sum()),
        },
        {
            "metric": "missing_rollback_ready",
            "value": int((df["rollback_ready"] == "no").sum()),
        },
        {
            "metric": "missing_point_in_time_training",
            "value": int((df["point_in_time_training"] == "no").sum()),
        },
        {
            "metric": "weak_evaluation_discipline",
            "value": int(df["evaluation_discipline"].isin(["low", "none"]).sum()),
        },
        {
            "metric": "unversioned_benchmarks",
            "value": int((df["benchmark_versioned"] == "no").sum()),
        },
        {
            "metric": "unclear_owner_boundaries",
            "value": int(df["owner_clarity"].isin(["low", "none"]).sum()),
        },
    ]
    return pd.DataFrame(metrics)


def build_integration_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Attach coarse integration recommendations and risks to each scenario."""
    records = []

    for row in df.to_dict(orient="records"):
        # Keep these contracts separate so the notebook can show that release
        # fragility may come from evaluation or ownership, not only from models.
        feature_contract = _feature_contract(row)
        deployment_pattern = _deployment_pattern(row)
        feedback_priority = _feedback_priority(row)
        evaluation_contract = _evaluation_contract(row)
        ownership_contract = _ownership_contract(row)
        primary_risk = _primary_risk(row, feature_contract, deployment_pattern)

        records.append(
            {
                **row,
                "feature_contract": feature_contract,
                "deployment_pattern": deployment_pattern,
                "feedback_priority": feedback_priority,
                "evaluation_contract": evaluation_contract,
                "ownership_contract": ownership_contract,
                "primary_risk": primary_risk,
                "recommended_action": _recommended_action(row, primary_risk),
            }
        )

    return pd.DataFrame(records)


def find_fragile_integrations(df: pd.DataFrame) -> pd.DataFrame:
    """Return the scenarios with the highest integration fragility."""
    risk_terms = (
        "training-serving skew",
        "promotion without rollback",
        "weak model lineage",
        "feedback blind spot",
        "local feature logic",
        "evaluation drift",
        "ownership boundary",
    )
    mask = df["primary_risk"].str.contains("|".join(risk_terms), case=False, regex=True)
    return df.loc[
        mask,
        [
            "scenario_name",
            "feature_contract",
            "deployment_pattern",
            "feedback_priority",
            "evaluation_contract",
            "ownership_contract",
            "primary_risk",
            "recommended_action",
        ],
    ].reset_index(drop=True)


def _feature_contract(row: dict) -> str:
    reuse = int(row["feature_reuse_need"])
    latency = int(row["online_latency_ms"])
    point_in_time = row["point_in_time_training"] == "yes"

    if reuse >= 8 and latency <= 100 and point_in_time:
        return "shared feature store with offline and online paths"
    if reuse >= 6 or point_in_time:
        return "shared feature definitions with disciplined offline retrieval"
    return "embedded feature logic may still be sufficient"


def _deployment_pattern(row: dict) -> str:
    # A critical service without rollback is not release-ready even if the
    # registry metadata looks tidy.
    if row["serving_criticality"] == "critical" and row["rollback_ready"] == "yes":
        return "registry-gated deployment with rollback path"
    if row["registry_discipline"] in {"high", "medium"}:
        return "registry-backed deployment with explicit approvals"
    return "basic artifact deployment, but promotion discipline is weak"


def _feedback_priority(row: dict) -> str:
    if row["serving_criticality"] in {"high", "critical"} and row["feedback_logging"] == "full":
        return "full prediction and outcome logging is mandatory"
    if row["feedback_logging"] == "partial":
        return "expand logging before the next production iteration"
    if row["feedback_logging"] == "none":
        return "feedback capture is missing"
    return "routine feedback capture is enough"


def _evaluation_contract(row: dict) -> str:
    if row["evaluation_discipline"] == "high" and row["benchmark_versioned"] == "yes":
        return "versioned benchmark and evaluation lineage are in place"
    if row["evaluation_discipline"] in {"medium", "high"}:
        return "evaluation exists, but benchmark or run lineage still needs tightening"
    return "evaluation discipline is too weak for reliable model comparison"


def _ownership_contract(row: dict) -> str:
    if row["owner_clarity"] == "high":
        return "feature, serving, and rollback ownership is explicit"
    if row["owner_clarity"] == "medium":
        return "core ownership exists, but escalation boundaries could still blur"
    return "ownership boundaries are too unclear for reliable incident response"


def _primary_risk(row: dict, feature_contract: str, deployment_pattern: str) -> str:
    # Return the first contract failure a team is likely to feel in practice.
    if row["evaluation_discipline"] == "none" or row["benchmark_versioned"] == "no":
        return "evaluation drift risk because benchmark or metric lineage is weak"
    if row["owner_clarity"] == "none":
        return "ownership boundary failure is likely during rollback or serving incidents"
    if row["point_in_time_training"] == "no" and row["online_latency_ms"] <= 100:
        return "training-serving skew risk because offline and online states are not tied together"
    if deployment_pattern == "basic artifact deployment, but promotion discipline is weak":
        return "weak model lineage and promotion evidence around production releases"
    if row["rollback_ready"] == "no" and row["serving_criticality"] in {"high", "critical"}:
        return "promotion without rollback for a model-facing critical service"
    if row["feedback_logging"] == "none":
        return "feedback blind spot after deployment"
    if feature_contract == "embedded feature logic may still be sufficient" and int(row["feature_reuse_need"]) >= 6:
        return "local feature logic is starting to outgrow the current pattern"
    return "moderate integration risk if teams keep lineage and review discipline intact"


def _recommended_action(row: dict, primary_risk: str) -> str:
    if "training-serving skew" in primary_risk:
        return "introduce point-in-time training retrieval before expanding online usage"
    if "promotion without rollback" in primary_risk:
        return "add rollback targets and release gates before the next critical deployment"
    if "weak model lineage" in primary_risk:
        return "tie experiment tracking and registry promotion to explicit review evidence"
    if "feedback blind spot" in primary_risk:
        return "capture predictions, model version, and outcomes before trusting online metrics"
    if "evaluation drift" in primary_risk:
        return "version benchmarks and tie every evaluation run to explicit dataset, metric, and environment state"
    if "ownership boundary" in primary_risk:
        return "assign explicit owners for features, registry approval, serving health, and rollback decisions"
    if "local feature logic" in primary_risk:
        return "move repeated feature definitions into a shared contract before reuse expands further"
    return "keep the current pattern, but review registry and feedback practices regularly"
