"""Scenario-driven helpers for Chapter 6 control and governance exercises."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "asset_name",
    "domain",
    "sensitivity_class",
    "criticality",
    "freshness_delay_hours",
    "null_rate_pct",
    "duplicate_rate_pct",
    "schema_drift_events_30d",
    "lineage_coverage_pct",
    "owner_defined",
    "retention_policy_defined",
    "access_review_age_days",
    "asset_type",
    "benchmark_owner_defined",
    "retrieval_freshness_hours",
    "embedding_refresh_age_days",
}

MODEL_FACING_ASSET_TYPES = {
    "feature_table",
    "benchmark",
    "retrieval_index",
    "embedding_store",
    "prompt_template",
    "context_trace_log",
    "tool_output_log",
    "generated_response_log",
    "retrieval_policy",
}


def load_quality_assets(path: str | Path) -> pd.DataFrame:
    """Load Chapter 6 asset-control scenarios from CSV."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def summarize_control_posture(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact summary of quality and governance posture."""
    metrics = [
        {"metric": "assets", "value": int(len(df))},
        {
            "metric": "restricted_assets",
            "value": int((df["sensitivity_class"] == "restricted").sum()),
        },
        {
            "metric": "stale_assets_over_24h",
            "value": int((df["freshness_delay_hours"] > 24).sum()),
        },
        {
            "metric": "lineage_under_80pct",
            "value": int((df["lineage_coverage_pct"] < 80).sum()),
        },
        {
            "metric": "missing_owner",
            "value": int((df["owner_defined"] == "no").sum()),
        },
        {
            "metric": "missing_retention_policy",
            "value": int((df["retention_policy_defined"] == "no").sum()),
        },
        {
            "metric": "model_facing_assets",
            "value": int(df["asset_type"].isin(MODEL_FACING_ASSET_TYPES).sum()),
        },
        {
            "metric": "benchmark_assets_without_owner",
            "value": int(
                ((df["asset_type"] == "benchmark") & (df["benchmark_owner_defined"] == "no")).sum()
            ),
        },
        {
            "metric": "retrieval_assets_stale_over_24h",
            "value": int(
                (
                    df["asset_type"].isin(["retrieval_index", "embedding_store"])
                    & (df["retrieval_freshness_hours"] > 24)
                ).sum()
            ),
        },
    ]
    return pd.DataFrame(metrics)


def assess_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Attach control-risk scores and recommended actions to each asset."""
    records = []

    for row in df.to_dict(orient="records"):
        # These scores are teaching heuristics. They help compare assets across
        # freshness, governance, and security dimensions without pretending that
        # one scalar replaces real operational judgment.
        quality_risk = _quality_risk(row)
        governance_risk = _governance_risk(row)
        security_risk = _security_risk(row)
        total_risk = quality_risk + governance_risk + security_risk

        if total_risk >= 13:
            priority = "critical"
        elif total_risk >= 8:
            priority = "high"
        elif total_risk >= 4:
            priority = "moderate"
        else:
            priority = "low"

        records.append(
            {
                **row,
                "quality_risk_score": quality_risk,
                "governance_risk_score": governance_risk,
                "security_risk_score": security_risk,
                "total_risk_score": total_risk,
                "priority": priority,
                "primary_gap": _primary_gap(row),
                "observability_gap": _observability_gap(row),
                "recommended_action": _recommended_action(row, priority),
            }
        )

    return pd.DataFrame(records)


def find_high_priority_assets(df: pd.DataFrame) -> pd.DataFrame:
    """Return assets that need the fastest control response."""
    return (
        df.loc[
            df["priority"].isin(["critical", "high"]),
            [
                "asset_name",
                "sensitivity_class",
                "criticality",
                "priority",
                "primary_gap",
                "observability_gap",
                "recommended_action",
            ],
        ]
        .sort_values(["priority", "asset_name"], ascending=[True, True])
        .reset_index(drop=True)
    )


def _quality_risk(row: dict) -> int:
    score = 0
    if float(row["freshness_delay_hours"]) > 24:
        score += 3
    elif float(row["freshness_delay_hours"]) > 6:
        score += 1

    if float(row["null_rate_pct"]) >= 5:
        score += 2
    elif float(row["null_rate_pct"]) >= 1:
        score += 1

    if float(row["duplicate_rate_pct"]) >= 1:
        score += 2
    elif float(row["duplicate_rate_pct"]) > 0:
        score += 1

    if int(row["schema_drift_events_30d"]) >= 3:
        score += 2
    elif int(row["schema_drift_events_30d"]) > 0:
        score += 1

    if row["asset_type"] in {"retrieval_index", "embedding_store"}:
        if float(row["retrieval_freshness_hours"]) > 24:
            score += 2
        elif float(row["retrieval_freshness_hours"]) > 6:
            score += 1

        if int(row["embedding_refresh_age_days"]) > 30:
            score += 1

    if row["asset_type"] in {"prompt_template", "context_trace_log", "tool_output_log", "generated_response_log", "retrieval_policy"}:
        if float(row["lineage_coverage_pct"]) < 80:
            score += 1
        if int(row["access_review_age_days"]) > 120:
            score += 1

    return score


def _governance_risk(row: dict) -> int:
    score = 0
    if float(row["lineage_coverage_pct"]) < 50:
        score += 3
    elif float(row["lineage_coverage_pct"]) < 80:
        score += 1

    if row["owner_defined"] == "no":
        score += 2

    if row["retention_policy_defined"] == "no":
        score += 2

    if row["asset_type"] == "benchmark" and row["benchmark_owner_defined"] == "no":
        score += 2

    return score


def _security_risk(row: dict) -> int:
    score = 0
    sensitivity = row["sensitivity_class"]
    review_age = int(row["access_review_age_days"])

    if sensitivity == "restricted":
        score += 2
    elif sensitivity == "confidential":
        score += 1

    if review_age > 180:
        score += 3
    elif review_age > 90:
        score += 1

    if row["criticality"] == "high" and sensitivity in {"restricted", "confidential"}:
        score += 1

    return score


def _primary_gap(row: dict) -> str:
    if row["owner_defined"] == "no":
        return "no accountable owner"
    if row["asset_type"] == "benchmark" and row["benchmark_owner_defined"] == "no":
        return "benchmark changes have no accountable owner"
    if row["asset_type"] == "prompt_template" and float(row["lineage_coverage_pct"]) < 80:
        return "prompt release evidence is too weak"
    if row["asset_type"] == "context_trace_log" and row["retention_policy_defined"] == "no":
        return "runtime traces have no explicit retention rule"
    if row["retention_policy_defined"] == "no":
        return "retention policy missing"
    if float(row["lineage_coverage_pct"]) < 50:
        return "lineage coverage too low for investigation"
    if int(row["access_review_age_days"]) > 180:
        return "access reviews are stale"
    if row["asset_type"] in {"retrieval_index", "embedding_store"} and float(row["retrieval_freshness_hours"]) > 24:
        return "retrieval state is stale enough to affect downstream behavior"
    if float(row["freshness_delay_hours"]) > 24:
        return "freshness failure exceeds operational tolerance"
    if int(row["schema_drift_events_30d"]) >= 3:
        return "repeated schema drift without stronger contract control"
    return "quality and control posture needs routine follow-up"


def _observability_gap(row: dict) -> str:
    if row["asset_type"] == "benchmark":
        if row["benchmark_owner_defined"] == "no":
            return "benchmark drift is likely because ownership and approval are unclear"
        return "watch for benchmark drift, relabel history, and split changes"
    if row["asset_type"] in {"retrieval_index", "embedding_store"}:
        # Retrieval assets often fail quietly; the model can look healthy while
        # stale chunks or embeddings are driving degraded answers.
        if float(row["retrieval_freshness_hours"]) > 24:
            return "retrieval freshness should page before model behavior degrades"
        return "track chunking, embedding, and index refresh changes alongside lineage"
    if row["asset_type"] == "prompt_template":
        return "track prompt version, regression results, and rollback target together"
    if row["asset_type"] in {"context_trace_log", "tool_output_log", "generated_response_log"}:
        return "watch retention, access review age, and trace completeness together"
    if row["asset_type"] == "retrieval_policy":
        return "track authorization-rule changes as part of the serving path, not only as static config"
    if row["asset_type"] == "feature_table":
        return "track freshness and schema drift with lineage-linked alerts"
    return "routine observability is usually enough"


def _recommended_action(row: dict, priority: str) -> str:
    if priority == "critical":
        return "pause downstream publication, assign owner, and review access plus retention immediately"
    # Ownership and approval gaps are treated as operational blockers because
    # later quality investigations have no credible decision-maker without them.
    if row["owner_defined"] == "no":
        return "assign a dataset owner before allowing new downstream use"
    if row["asset_type"] == "benchmark" and row["benchmark_owner_defined"] == "no":
        return "assign a benchmark owner and approval path before using this asset for model comparison"
    if row["asset_type"] == "prompt_template" and float(row["lineage_coverage_pct"]) < 80:
        return "block prompt promotion until regression evidence and lineage are attached"
    if row["asset_type"] == "context_trace_log" and row["retention_policy_defined"] == "no":
        return "define retention and redaction policy before keeping more runtime traces"
    if row["retention_policy_defined"] == "no":
        return "define retention and deletion workflow before the next promotion"
    if float(row["lineage_coverage_pct"]) < 80:
        return "improve lineage capture and run metadata for faster investigation"
    if row["asset_type"] in {"retrieval_index", "embedding_store"} and float(row["retrieval_freshness_hours"]) > 6:
        return "tighten retrieval freshness monitoring and tie index rebuilds to source and embedding lineage"
    if float(row["freshness_delay_hours"]) > 6:
        return "tighten freshness monitoring and escalation thresholds"
    return "keep current controls but review severity thresholds with the owning team"
