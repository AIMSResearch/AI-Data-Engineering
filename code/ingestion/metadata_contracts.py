"""Minimal helpers for ingestion metadata and contract checks.

These helpers are intentionally small so the Chapter 2 notebook can focus on
what the contract preserves rather than on framework-specific boilerplate.
"""


def validate_required_fields(record, required_fields):
    # Treat missing and empty-string values the same way because both break the
    # downstream assumption that a required source field is usable.
    missing = [field for field in required_fields if not record.get(field)]
    return {"valid": not missing, "missing_fields": missing}


def validate_timestamp_not_null(record, field_name="event_time"):
    # Timestamps are a minimum requirement for replay, ordering, and freshness.
    value = record.get(field_name)
    return {"valid": value is not None and value != "", "field": field_name}


def build_ingestion_sidecar(
    source_id,
    schema_version,
    extraction_window,
    validation_status,
    ingestion_time,
):
    # The sidecar keeps source-contract metadata close to the payload so later
    # debugging can reconstruct where the record came from and what checks ran.
    return {
        "source_id": source_id,
        "schema_version": schema_version,
        "extraction_window": extraction_window,
        "validation_status": validation_status,
        "ingestion_time": ingestion_time,
    }
