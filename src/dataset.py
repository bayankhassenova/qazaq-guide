"""Loading the dataset and turning each record into a text chunk + metadata."""
from __future__ import annotations

import json
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "places.json"

REQUIRED_FIELDS = [
    "id", "name", "doc_type", "region", "category", "tags",
    "best_seasons", "budget", "description", "source",
]
SEASONS = {"spring", "summer", "autumn", "winter"}
BUDGETS = {"low", "medium", "high"}


def load_records(path: Path = DATA_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    validate(records)
    return records


def validate(records: list[dict]) -> None:
    """Fail early if a record is badly annotated."""
    seen = set()
    for r in records:
        missing = [k for k in REQUIRED_FIELDS if k not in r or r[k] in (None, "")]
        if missing:
            raise ValueError(f"Record {r.get('id')} is missing fields: {missing}")
        if r["id"] in seen:
            raise ValueError(f"Duplicate id: {r['id']}")
        seen.add(r["id"])
        if not set(r["best_seasons"]) <= SEASONS:
            raise ValueError(f"Record {r['id']} has unknown season in {r['best_seasons']}")
        if r["budget"] not in BUDGETS:
            raise ValueError(f"Record {r['id']} has unknown budget {r['budget']}")


def record_to_chunk(r: dict) -> str:
    """Build the text that will be embedded.

    Every record is short (about 150-250 words), so one record = one chunk.
    The metadata (region, season, budget...) is written into the text too,
    so a question like "cheap trip in winter" also matches semantically.
    """
    parts = [
        f"Name: {r['name']}",
        f"Region: {r['region']}" + (f" (nearest city: {r['nearest_city']})" if r.get("nearest_city") else ""),
        f"Category: {r['category']}. Tags: {', '.join(r['tags'])}",
        f"Best seasons: {', '.join(r['best_seasons'])}. Budget: {r['budget']}.",
        f"Description: {r['description']}",
    ]
    if r.get("how_to_get_there"):
        parts.append(f"How to get there: {r['how_to_get_there']}")
    if r.get("tips"):
        parts.append(f"Tips: {r['tips']}")
    return "\n".join(parts)


def record_to_payload(r: dict) -> dict:
    """Metadata stored next to the vector in the database (used for filters and sources)."""
    return {
        "doc_id": r["id"],
        "name": r["name"],
        "doc_type": r["doc_type"],
        "region": r["region"],
        "nearest_city": r.get("nearest_city", ""),
        "distance_km": r.get("distance_km", 0),
        "category": r["category"],
        "tags": r["tags"],
        "best_seasons": r["best_seasons"],
        "budget": r["budget"],
        "recommended_days": r.get("recommended_days", 0),
        "source": r["source"],
        "text": record_to_chunk(r),
    }
