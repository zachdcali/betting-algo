import json
import math
from pathlib import Path
import sys

import pandas as pd
import pytest


PRODUCTION = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PRODUCTION))

from feature_contract import feature_fingerprint  # noqa: E402
from feature_lineage import (  # noqa: E402
    FeatureLineageConflict,
    read_feature_csv,
    resolve_feature_lineage,
)
from models.inference import EXACT_141_FEATURES  # noqa: E402


def _vector(rank: float = 12.0) -> dict[str, float]:
    vector = {name: 0.0 for name in EXACT_141_FEATURES}
    vector.update({
        "Player1_Rank": rank,
        "Surface_Hard": 1.0,
        "Level_A": 1.0,
        "Round_R32": 1.0,
        "P1_Hand_U": 1.0,
        "P2_Hand_U": 1.0,
        "P1_Country_Other": 1.0,
        "P2_Country_Other": 1.0,
    })
    return vector


def _immutable(vector: dict[str, float], *, run_id: str = "run_1") -> dict:
    schema_hash, vector_hash, count = feature_fingerprint(
        vector, EXACT_141_FEATURES
    )
    return {
        **vector,
        "run_id": run_id,
        "match_uid": "match_1",
        "feature_snapshot_id": "feature_1",
        "status": "ok",
        "feature_schema_sha256": schema_hash,
        "feature_vector_sha256": vector_hash,
        "feature_count": count,
    }


def _derived(vector: dict[str, float], *, run_id: str = "run_1") -> dict:
    schema_hash, vector_hash, count = feature_fingerprint(
        vector, EXACT_141_FEATURES
    )
    return {
        "run_id": run_id,
        "match_uid": "match_1",
        "feature_snapshot_id": "feature_1",
        "build_status": "ok",
        "features_complete": True,
        "feature_schema_sha256": schema_hash,
        "feature_vector_sha256": vector_hash,
        "feature_count": count,
        "features_json": json.dumps(vector, separators=(",", ":")),
    }


def test_ulp_equivalent_derived_copy_uses_immutable_vector_and_hash():
    immutable_vector = _vector(12.0)
    derived_vector = _vector(math.nextafter(12.0, math.inf))
    immutable = _immutable(immutable_vector)
    derived = _derived(derived_vector)
    assert immutable["feature_vector_sha256"] != derived["feature_vector_sha256"]

    result = resolve_feature_lineage(
        ordered_names=EXACT_141_FEATURES,
        immutable_sources=[("logs/features_run.csv", pd.DataFrame([immutable]))],
        derived_sources=[("logs/feature_vectors.csv", pd.DataFrame([derived]))],
    )

    authority = result.canonical_by_id["feature_1"]
    assert authority.source_kind == "immutable"
    assert authority.vector["Player1_Rank"] == 12.0
    assert authority.verified_vector_sha256 == immutable["feature_vector_sha256"]
    assert result.referential_vector_hashes("feature_1") == {
        immutable["feature_vector_sha256"],
    }


def test_materially_divergent_derived_copy_fails_closed():
    with pytest.raises(FeatureLineageConflict, match="materially divergent derived"):
        resolve_feature_lineage(
            ordered_names=EXACT_141_FEATURES,
            immutable_sources=[(
                "logs/features_run.csv", pd.DataFrame([_immutable(_vector(12.0))])
            )],
            derived_sources=[(
                "logs/feature_vectors.csv", pd.DataFrame([_derived(_vector(12.01))])
            )],
        )


def test_conflicting_immutable_sources_fail_closed():
    with pytest.raises(FeatureLineageConflict, match="immutable vectors"):
        resolve_feature_lineage(
            ordered_names=EXACT_141_FEATURES,
            immutable_sources=[
                ("logs/features_first.csv", pd.DataFrame([_immutable(_vector(12.0))])),
                ("logs/features_second.csv", pd.DataFrame([_immutable(_vector(12.01))])),
            ],
        )


def test_run_identity_mismatch_fails_closed():
    with pytest.raises(FeatureLineageConflict, match="run/match/schema identity"):
        resolve_feature_lineage(
            ordered_names=EXACT_141_FEATURES,
            immutable_sources=[(
                "logs/features_run.csv", pd.DataFrame([_immutable(_vector(), run_id="run_1")])
            )],
            derived_sources=[(
                "logs/feature_vectors.csv", pd.DataFrame([_derived(_vector(), run_id="run_2")])
            )],
        )


@pytest.mark.parametrize("malformed_count", ["not-a-count", "141.5"])
def test_malformed_nonblank_feature_count_fails_closed(malformed_count):
    immutable = _immutable(_vector())
    immutable["feature_count"] = malformed_count

    result = resolve_feature_lineage(
        ordered_names=EXACT_141_FEATURES,
        immutable_sources=[("logs/features_run.csv", pd.DataFrame([immutable]))],
    )

    authority = result.canonical_by_id["feature_1"]
    assert "source_feature_count_invalid" in authority.metadata_issues
    assert "feature_1" in result.invalid_ids
    assert not result.referential_vector_hashes("feature_1")


def test_partial_status_ok_derived_copies_cannot_hide_material_difference():
    first = _derived(_vector(12.0))
    second = _derived(_vector(13.0))
    for row in (first, second):
        payload = json.loads(row["features_json"])
        payload.pop("Player2_Age")
        row["features_json"] = json.dumps(payload, separators=(",", ":"))
        row["feature_vector_sha256"] = ""

    with pytest.raises(
        FeatureLineageConflict,
        match="contradictory derived copies|invalid/partial",
    ):
        resolve_feature_lineage(
            ordered_names=EXACT_141_FEATURES,
            derived_sources=[
                ("durable.csv", pd.DataFrame([first])),
                ("local.csv", pd.DataFrame([second])),
            ],
        )


def test_sparse_nondecision_skip_copy_can_follow_full_immutable_skip_row():
    immutable = _immutable(_vector(12.0))
    immutable.update(status="skip", feature_vector_sha256="")
    derived = _derived(_vector(12.0))
    derived.update(
        build_status="skip",
        features_complete=False,
        feature_vector_sha256="",
        features_json=json.dumps({"_defaulted_features": ""}),
    )

    result = resolve_feature_lineage(
        ordered_names=EXACT_141_FEATURES,
        immutable_sources=[("features_run.csv", pd.DataFrame([immutable]))],
        derived_sources=[("feature_vectors.csv", pd.DataFrame([derived]))],
    )

    assert "feature_1" in result.invalid_ids
    assert result.canonical_by_id["feature_1"].source_kind == "immutable"


def test_sparse_nondecision_skip_copy_cannot_contradict_retained_field():
    immutable = _immutable(_vector(12.0))
    immutable.update(status="skip", feature_vector_sha256="")
    derived = _derived(_vector(12.0))
    derived.update(
        build_status="skip",
        features_complete=False,
        feature_vector_sha256="",
        features_json=json.dumps({"Player1_Rank": 13.0}),
    )

    with pytest.raises(
        FeatureLineageConflict,
        match="contradictory sparse non-decision-grade payloads",
    ):
        resolve_feature_lineage(
            ordered_names=EXACT_141_FEATURES,
            immutable_sources=[("features_run.csv", pd.DataFrame([immutable]))],
            derived_sources=[("feature_vectors.csv", pd.DataFrame([derived]))],
        )


def test_feature_csv_parser_round_trips_serialized_float(tmp_path):
    value = 29.215605749486652
    path = tmp_path / "features_run.csv"
    pd.DataFrame([{"feature_snapshot_id": "feature_1", "Player2_Age": value}]).to_csv(
        path, index=False
    )

    parsed = read_feature_csv(path)

    assert parsed.iloc[0]["Player2_Age"] == float("29.215605749486652")
