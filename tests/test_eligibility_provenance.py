from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
sys.path.insert(0, str(PRODUCTION))

from storage.eligibility import (  # noqa: E402
    ELIGIBILITY_CONTRACT_VERSION,
    EligibilityContractError,
    EligibilityProjectionReader,
    EvidenceSource,
    ResolutionStatus,
    eligibility_generation,
    eligibility_generation_sha256,
    eligibility_generation_status_event,
    eligibility_review_event,
    normalize_identity_name,
    normalize_profile_value,
    normalize_round_code,
    player_alias_observation,
    player_profile_observation,
    projection_seal_from_batches,
    project_compatibility_rows,
)
from storage.live_records import (  # noqa: E402
    LiveRecordBuilder, eligibility_match_round_observation, match_identity,
)
from versioning import OPERATIONAL_SCHEMA_VERSION  # noqa: E402


NOW = datetime(2026, 7, 14, 21, 0, tzinfo=timezone.utc)
DIGEST = "a" * 64
GENERATION = "b" * 64
SEAL = "c" * 64
ARTIFACT_ID = "00000000-0000-0000-0000-000000000099"


def _source(**overrides):
    values = {
        "source_name": "atp_official",
        "source_uri": "https://www.atptour.com/en/players/example/overview",
        "source_content_sha256": DIGEST,
        "observed_at": NOW,
        "confidence": Decimal("1"),
        "initial_review_state": "unreviewed",
        "source_artifact_id": ARTIFACT_ID,
    }
    values.update(overrides)
    return EvidenceSource.validated(**values)


def _row(batch):
    assert len(batch.unique_records) == 1
    return batch.unique_records[0]


def test_generation_and_status_are_separate_append_only_records():
    manifest = {"players_sha256": DIGEST}
    generation = eligibility_generation(
        generation_sequence=7,
        effective_at=NOW,
        source_manifest=manifest,
        expected_projection_seal_sha256=SEAL,
        expected_projection_row_count=4,
    )
    row = _row(generation)
    assert row["contract_version"] == ELIGIBILITY_CONTRACT_VERSION
    assert "status" not in row
    assert row["generation_sha256"] == eligibility_generation_sha256(manifest)
    assert row["expected_projection_seal_sha256"] == SEAL
    assert row["expected_projection_row_count"] == 4

    candidate = _row(eligibility_generation_status_event(
        generation_sha256=row["generation_sha256"],
        generation_sequence=7,
        status="candidate",
        effective_at=NOW + timedelta(minutes=5),
        reviewed_by="staging-parity-gate",
        reason="sealed candidate projection",
        projection_seal_sha256=SEAL,
        projection_row_count=4,
    ))
    assert candidate["status"] == "candidate"
    assert candidate["projection_seal_sha256"] == SEAL
    assert candidate["projection_row_count"] == 4

    with pytest.raises(ValueError, match="requires projection_seal"):
        eligibility_generation_status_event(
            generation_sha256=row["generation_sha256"],
            generation_sequence=7,
            status="accepted",
            effective_at=NOW + timedelta(minutes=6),
            reviewed_by="staging-parity-gate",
            reason="missing content pin",
        )


def test_evidence_requires_real_uri_hash_timezone_confidence_and_ttl_order():
    canonical = _source(source_uri="HTTPS://Example.COM/Players/Exact?A=B")
    assert canonical.source_uri == "https://Example.COM/Players/Exact?A=B"
    with pytest.raises(ValueError, match="explicit URI"):
        _source(source_uri="atp-profile.html")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _source(source_content_sha256="not-a-hash")
    with pytest.raises(ValueError, match="explicit timezone"):
        _source(observed_at=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        _source(confidence=1.01)
    with pytest.raises(ValueError, match="after observed_at"):
        _source(expires_at=NOW)
    with pytest.raises(ValueError, match="source_artifact_id"):
        _source(source_artifact_id=None)
    with pytest.raises(ValueError, match="initial_review_state"):
        _source(initial_review_state="accepted")


@pytest.mark.parametrize("value", [None, 0, 149, 231, 999, "unknown"])
def test_height_validator_never_fabricates_or_accepts_sentinel_values(value):
    with pytest.raises(ValueError):
        normalize_profile_value("height_cm", value)


def test_profile_and_round_values_are_typed_and_closed_world():
    height = _row(player_profile_observation(
        generation_sha256=GENERATION,
        canonical_player_id=123,
        field_name="height_cm",
        field_value="183.0",
        source=_source(),
    ))
    assert height["height_cm"] == Decimal("183")
    assert height["hand"] is None
    assert "field_value" not in height
    assert normalize_profile_value("hand", "A") == "A"
    with pytest.raises(ValueError, match="L, R, or A"):
        normalize_profile_value("hand", "U")
    with pytest.raises(ValueError, match="birthdate"):
        normalize_profile_value("birthdate", "2026-02-31")
    with pytest.raises(ValueError, match="explicit URI"):
        normalize_profile_value("atp_url", "/players/example")
    assert (
        normalize_profile_value("atp_url", " HTTPS://Example.COM/Player/Exact ")
        == "https://Example.COM/Player/Exact"
    )
    assert normalize_round_code("qf") == "QF"
    assert normalize_round_code("er") == "ER"
    with pytest.raises(ValueError, match="unsupported round_code"):
        normalize_round_code("quarter-final-ish")


def test_compatibility_projection_imports_entities_without_inventing_missing_fields():
    plan = project_compatibility_rows(
        generation_sha256=GENERATION,
        players=[{
            "player_id": 123,
            "name": "Renée Example",
            "hand": "R",
            "height_cm": None,
            "country": "FRA",
            "birthdate": None,
            "ta_slug": None,
            "atp_url": None,
        }],
        aliases=[{"alias": "Renee E.", "player_id": 123}],
        source=_source(
            source_name="public_compatibility_projection",
            source_uri=(
                "compatibility://public.players-player_aliases/"
                "players-20260714.csv"
            ),
            source_artifact_id=None,
            compatibility_import=True,
        ),
    )
    assert plan.row_counts == {
        "ops.player_entities": 1,
        "ops.player_identity_observations": 1,
        "ops.player_alias_observations": 1,
        "ops.player_profile_observations": 2,
    }
    profile_rows = plan.batches[-1].unique_records
    assert {row["field_name"] for row in profile_rows} == {"hand", "country"}
    assert all(row["height_cm"] is None for row in profile_rows)


def test_alias_and_review_records_keep_exact_target_and_do_not_guess():
    alias = _row(player_alias_observation(
        generation_sha256=GENERATION,
        canonical_player_id=123,
        alias="O'Connell, Benjamin",
        source=_source(),
    ))
    assert alias["alias_norm"] == "oconnellbenjamin"
    review = _row(eligibility_review_event(
        generation_sha256=GENERATION,
        target_table="ops.player_alias_observations",
        target_idempotency_key=alias["idempotency_key"],
        review_state="quarantined",
        reviewed_at=NOW + timedelta(minutes=1),
        reviewed_by="identity-reviewer",
        reason="source profile lookup failed; no exact cross-source proof",
    ))
    assert review["review_state"] == "quarantined"


def test_round_builder_requires_complete_durable_provenance_and_ttl():
    identity = match_identity(
        player1="Alpha One", player2="Beta Two",
        match_date=NOW.date(), tournament="Example Open",
    )
    batch = eligibility_match_round_observation(
        identity=identity,
        run_id="00000000-0000-0000-0000-000000000001",
        observed_at=NOW,
        source_name="atp_official",
        round_code="r32",
        eligibility_generation_sha256=GENERATION,
        source_uri="HTTPS://www.atptour.com/en/scores/example/draws",
        source_content_sha256=DIGEST,
        confidence=1,
        initial_review_state="unreviewed",
        expires_at=NOW + timedelta(hours=36),
        source_artifact_id=ARTIFACT_ID,
        source_fetch_id="00000000-0000-0000-0000-000000000098",
    )
    row = _row(batch)
    assert batch.table == "ops.eligibility_match_round_observations"
    assert row["round_code"] == "R32"
    assert row["source_uri"] == "https://www.atptour.com/en/scores/example/draws"
    assert row["expires_at"] == NOW + timedelta(hours=36)

    with pytest.raises(ValueError, match="future expires_at"):
        eligibility_match_round_observation(
            identity=identity,
            run_id="00000000-0000-0000-0000-000000000001",
            observed_at=NOW,
            source_name="atp_official",
            round_code="R32",
            eligibility_generation_sha256=GENERATION,
            source_uri="https://www.atptour.com/en/scores/example/draws",
            source_content_sha256=DIGEST,
            confidence=1,
            initial_review_state="unreviewed",
            source_artifact_id=ARTIFACT_ID,
            source_fetch_id="00000000-0000-0000-0000-000000000098",
            expires_at=NOW,
        )


def test_projection_seal_is_order_independent_and_includes_reviews():
    profile = player_profile_observation(
        generation_sha256=GENERATION,
        canonical_player_id=123,
        field_name="hand",
        field_value="A",
        source=_source(),
    )
    review = eligibility_review_event(
        generation_sha256=GENERATION,
        target_table="ops.player_profile_observations",
        target_idempotency_key=_row(profile)["idempotency_key"],
        review_state="accepted",
        reviewed_at=NOW + timedelta(minutes=1),
        reviewed_by="identity-reviewer",
        reason="official profile evidence verified",
    )
    forward = projection_seal_from_batches(
        (profile, review), generation_sha256=GENERATION,
    )
    reverse = projection_seal_from_batches(
        (review, profile, profile), generation_sha256=GENERATION,
    )
    assert forward == reverse
    assert forward.projection_row_count == 2
    assert len(forward.projection_seal_sha256) == 64


def test_raw_artifact_builder_links_private_body_by_hash_not_inline_payload():
    builder = LiveRecordBuilder(
        external_run_id="run_20260714T210027Z", run_started_at=NOW,
    )
    artifact = _row(builder.source_artifact(
        source_fetch_id="00000000-0000-0000-0000-000000000002",
        artifact_kind="atp_draw_html",
        storage_uri="s3://private-bucket/atp/example.html",
        content_sha256=DIGEST,
        captured_at=NOW,
        content_type="text/html",
        byte_size=1234,
        metadata={"source_uri": "HTTPS://www.atptour.com/Example"},
    ))
    assert artifact["storage_uri"].startswith("s3://")
    assert artifact["content_sha256"] == DIGEST
    assert json.loads(artifact["metadata"])["source_uri"] == (
        "https://www.atptour.com/Example"
    )
    assert "payload" not in artifact


class _Cursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def execute(self, query, params=()):
        self.connection.queries.append((" ".join(query.split()), params))
        self.rows = self.connection.response(query, params)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        return None


class _Connection:
    def __init__(self, response):
        self.response = response
        self.queries = []

    def cursor(self):
        return _Cursor(self)


def test_reader_requires_exact_schema_and_configured_accepted_generation():
    def wrong_schema(query, _params):
        if "ops.schema_versions" in query:
            return [("1.1.0",)]
        return []

    with pytest.raises(EligibilityContractError, match="schema"):
        EligibilityProjectionReader(
            _Connection(wrong_schema), expected_generation_sha256=GENERATION,
            expected_projection_seal_sha256=SEAL,
        ).verify_contract()

    def wrong_generation(query, _params):
        if "ops.schema_versions" in query:
            return [(OPERATIONAL_SCHEMA_VERSION,)]
        if "current_eligibility_generation" in query:
            return [("d" * 64, ELIGIBILITY_CONTRACT_VERSION, SEAL, 4)]
        return []

    with pytest.raises(EligibilityContractError, match="configured generation"):
        EligibilityProjectionReader(
            _Connection(wrong_generation), expected_generation_sha256=GENERATION,
            expected_projection_seal_sha256=SEAL,
        ).verify_contract()


def test_reader_fails_closed_on_conflict_and_returns_one_exact_round():
    def response(query, params):
        if "ops.schema_versions" in query:
            return [(OPERATIONAL_SCHEMA_VERSION,)]
        if "current_eligibility_generation" in query:
            return [(GENERATION, ELIGIBILITY_CONTRACT_VERSION, SEAL, 4)]
        if "api.eligibility_conflicts" in query:
            return [(1,)] if params[1:] == (
                "player_source_identity", "ambiguousname",
            ) else []
        if "current_player_identities" in query:
            return [(123,)]
        if "api.current_match_rounds" in query:
            return [("QF", NOW, NOW + timedelta(hours=12), Decimal("0.98"))]
        return []

    reader = EligibilityProjectionReader(
        _Connection(response), expected_generation_sha256=GENERATION,
        expected_projection_seal_sha256=SEAL,
    )
    identity = reader.resolve_player_id("Ambiguous Name")
    assert identity.status is ResolutionStatus.CONFLICT
    resolved_round = reader.resolve_round("match_anchor_exact")
    assert resolved_round.status is ResolutionStatus.RESOLVED
    assert resolved_round.value == "QF"
    assert resolved_round.confidence == Decimal("0.98")
    assert resolved_round.projection_seal_sha256 == SEAL
    assert reader.projection_row_count == 4


def test_reader_preserves_identity_and_profile_expiry_lineage():
    identity_expiry = NOW + timedelta(hours=6)
    profile_expiry = NOW + timedelta(hours=4)

    def response(query, _params):
        if "ops.schema_versions" in query:
            return [(OPERATIONAL_SCHEMA_VERSION,)]
        if "current_eligibility_generation" in query:
            return [(GENERATION, ELIGIBILITY_CONTRACT_VERSION, SEAL, 4)]
        if "api.eligibility_conflicts" in query:
            return []
        if "api.current_player_name_bindings" in query:
            return [(123, identity_expiry)]
        if "api.current_player_profiles" in query:
            return [("183", NOW, profile_expiry)]
        return []

    reader = EligibilityProjectionReader(
        _Connection(response), expected_generation_sha256=GENERATION,
        expected_projection_seal_sha256=SEAL,
    )
    identity = reader.resolve_player_id("Exact Player")
    assert identity.status is ResolutionStatus.RESOLVED
    assert identity.value == 123
    assert identity.expires_at == identity_expiry
    height = reader.resolve_profile_field(123, "height_cm")
    assert height.status is ResolutionStatus.RESOLVED
    assert height.value == 183.0
    assert height.observed_at == NOW
    assert height.expires_at == profile_expiry


def test_name_normalization_is_exact_and_deterministic():
    assert normalize_identity_name(" Renée O'Connell ") == "reneeoconnell"
    with pytest.raises(ValueError, match="empty"):
        normalize_identity_name("---")
