#!/usr/bin/env python3
"""
Comprehensive Bet Tracking and Results Settlement System
Tracks all bets, settles results, and maintains bankroll history
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple
import json

from logging_utils import atomic_write_csv, ensure_csv_columns, normalize_name
from operations.operational_lock import locked_operational_csv
from settlement_attribution import (
    ATTRIBUTION_QUALITY_EXACT_UID,
    ATTRIBUTION_QUALITY_EXACT_UID_UNVERIFIED,
    ATTRIBUTION_QUALITY_ROTATED_UID,
    ATTRIBUTION_QUALITY_UID_UNLINKED,
    ATTRIBUTION_QUALITY_UNVERIFIED,
    SETTLEMENT_QUALITY_CANONICAL_ALIAS,
    SETTLEMENT_QUALITY_COMPATIBILITY_PAIR,
    SETTLEMENT_QUALITY_EXACT_UID,
    SETTLEMENT_QUALITY_UNATTRIBUTED,
    BoundResultEvidence,
    FeatureAttributionEvidence,
    bet_players_match_result,
    feature_evidence_matches_bet,
    normalize_evidence_sha256,
    parse_actual_winner,
    repair_settled_bet_attribution_frame,
    result_evidence_matches_settlement,
)


BETS_COLUMNS = [
    'bet_id', 'session_id', 'timestamp', 'event', 'match', 'match_uid', 'feature_snapshot_id',
    'run_id', 'bet_on', 'bet_on_player1', 'odds_decimal', 'stake', 'stake_fraction',
    'model_prob', 'market_prob', 'edge', 'kelly_fraction', 'potential_profit',
    'potential_loss', 'bankroll_before', 'model_version', 'status', 'outcome',
    'actual_profit', 'bankroll_after', 'settled_timestamp', 'match_date',
    'match_start_time', 'notes', 'settlement_quality', 'attribution_quality',
    'metric_eligible', 'result_evidence_kind', 'result_evidence_sha256'
]

BANKROLL_COLUMNS = [
    'timestamp', 'session_id', 'bankroll', 'change_amount', 'change_reason',
    'account_equity', 'pending_exposure', 'available_bankroll',
    'total_staked', 'num_pending_bets', 'num_settled_bets'
]

SESSION_COLUMNS = [
    'session_id', 'start_time', 'end_time', 'initial_bankroll', 'final_bankroll',
    'total_bets_placed', 'total_staked', 'total_profit_loss', 'win_rate',
    'avg_odds', 'avg_edge', 'kelly_multiplier_used', 'notes'
]

INVALID_RECOMMENDATION_REASON_RANK_IDENTITY_COLLISION = (
    'rank_identity_collision'
)
INVALID_RECOMMENDATION_REASON_CODES = frozenset({
    INVALID_RECOMMENDATION_REASON_RANK_IDENTITY_COLLISION,
})
SETTLEMENT_QUALITY_INVALID_RECOMMENDATION = (
    'administrative_invalid_recommendation_refund'
)
ATTRIBUTION_QUALITY_INVALID_RECOMMENDATION = (
    'bad_input_rank_identity_collision'
)


class BetTracker:
    """Track bets, settle results, and maintain bankroll history"""
    
    def __init__(self, logs_dir: str = "./logs", initial_bankroll: float = 1000.0):
        self.logs_dir = Path(logs_dir)
        self.initial_bankroll = float(initial_bankroll)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.bets_file = self.logs_dir / "all_bets.csv"
        self.bankroll_file = self.logs_dir / "bankroll_history.csv"
        self.session_file = self.logs_dir / "betting_sessions.csv"
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    @locked_operational_csv
    def _initialize_files(self):
        """Initialize tracking files with headers"""
        atomic_write_csv(
            ensure_csv_columns(self.bets_file, BETS_COLUMNS), self.bets_file,
        )
        atomic_write_csv(
            ensure_csv_columns(self.bankroll_file, BANKROLL_COLUMNS),
            self.bankroll_file,
        )
        atomic_write_csv(
            ensure_csv_columns(self.session_file, SESSION_COLUMNS), self.session_file,
        )
    
    @locked_operational_csv
    def start_session(self, initial_bankroll: float, kelly_multiplier: float, notes: str = "") -> str:
        """Start a new betting session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add to sessions file
        session_data = {
            'session_id': session_id,
            'start_time': timestamp,
            'end_time': None,
            'initial_bankroll': initial_bankroll,
            'final_bankroll': None,
            'total_bets_placed': 0,
            'total_staked': 0.0,
            'total_profit_loss': 0.0,
            'win_rate': None,
            'avg_odds': None,
            'avg_edge': None,
            'kelly_multiplier_used': kelly_multiplier,
            'notes': notes
        }
        
        # Append to sessions file
        sessions_df = pd.read_csv(self.session_file)
        sessions_df = pd.concat([sessions_df, pd.DataFrame([session_data])], ignore_index=True)
        sessions_df.to_csv(self.session_file, index=False)
        
        # Log bankroll at session start
        self.log_bankroll_change(session_id, initial_bankroll, 0.0, "Session started")
        
        print(f"🎯 Started betting session: {session_id}")
        print(f"   Initial bankroll: ${initial_bankroll:.2f}")
        print(f"   Kelly multiplier: {kelly_multiplier:.1%}")
        
        return session_id

    @locked_operational_csv
    def discard_empty_session(self, session_id: str) -> bool:
        """Remove a just-created session when dedupe produced zero new bets.

        This keeps sessions and bankroll events aligned with actual paper
        exposure. Refuse to remove anything once a bet references the session.
        """
        bets = ensure_csv_columns(self.bets_file, BETS_COLUMNS)
        if (bets.get('session_id', pd.Series('', index=bets.index)).fillna('').astype(str)
                == str(session_id)).any():
            return False

        sessions = ensure_csv_columns(self.session_file, SESSION_COLUMNS)
        session_mask = (
            sessions.get('session_id', pd.Series('', index=sessions.index))
            .fillna('').astype(str) == str(session_id)
        )
        if not session_mask.any():
            return False
        sessions.loc[~session_mask].to_csv(self.session_file, index=False)

        bankroll = ensure_csv_columns(self.bankroll_file, BANKROLL_COLUMNS)
        bankroll_mask = (
            bankroll.get('session_id', pd.Series('', index=bankroll.index))
            .fillna('').astype(str) == str(session_id)
        )
        bankroll.loc[~bankroll_mask].to_csv(self.bankroll_file, index=False)
        print(f"🧹 Removed empty betting session: {session_id}")
        return True
    
    @locked_operational_csv
    def log_bets(self, bet_slips_df: pd.DataFrame, session_id: str, current_bankroll: float):
        """Log new bets from bet slips"""
        if bet_slips_df.empty:
            return 0

        all_bets_df = ensure_csv_columns(self.bets_file, BETS_COLUMNS)
        pending = all_bets_df[
            all_bets_df.get('status', pd.Series(dtype=str)).fillna('').astype(str).str.lower() == 'pending'
        ].copy()

        def _text(value) -> str:
            if pd.isna(value):
                return ''
            return str(value).strip()

        def _norm(value) -> str:
            return normalize_name(_text(value))

        def _already_pending(record: dict) -> bool:
            if pending.empty:
                return False

            bet_on = _norm(record.get('bet_on', ''))
            match_uid = _text(record.get('match_uid', ''))
            if match_uid:
                uid_mask = pending.get('match_uid', pd.Series('', index=pending.index)).fillna('').astype(str).str.strip() == match_uid
                bet_mask = pending.get('bet_on', pd.Series('', index=pending.index)).apply(_norm) == bet_on
                if (uid_mask & bet_mask).any():
                    return True

            match_label = _norm(record.get('match', ''))
            match_date = _text(record.get('match_date', ''))
            event = _norm(record.get('event', ''))
            match_mask = pending.get('match', pd.Series('', index=pending.index)).apply(_norm) == match_label
            bet_mask = pending.get('bet_on', pd.Series('', index=pending.index)).apply(_norm) == bet_on
            date_mask = pending.get('match_date', pd.Series('', index=pending.index)).fillna('').astype(str).str.strip() == match_date
            # players+side+date is the identity of a bet. Never key on the event
            # label: Bovada's "(N)" suffix mutates every scrape, which let the
            # same match get staked repeatedly across hourly runs.
            return bool((match_mask & bet_mask & date_mask).any())
        
        # Prepare bet records
        bet_records = []
        skipped_duplicates = 0
        for _, bet in bet_slips_df.iterrows():
            bet_id = bet.get('bet_id')
            if pd.isna(bet_id) or not str(bet_id).strip():
                bet_id = f"bet_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(bet_records)}"
            bet_record = {
                'bet_id': bet_id,
                'session_id': session_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'event': bet['event'],
                'match': bet['match'],
                'match_uid': bet.get('match_uid'),
                'feature_snapshot_id': bet.get('feature_snapshot_id'),
                'run_id': bet.get('run_id'),
                'bet_on': bet['bet_on'],
                'bet_on_player1': bet['bet_on_player1'],
                'odds_decimal': bet['odds_decimal'],
                'stake': bet['stake'],
                'stake_fraction': bet['stake_fraction'],
                'model_prob': bet['model_prob'],
                'market_prob': bet['market_prob'],
                'edge': bet['edge'],
                'kelly_fraction': bet['kelly_fraction'],
                'potential_profit': bet['potential_profit'],
                'potential_loss': bet['potential_loss'],
                'bankroll_before': bet.get('bankroll', current_bankroll),
                'model_version': bet.get('model_version', 'NN-143'),
                'status': 'pending',
                'outcome': None,
                'actual_profit': None,
                'bankroll_after': None,
                'settled_timestamp': None,
                'match_date': bet.get('match_date', ''),
                'match_start_time': bet.get('match_start_time', ''),
                'notes': ''
            }
            if _already_pending(bet_record):
                skipped_duplicates += 1
                continue
            bet_records.append(bet_record)

        if not bet_records:
            print(f"📝 No new bets logged ({skipped_duplicates} duplicate pending bet(s) skipped)")
            self._refresh_session_record(session_id)
            return 0

        # Enforce the portfolio invariant at the write boundary as well as in
        # the allocator. This closes the gap for direct callers and fails the
        # whole batch rather than partially recording a differently sized book.
        requested_exposure = sum(float(bet['stake']) for bet in bet_records)
        available_capital = self.get_available_bankroll()
        if (not np.isfinite(requested_exposure) or requested_exposure <= 0
                or requested_exposure > available_capital + 1e-9):
            print(
                f"🛑 Refusing bet batch: requested ${requested_exposure:.2f}, "
                f"only ${available_capital:.2f} unreserved"
            )
            self._refresh_session_record(session_id)
            return 0
        
        # Append to all bets file
        new_bets_df = pd.DataFrame(bet_records)
        all_bets_df = pd.concat([all_bets_df, new_bets_df], ignore_index=True)
        all_bets_df.to_csv(self.bets_file, index=False)
        
        # Placement reserves exposure but does not change account equity.
        total_staked = requested_exposure
        account_equity = self.get_current_bankroll()
        self.log_bankroll_change(
            session_id, account_equity, 0.0,
            f"Reserved ${total_staked:.2f} for {len(bet_records)} pending bets",
        )
        self._refresh_session_record(session_id)
        
        print(f"📝 Logged {len(bet_records)} new bets")
        if skipped_duplicates:
            print(f"   Skipped duplicate pending bets: {skipped_duplicates}")
        print(f"   Total stakes: ${total_staked:.2f}")
        return len(bet_records)
    
    @locked_operational_csv
    def settle_bet(
        self,
        bet_id: str,
        won: bool,
        notes: str = "",
        *,
        settlement_quality: str = SETTLEMENT_QUALITY_UNATTRIBUTED,
        attribution_quality: str = ATTRIBUTION_QUALITY_UNVERIFIED,
        metric_eligible: bool = False,
        result_evidence_kind: str = "",
        result_evidence_sha256: str = "",
        exact_feature_evidence: Mapping[
            str, FeatureAttributionEvidence
        ] | None = None,
        bound_result_evidence: BoundResultEvidence | None = None,
    ) -> float:
        """Settle a specific bet and return profit/loss"""
        all_bets_df = ensure_csv_columns(self.bets_file, BETS_COLUMNS)
        for col in [
            'status', 'outcome', 'settled_timestamp', 'notes',
            'settlement_quality', 'attribution_quality', 'metric_eligible',
            'result_evidence_kind', 'result_evidence_sha256',
        ]:
            if col in all_bets_df.columns:
                all_bets_df[col] = all_bets_df[col].astype(object)
        
        # Find the bet
        bet_idx = all_bets_df[all_bets_df['bet_id'] == bet_id].index
        if len(bet_idx) == 0:
            print(f"⚠️  Bet {bet_id} not found")
            return 0.0
        
        idx = bet_idx[0]
        bet = all_bets_df.loc[idx]
        
        if bet['status'] != 'pending':
            print(f"⚠️  Bet {bet_id} already settled")
            return 0.0
        
        # Calculate actual profit/loss
        if won:
            actual_profit = bet['stake'] * (bet['odds_decimal'] - 1.0)
            outcome = 'win'
        else:
            actual_profit = -bet['stake']
            outcome = 'loss'
        
        # Capture equity before changing the row to settled. Once saved,
        # get_current_bankroll() will already include this result.
        equity_before = self.get_current_bankroll()

        # Update bet record
        all_bets_df.loc[idx, 'status'] = 'settled'
        all_bets_df.loc[idx, 'outcome'] = outcome
        all_bets_df.loc[idx, 'actual_profit'] = actual_profit
        all_bets_df.loc[idx, 'settled_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        all_bets_df.loc[idx, 'notes'] = notes
        def _metadata_text(value, default='') -> str:
            if value is None or pd.isna(value):
                return default
            return str(value).strip() or default

        settlement_value = _metadata_text(
            settlement_quality, SETTLEMENT_QUALITY_UNATTRIBUTED
        )
        attribution_value = _metadata_text(
            attribution_quality, ATTRIBUTION_QUALITY_UNVERIFIED
        )
        if metric_eligible is None or (
            isinstance(metric_eligible, float) and np.isnan(metric_eligible)
        ):
            metric_requested = None
        elif isinstance(metric_eligible, (bool, np.bool_)):
            metric_requested = bool(metric_eligible)
        else:
            metric_text = str(metric_eligible).strip().lower()
            if metric_text in {'true', '1', '1.0', 'yes', 'y'}:
                metric_requested = True
            elif metric_text in {'false', '0', '0.0', 'no', 'n'}:
                metric_requested = False
            else:
                metric_requested = None
        supplied_kind = _metadata_text(result_evidence_kind)
        supplied_hash = normalize_evidence_sha256(result_evidence_sha256)
        bound_valid = isinstance(bound_result_evidence, BoundResultEvidence)
        if bound_valid:
            bound_kind = _metadata_text(bound_result_evidence.kind)
            bound_hash = normalize_evidence_sha256(bound_result_evidence.sha256)
            # A caller may redundantly pass the persisted pair, but it may not
            # contradict the exact in-memory payload used for authorization.
            if (
                (supplied_kind and supplied_kind != bound_kind)
                or (supplied_hash and supplied_hash != bound_hash)
            ):
                bound_valid = False
            else:
                supplied_kind, supplied_hash = bound_kind, bound_hash

        exact_feature_bound = feature_evidence_matches_bet(
            bet,
            bet.get('match_uid'),
            exact_feature_evidence or {},
        )
        bound_valid = bool(
            bound_valid
            and result_evidence_matches_settlement(
                bound_result_evidence,
                match_uid=bound_result_evidence.match_uid,
                p1=bound_result_evidence.p1,
                p2=bound_result_evidence.p2,
                actual_winner=bound_result_evidence.actual_winner,
            )
        )
        result_bound = bool(
            bound_valid
            and _metadata_text(bet.get('match_uid'))
            == _metadata_text(bound_result_evidence.match_uid)
            and bet_players_match_result(
                bet, bound_result_evidence.p1, bound_result_evidence.p2
            )
        )
        if result_bound:
            result_winner = (
                bound_result_evidence.p1
                if bound_result_evidence.actual_winner == 1
                else bound_result_evidence.p2
            )
            result_loser = (
                bound_result_evidence.p2
                if bound_result_evidence.actual_winner == 1
                else bound_result_evidence.p1
            )
            bet_on = normalize_name(_metadata_text(bet.get('bet_on')))
            winner_norm = normalize_name(result_winner)
            loser_norm = normalize_name(result_loser)
            expected_won = (
                True if bet_on == winner_norm
                else False if bet_on == loser_norm
                else None
            )
            result_bound = expected_won is not None and expected_won == bool(won)

        metric_verified = bool(
            metric_requested is True
            and settlement_value == SETTLEMENT_QUALITY_EXACT_UID
            and attribution_value == ATTRIBUTION_QUALITY_EXACT_UID
            and exact_feature_bound
            and result_bound
        )
        if metric_verified:
            metric_value = 'true'
        elif metric_requested is False:
            metric_value = 'false'
        elif settlement_value == SETTLEMENT_QUALITY_EXACT_UID:
            # Direct UID settlement with temporarily incomplete proof remains
            # repairable unknown. Downgrade the attribution in the same row
            # write so an exact label can never survive failed proof.
            metric_value = ''
            attribution_value = ATTRIBUTION_QUALITY_EXACT_UID_UNVERIFIED
        else:
            metric_value = 'false'
        # A malformed or contradictory pair is not durable result evidence.
        if not supplied_kind or not supplied_hash or not bound_valid:
            supplied_kind = ''
            supplied_hash = ''

        all_bets_df.loc[idx, 'settlement_quality'] = settlement_value
        all_bets_df.loc[idx, 'attribution_quality'] = attribution_value
        all_bets_df.loc[idx, 'metric_eligible'] = metric_value
        all_bets_df.loc[idx, 'result_evidence_kind'] = supplied_kind
        all_bets_df.loc[idx, 'result_evidence_sha256'] = supplied_hash
        
        # Calculate new bankroll
        new_bankroll = equity_before + actual_profit
        all_bets_df.loc[idx, 'bankroll_after'] = new_bankroll
        
        # Save updated bets
        all_bets_df.to_csv(self.bets_file, index=False)
        
        # Log bankroll change
        session_id = bet['session_id']
        result_desc = f"Settled bet: {bet['match']} ({'WIN' if won else 'LOSS'})"
        self.log_bankroll_change(session_id, new_bankroll, actual_profit, result_desc)
        self._refresh_session_record(session_id)
        
        print(f"✅ Settled bet {bet_id}: {outcome.upper()}")
        print(f"   Profit/Loss: ${actual_profit:+.2f}")
        print(f"   New bankroll: ${new_bankroll:.2f}")
        
        return actual_profit

    @locked_operational_csv
    def void_invalid_recommendation(
        self,
        bet_id: str,
        *,
        reason_code: str,
        expected_match_uid: str,
        expected_feature_snapshot_id: str,
        expected_run_id: str,
        detail: str = "",
    ) -> bool:
        """Refund one bad-input paper recommendation without asserting a result.

        This is an administrative exposure correction, not match settlement.
        It is deliberately limited to reviewed rank-identity collisions and
        requires the immutable bet lineage as a three-field precondition.  An
        exact replay is idempotent; any competing terminal state fails closed.

        Returns ``True`` when a pending row is newly voided and ``False`` for an
        already-complete exact replay.
        """
        normalized_reason = str(reason_code or '').strip().lower()
        if normalized_reason not in INVALID_RECOMMENDATION_REASON_CODES:
            raise ValueError(
                f"unsupported invalid-recommendation reason: "
                f"{normalized_reason or '<blank>'}"
            )

        expected = {
            'match_uid': str(expected_match_uid or '').strip(),
            'feature_snapshot_id': str(expected_feature_snapshot_id or '').strip(),
            'run_id': str(expected_run_id or '').strip(),
        }
        if any(not value for value in expected.values()):
            raise ValueError(
                'invalid-recommendation refund requires match, feature, and run lineage'
            )

        def _clean(value) -> str:
            if value is None or pd.isna(value):
                return ''
            return str(value).strip()

        normalized_detail = ' '.join(str(detail or '').split())
        note = (
            'Administrative invalid-recommendation refund; '
            'underlying match result not asserted; '
            f'reason_code={normalized_reason}'
        )
        if normalized_detail:
            note += f'; detail={normalized_detail}'
        audit_reason = (
            f'Administrative invalid-recommendation refund {bet_id}; '
            f'reason_code={normalized_reason}'
        )

        bets = ensure_csv_columns(self.bets_file, BETS_COLUMNS)
        matches = bets.index[
            bets['bet_id'].fillna('').astype(str).str.strip().eq(str(bet_id).strip())
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"invalid-recommendation refund requires exactly one bet row: "
                f"{bet_id} ({len(matches)} found)"
            )
        idx = matches[0]
        row = bets.loc[idx]
        session_id = _clean(row.get('session_id'))
        if not session_id:
            raise RuntimeError(
                f"invalid-recommendation refund requires a session: {bet_id}"
            )
        for field, expected_value in expected.items():
            actual = _clean(row.get(field))
            if actual != expected_value:
                raise RuntimeError(
                    f"invalid-recommendation refund lineage mismatch for "
                    f"{bet_id}:{field}; expected {expected_value}, found "
                    f"{actual or '<blank>'}"
                )

        bankroll = ensure_csv_columns(self.bankroll_file, BANKROLL_COLUMNS)
        audit_mask = (
            bankroll['change_reason'].fillna('').astype(str).str.strip()
            == audit_reason
        )
        audit_count = int(audit_mask.sum())
        if audit_count > 1:
            raise RuntimeError(
                f"duplicate invalid-recommendation refund audit for {bet_id}"
            )

        status = _clean(row.get('status')).lower()
        if status != 'pending':
            metric = _clean(row.get('metric_eligible')).lower()
            profit = pd.to_numeric(
                pd.Series([row.get('actual_profit')]), errors='coerce'
            ).iloc[0]
            exact_replay = bool(
                status == 'void'
                and _clean(row.get('outcome')).lower() == 'void'
                and pd.notna(profit) and float(profit) == 0.0
                and _clean(row.get('notes')) == note
                and _clean(row.get('settlement_quality'))
                == SETTLEMENT_QUALITY_INVALID_RECOMMENDATION
                and _clean(row.get('attribution_quality'))
                == ATTRIBUTION_QUALITY_INVALID_RECOMMENDATION
                and metric in {'false', '0', '0.0', 'no', 'n'}
                and not _clean(row.get('result_evidence_kind'))
                and not _clean(row.get('result_evidence_sha256'))
            )
            if not exact_replay:
                raise RuntimeError(
                    f"bet {bet_id} already has a competing terminal state: "
                    f"{status or '<blank>'}"
                )
            # Repair a crash between the bet-row write and its zero-P&L audit.
            if audit_count == 0:
                equity = self.get_current_bankroll()
                self.log_bankroll_change(
                    session_id,
                    equity,
                    0.0,
                    audit_reason,
                )
            self._refresh_session_record(session_id)
            return False

        if audit_count:
            raise RuntimeError(
                f"pending bet {bet_id} already has a terminal refund audit"
            )
        pending_terminal_fields = (
            'outcome', 'actual_profit', 'bankroll_after', 'settled_timestamp',
            'settlement_quality', 'attribution_quality', 'metric_eligible',
            'result_evidence_kind', 'result_evidence_sha256',
        )
        dirty_fields = [
            field for field in pending_terminal_fields if _clean(row.get(field))
        ]
        if dirty_fields:
            raise RuntimeError(
                f"pending bet {bet_id} carries terminal state: "
                f"{','.join(dirty_fields)}"
            )

        equity_before = self.get_current_bankroll()
        object_columns = (
            'status', 'outcome', 'actual_profit', 'bankroll_after',
            'settled_timestamp', 'notes', 'settlement_quality',
            'attribution_quality', 'metric_eligible', 'result_evidence_kind',
            'result_evidence_sha256',
        )
        for column in object_columns:
            bets[column] = bets[column].astype(object)
        bets.at[idx, 'status'] = 'void'
        bets.at[idx, 'outcome'] = 'void'
        bets.at[idx, 'actual_profit'] = 0.0
        bets.at[idx, 'bankroll_after'] = equity_before
        bets.at[idx, 'settled_timestamp'] = datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S'
        )
        bets.at[idx, 'notes'] = note
        bets.at[idx, 'settlement_quality'] = (
            SETTLEMENT_QUALITY_INVALID_RECOMMENDATION
        )
        bets.at[idx, 'attribution_quality'] = (
            ATTRIBUTION_QUALITY_INVALID_RECOMMENDATION
        )
        bets.at[idx, 'metric_eligible'] = 'false'
        bets.at[idx, 'result_evidence_kind'] = ''
        bets.at[idx, 'result_evidence_sha256'] = ''
        atomic_write_csv(bets, self.bets_file)

        self.log_bankroll_change(
            session_id,
            equity_before,
            0.0,
            audit_reason,
        )
        self._refresh_session_record(session_id)
        print(f"↩️  Refunded invalid recommendation {bet_id}: {normalized_reason}")
        print("   Underlying match result was not asserted")
        return True
    
    @locked_operational_csv
    def settle_bets_batch(self, results: List[Dict]) -> float:
        """
        Settle multiple bets at once
        results: [{'bet_id': 'bet_123', 'won': True, 'notes': '...'}]
        """
        total_profit = 0.0
        for result in results:
            profit = self.settle_bet(
                result['bet_id'],
                result['won'],
                result.get('notes', ''),
                settlement_quality=result.get(
                    'settlement_quality', SETTLEMENT_QUALITY_UNATTRIBUTED
                ),
                attribution_quality=result.get(
                    'attribution_quality', ATTRIBUTION_QUALITY_UNVERIFIED
                ),
                metric_eligible=result.get('metric_eligible', False),
                result_evidence_kind=result.get('result_evidence_kind', ''),
                result_evidence_sha256=result.get(
                    'result_evidence_sha256', ''
                ),
                exact_feature_evidence=result.get(
                    'exact_feature_evidence', {}
                ),
                bound_result_evidence=result.get('bound_result_evidence'),
            )
            total_profit += profit
        
        return total_profit
    
    @locked_operational_csv
    def log_bankroll_change(self, session_id: str, new_bankroll: float, change_amount: float, reason: str):
        """Log a bankroll change"""
        bankroll_df = pd.read_csv(self.bankroll_file)
        
        # Account fields are global across sessions; sessions are reporting
        # slices, not independent bankroll resets.
        all_bets_df = ensure_csv_columns(self.bets_file, BETS_COLUMNS)
        status = all_bets_df['status'].fillna('').astype(str).str.lower()
        pending = all_bets_df[status == 'pending']
        settled = all_bets_df[status == 'settled']
        pending_exposure = pd.to_numeric(pending['stake'], errors='coerce').fillna(0).sum()
        account_equity = self.get_current_bankroll()
        available = max(0.0, account_equity - float(pending_exposure))
        num_pending = len(pending)
        num_settled = len(settled)
        total_staked = pd.to_numeric(all_bets_df['stake'], errors='coerce').fillna(0).sum()
        
        bankroll_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': session_id,
            'bankroll': account_equity,
            'change_amount': change_amount,
            'change_reason': reason,
            'account_equity': account_equity,
            'pending_exposure': pending_exposure,
            'available_bankroll': available,
            'total_staked': total_staked,
            'num_pending_bets': num_pending,
            'num_settled_bets': num_settled
        }
        
        bankroll_df = pd.concat([bankroll_df, pd.DataFrame([bankroll_record])], ignore_index=True)
        atomic_write_csv(bankroll_df, self.bankroll_file)
    
    @locked_operational_csv
    def get_current_bankroll(self) -> float:
        """Return account equity from initial capital plus settled net P&L."""
        bets = ensure_csv_columns(self.bets_file, BETS_COLUMNS)
        if bets.empty:
            return self.initial_bankroll
        status = bets['status'].fillna('').astype(str).str.lower()
        settled_profit = pd.to_numeric(
            bets.loc[status == 'settled', 'actual_profit'], errors='coerce'
        ).fillna(0).sum()
        return self.initial_bankroll + float(settled_profit)

    @locked_operational_csv
    def get_pending_exposure(self) -> float:
        pending = self.get_pending_bets()
        if pending.empty:
            return 0.0
        return float(pd.to_numeric(pending['stake'], errors='coerce').fillna(0).sum())

    @locked_operational_csv
    def get_available_bankroll(self) -> float:
        return max(0.0, self.get_current_bankroll() - self.get_pending_exposure())
    
    @locked_operational_csv
    def get_pending_bets(self) -> pd.DataFrame:
        """Get all pending bets"""
        all_bets_df = ensure_csv_columns(self.bets_file, BETS_COLUMNS)
        status = all_bets_df['status'].fillna('').astype(str).str.strip().str.lower()
        return all_bets_df[status == 'pending']
    
    @locked_operational_csv
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary statistics for a session"""
        all_bets_df = pd.read_csv(self.bets_file)
        session_bets = all_bets_df[all_bets_df['session_id'] == session_id]
        
        if session_bets.empty:
            return {'error': 'No bets found for session'}
        
        settled_bets = session_bets[session_bets['status'] == 'settled']
        pending_bets = session_bets[session_bets['status'] == 'pending']
        
        summary = {
            'session_id': session_id,
            'total_bets': len(session_bets),
            'pending_bets': len(pending_bets),
            'settled_bets': len(settled_bets),
            'total_staked': session_bets['stake'].sum(),
            'total_profit_loss': settled_bets['actual_profit'].sum() if not settled_bets.empty else 0.0,
            'win_rate': (settled_bets['outcome'] == 'win').mean() if not settled_bets.empty else None,
            'avg_odds': session_bets['odds_decimal'].mean(),
            'avg_edge': session_bets['edge'].mean(),
            'roi': None
        }
        
        # Calculate ROI
        if summary['total_staked'] > 0 and not settled_bets.empty:
            summary['roi'] = (summary['total_profit_loss'] / summary['total_staked']) * 100
        
        return summary

    @locked_operational_csv
    def _refresh_session_record(self, session_id: str):
        """Synchronize cached session summary columns with the underlying bets file."""
        summary = self.get_session_summary(session_id)
        if 'error' in summary:
            return

        sessions_df = pd.read_csv(self.session_file)
        if 'end_time' in sessions_df.columns:
            sessions_df['end_time'] = sessions_df['end_time'].astype(object)
        session_idx = sessions_df[sessions_df['session_id'] == session_id].index
        if len(session_idx) == 0:
            return

        idx = session_idx[0]
        sessions_df.loc[idx, 'total_bets_placed'] = summary['total_bets']
        sessions_df.loc[idx, 'total_staked'] = summary['total_staked']
        sessions_df.loc[idx, 'total_profit_loss'] = summary['total_profit_loss']
        sessions_df.loc[idx, 'win_rate'] = summary['win_rate']
        sessions_df.loc[idx, 'avg_odds'] = summary['avg_odds']
        sessions_df.loc[idx, 'avg_edge'] = summary['avg_edge']

        if summary['pending_bets'] == 0 and summary['settled_bets'] > 0:
            sessions_df.loc[idx, 'end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sessions_df.loc[idx, 'final_bankroll'] = self.get_current_bankroll()

        atomic_write_csv(sessions_df, self.session_file)

    @locked_operational_csv
    def settle_pending_bets_for_match(
        self,
        match_uid: str = None,
        alias_match_uids=None,
        p1: str = None,
        p2: str = None,
        actual_winner: int = None,
        notes: str = "",
        exact_feature_evidence: Mapping[
            str, FeatureAttributionEvidence
        ] | None = None,
        result_evidence_kind: str = "",
        result_evidence_sha256: str = "",
        bound_result_evidence: BoundResultEvidence | None = None,
    ) -> int:
        """Auto-settle linked bets and classify attribution independently.

        A direct UID is metric-eligible only when the caller supplied verified
        persisted feature evidence and a canonical result payload whose digest,
        UID, ordered players, and winner revalidate at the write boundary. The
        bet itself must match the feature snapshot, run, player pair, and side.
        Aliases and the legacy blank-UID pair fallback remain accounting-only.
        """
        strict_winner = parse_actual_winner(actual_winner)
        if strict_winner is None:
            return 0

        pending = self.get_pending_bets()
        if pending.empty:
            return 0

        def _clean(value) -> str:
            if value is None or pd.isna(value):
                return ''
            return str(value).strip()

        candidates = pd.DataFrame()
        direct_uid = _clean(match_uid)
        alias_uids = {
            _clean(value)
            for value in (alias_match_uids or [])
            if _clean(value)
        }
        alias_uids.discard(direct_uid)
        allowed_uids = {direct_uid, *alias_uids} - {''}
        if match_uid and 'match_uid' in pending.columns:
            candidates = pending[
                pending['match_uid'].fillna('').astype(str).str.strip().isin(allowed_uids)
            ]

        if (
            candidates.empty
            and not direct_uid
            and p1
            and p2
            and 'match_uid' in pending.columns
        ):
            p1_norm = normalize_name(p1)
            p2_norm = normalize_name(p2)

            def _match_pair(match_text: str) -> bool:
                parts = [normalize_name(part) for part in str(match_text).split(' vs ')]
                if len(parts) != 2:
                    return False
                return parts == [p1_norm, p2_norm] or parts == [p2_norm, p1_norm]

            # Pair-only matching is a legacy blank-to-blank compatibility
            # contract. A result with a UID or a bet with a UID must never enter
            # this path, even when the participant names happen to match.
            pair_pool = pending[
                pending['match_uid'].fillna('').astype(str).str.strip().eq('')
            ]
            candidates = pair_pool[pair_pool['match'].apply(_match_pair)]

        settled = 0
        auto_note = notes or "Auto-settled from Tennis Abstract"

        winner_name = p1 if strict_winner == 1 else p2
        loser_name = p2 if strict_winner == 1 else p1
        winner_norm = normalize_name(_clean(winner_name))
        loser_norm = normalize_name(_clean(loser_name))
        if not winner_norm or not loser_norm or winner_norm == loser_norm:
            print(
                "⚠️  Automatic settlement requires two distinct result "
                "participants; leaving linked bets pending"
            )
            return 0

        for _, bet in candidates.iterrows():
            bet_uid = _clean(bet.get('match_uid'))
            bet_on_norm = normalize_name(_clean(bet.get('bet_on')))
            # Winner identity is stable across P1/P2 orientation changes.
            # Require the complete pair too; one overlapping player is not
            # enough to bind a result to a paper exposure.
            if (
                not bet_players_match_result(bet, p1, p2)
                or bet_on_norm not in {winner_norm, loser_norm}
            ):
                print(
                    f"⚠️  Bet {bet['bet_id']} players do not match result "
                    "participants; leaving pending"
                )
                continue
            won = bet_on_norm == winner_norm

            if direct_uid and bet_uid == direct_uid:
                settlement_quality = SETTLEMENT_QUALITY_EXACT_UID
                feature_attribution_verified = bool(
                    winner_norm
                    and loser_norm
                    and feature_evidence_matches_bet(
                        bet,
                        direct_uid,
                        exact_feature_evidence or {},
                    )
                )
                result_attribution_verified = (
                    result_evidence_matches_settlement(
                        bound_result_evidence,
                        match_uid=direct_uid,
                        p1=p1,
                        p2=p2,
                        actual_winner=strict_winner,
                    )
                )
                metric_eligible = bool(
                    feature_attribution_verified
                    and result_attribution_verified
                )
                attribution_quality = (
                    ATTRIBUTION_QUALITY_EXACT_UID
                    if metric_eligible
                    else ATTRIBUTION_QUALITY_EXACT_UID_UNVERIFIED
                )
                if not metric_eligible:
                    metric_eligible = None
            elif bet_uid and bet_uid in alias_uids:
                settlement_quality = SETTLEMENT_QUALITY_CANONICAL_ALIAS
                attribution_quality = ATTRIBUTION_QUALITY_ROTATED_UID
                metric_eligible = False
            else:
                settlement_quality = SETTLEMENT_QUALITY_COMPATIBILITY_PAIR
                attribution_quality = ATTRIBUTION_QUALITY_UID_UNLINKED
                metric_eligible = False

            self.settle_bet(
                bet['bet_id'],
                won=won,
                notes=auto_note,
                settlement_quality=settlement_quality,
                attribution_quality=attribution_quality,
                metric_eligible=metric_eligible,
                result_evidence_kind=result_evidence_kind,
                result_evidence_sha256=result_evidence_sha256,
                exact_feature_evidence=exact_feature_evidence,
                bound_result_evidence=bound_result_evidence,
            )
            settled += 1

        return settled

    @locked_operational_csv
    def repair_settled_bet_attribution(
        self,
        predictions: pd.DataFrame,
        feature_evidence: Mapping[str, FeatureAttributionEvidence],
    ) -> int:
        """Repair only wholly blank settled rows with complete exact proof."""
        bets = ensure_csv_columns(self.bets_file, BETS_COLUMNS)
        repaired_bets, repaired = repair_settled_bet_attribution_frame(
            bets, predictions, feature_evidence
        )
        if repaired:
            repaired_bets.reindex(columns=BETS_COLUMNS).to_csv(
                self.bets_file, index=False
            )
        return repaired
    
    @locked_operational_csv
    def end_session(self, session_id: str) -> Dict:
        """End a betting session and update summary"""
        self._refresh_session_record(session_id)
        summary = self.get_session_summary(session_id)
        
        # Update sessions file
        sessions_df = pd.read_csv(self.session_file)
        session_idx = sessions_df[sessions_df['session_id'] == session_id].index
        
        if len(session_idx) > 0:
            idx = session_idx[0]
            sessions_df.loc[idx, 'end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sessions_df.loc[idx, 'final_bankroll'] = self.get_current_bankroll()
            sessions_df.loc[idx, 'total_bets_placed'] = summary['total_bets']
            sessions_df.loc[idx, 'total_staked'] = summary['total_staked']
            sessions_df.loc[idx, 'total_profit_loss'] = summary['total_profit_loss']
            sessions_df.loc[idx, 'win_rate'] = summary['win_rate']
            sessions_df.loc[idx, 'avg_odds'] = summary['avg_odds']
            sessions_df.loc[idx, 'avg_edge'] = summary['avg_edge']
            sessions_df.to_csv(self.session_file, index=False)
        
        print(f"🏁 Ended session {session_id}")
        print(f"   Final P&L: ${summary['total_profit_loss']:+.2f}")
        print(f"   Win rate: {summary['win_rate']:.1%}" if summary['win_rate'] is not None else "   Win rate: N/A")
        print(f"   ROI: {summary['roi']:+.1f}%" if summary['roi'] is not None else "   ROI: N/A")
        
        return summary

def main():
    """Test the bet tracker"""
    tracker = BetTracker()
    
    # Start a test session
    session_id = tracker.start_session(1000.0, 0.18, "Test session")
    
    # Create sample bet slips
    sample_bets = pd.DataFrame([
        {
            'event': 'ATP Test Tournament',
            'match': 'Player A vs Player B',
            'bet_on': 'Player A',
            'bet_on_player1': True,
            'odds_decimal': 2.0,
            'stake': 50.0,
            'stake_fraction': 0.05,
            'model_prob': 0.55,
            'market_prob': 0.5,
            'edge': 0.05,
            'kelly_fraction': 0.1,
            'potential_profit': 50.0,
            'potential_loss': 50.0,
            'bankroll': 1000.0,
            'model_version': 'NN-143'
        }
    ])
    
    # Log bets
    tracker.log_bets(sample_bets, session_id, 1000.0)
    
    # Get pending bets
    pending = tracker.get_pending_bets()
    print(f"\n📋 Pending bets: {len(pending)}")
    
    # Simulate settling a bet (win)
    if not pending.empty:
        bet_id = pending.iloc[0]['bet_id']
        tracker.settle_bet(bet_id, won=True, notes="Simulated win")
    
    # Get session summary
    summary = tracker.get_session_summary(session_id)
    print(f"\n📊 Session summary: {summary}")
    
    # End session
    tracker.end_session(session_id)

if __name__ == "__main__":
    main()
