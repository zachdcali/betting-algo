#!/usr/bin/env python3
"""Tests for dynamic round-day offset heuristic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timedelta
from round_offsets import get_round_day_offset, infer_draw_size


def _check(label, tourney_level, draw_size, start_date, expected_finals_dow=6):
    """Verify all rounds for a tournament, printing inferred dates."""
    dt = datetime.strptime(start_date, '%Y-%m-%d')
    rounds = ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']
    print(f"\n--- {label} ({tourney_level}, draw={draw_size}, start={start_date} {dt.strftime('%a')}) ---")
    for rnd in rounds:
        off = get_round_day_offset(tourney_level, draw_size, rnd, dt)
        inferred = dt + timedelta(days=off)
        print(f"  {rnd:>4}: +{off:>3}d -> {inferred.strftime('%a %b %d')}")
    # Finals should be on expected day (Sunday=6 by default)
    f_off = get_round_day_offset(tourney_level, draw_size, 'F', dt)
    finals_date = dt + timedelta(days=f_off)
    assert finals_date.weekday() == expected_finals_dow, \
        f"Finals for {label} on {finals_date.strftime('%A')}, expected Sunday"
    return f_off


def test_indian_wells_2026():
    """IW 2026: Wed March 4, M, draw 96 -> finals Sun March 15"""
    dt = datetime(2026, 3, 4)
    f_off = _check("Indian Wells 2026", 'M', 96, '2026-03-04')
    assert f_off == 11, f"Expected final_offset=11, got {f_off}"
    finals = dt + timedelta(days=f_off)
    assert finals == datetime(2026, 3, 15)
    # Check specific rounds
    assert get_round_day_offset('M', 96, 'R128', dt) == 0   # Wed Mar 4
    assert get_round_day_offset('M', 96, 'R64', dt) == 2    # Fri Mar 6
    assert get_round_day_offset('M', 96, 'R32', dt) == 4    # Sun Mar 8
    assert get_round_day_offset('M', 96, 'R16', dt) == 6    # Tue Mar 10
    assert get_round_day_offset('M', 96, 'QF', dt) == 7     # Wed Mar 11
    assert get_round_day_offset('M', 96, 'SF', dt) == 9     # Fri Mar 13
    print("  ✓ Indian Wells 2026 passed")


def test_miami_2026():
    """Miami 2026: Wed March 18, M, draw 96 -> finals Sun March 29"""
    dt = datetime(2026, 3, 18)
    f_off = _check("Miami 2026", 'M', 96, '2026-03-18')
    assert f_off == 11
    finals = dt + timedelta(days=f_off)
    assert finals == datetime(2026, 3, 29)
    print("  ✓ Miami 2026 passed")


def test_murcia_challenger():
    """Murcia CH 2026: Mon March 16, C, draw 32 -> finals Sun March 22"""
    dt = datetime(2026, 3, 16)
    f_off = _check("Murcia CH 2026", 'C', 32, '2026-03-16')
    assert f_off == 6
    finals = dt + timedelta(days=f_off)
    assert finals == datetime(2026, 3, 22)
    print("  ✓ Murcia CH passed")


def test_australian_open_2026():
    """AO 2026: Mon Jan 19, G, draw 128 -> finals Sun Feb 1"""
    dt = datetime(2026, 1, 19)
    f_off = _check("Australian Open 2026", 'G', 128, '2026-01-19')
    assert f_off == 13
    finals = dt + timedelta(days=f_off)
    assert finals == datetime(2026, 2, 1)
    # Grand Slam qualifier offsets
    assert get_round_day_offset('G', 128, 'Q1', dt) == -7
    assert get_round_day_offset('G', 128, 'Q2', dt) == -5
    assert get_round_day_offset('G', 128, 'Q3', dt) == -3
    print("  ✓ Australian Open 2026 passed")


def test_masters_64_monday():
    """M/64 Monday start (Canada/Cincinnati style, 1-week): finals on Sunday (+6)"""
    dt = datetime(2026, 8, 3)  # Monday
    f_off = _check("M/64 Mon start (Canada)", 'M', 64, '2026-08-03')
    assert f_off == 6, f"M/64 Monday start: expected F=+6 (Sunday), got +{f_off}"
    finals = dt + timedelta(days=f_off)
    assert finals.weekday() == 6  # Sunday
    print("  ✓ M/64 Monday start (1-week) passed")


def test_masters_64_nonmonday():
    """M/64 non-Monday start (Madrid/Rome style, 2-week): finals on SECOND Sunday"""
    # Madrid 2024: Tue Apr 23 -> Sun May 5 (12 days)
    dt = datetime(2024, 4, 23)  # Tuesday
    f_off = _check("Madrid 2024 (M/64 Tue)", 'M', 64, '2024-04-23', expected_finals_dow=6)
    assert f_off == 12, f"Madrid M/64 Tue start: expected F=+12 (2nd Sunday), got +{f_off}"
    finals = dt + timedelta(days=f_off)
    assert finals == datetime(2024, 5, 5)
    # Check intermediate rounds
    assert get_round_day_offset('M', 64, 'R64', dt) == 0   # Tue Apr 23
    assert get_round_day_offset('M', 64, 'R32', dt) == 2   # Thu Apr 25
    assert get_round_day_offset('M', 64, 'QF', dt) == 8    # Wed May 1
    assert get_round_day_offset('M', 64, 'SF', dt) == 10   # Fri May 3
    print("  ✓ M/64 non-Monday (Madrid 2-week) passed")


def test_fallback_no_date():
    """No tourney_date: defaults to Monday assumption"""
    off_with = get_round_day_offset('C', 32, 'F', datetime(2026, 3, 16))  # Monday
    off_without = get_round_day_offset('C', 32, 'F', None)
    assert off_with == off_without == 6
    print("\n  ✓ Fallback (no date) matches Monday assumption")


def test_sunday_start():
    """Sunday start: final_offset=7 for 1-week (next Sunday)"""
    dt = datetime(2026, 3, 22)  # Sunday
    assert dt.weekday() == 6
    f_off = get_round_day_offset('A', 32, 'F', dt)
    assert f_off == 7, f"Sunday start 1-week: expected F=+7, got +{f_off}"
    finals = dt + timedelta(days=f_off)
    assert finals.weekday() == 6  # Next Sunday
    print("\n  ✓ Sunday start edge case passed (final_offset=7)")


def test_tour_finals():
    """Tour Finals (O): static offsets"""
    dt = datetime(2026, 11, 15)
    assert get_round_day_offset('O', 8, 'RR', dt) == 1
    assert get_round_day_offset('O', 8, 'SF', dt) == 6
    assert get_round_day_offset('O', 8, 'F', dt) == 7
    print("\n  ✓ Tour Finals static offsets passed")


def test_davis_cup():
    """Davis Cup (D): static offsets"""
    dt = datetime(2026, 9, 12)
    assert get_round_day_offset('D', 16, 'RR', dt) == 0
    assert get_round_day_offset('D', 16, 'SF', dt) == 0
    assert get_round_day_offset('D', 16, 'F', dt) == 2
    print("\n  ✓ Davis Cup static offsets passed")


def test_dynamic_qualifier_rounds():
    """Qualifier offsets adapt to actual number of qualifying rounds"""
    dt = datetime(2026, 3, 16)  # Monday

    # 1 round of qualifying: Q1 is day -1
    assert get_round_day_offset('C', 32, 'Q1', dt, num_qual_rounds=1) == -1

    # 2 rounds: Q1=-2, Q2=-1
    assert get_round_day_offset('C', 32, 'Q1', dt, num_qual_rounds=2) == -2
    assert get_round_day_offset('C', 32, 'Q2', dt, num_qual_rounds=2) == -1

    # 3 rounds: Q1=-3, Q2=-2, Q3=-1
    assert get_round_day_offset('C', 32, 'Q1', dt, num_qual_rounds=3) == -3
    assert get_round_day_offset('C', 32, 'Q2', dt, num_qual_rounds=3) == -2
    assert get_round_day_offset('C', 32, 'Q3', dt, num_qual_rounds=3) == -1

    # Grand Slam 3 rounds
    ao = datetime(2026, 1, 19)
    assert get_round_day_offset('G', 128, 'Q1', ao, num_qual_rounds=3) == -3
    assert get_round_day_offset('G', 128, 'Q2', ao, num_qual_rounds=3) == -2
    assert get_round_day_offset('G', 128, 'Q3', ao, num_qual_rounds=3) == -1

    # Zadar/Murcia 2026: only 1 qual round, Q1 was 1 day before main draw
    assert get_round_day_offset('C', 32, 'Q1', dt, num_qual_rounds=1) == -1

    print("\n  ✓ Dynamic qualifier rounds passed")


def test_zadar_challenger_real():
    """Zadar CH 2026: verified against real dates from user"""
    dt = datetime(2026, 3, 16)  # Monday (main draw start)

    # Real dates: Q1=Mar 15, R32=Mar 16-17, R16=Mar 18-19, QF=Mar 20, SF=Mar 21, F=Mar 22
    assert get_round_day_offset('C', 32, 'Q1', dt, num_qual_rounds=1) == -1  # Mar 15 Sat
    q1_date = dt + timedelta(days=-1)
    assert q1_date == datetime(2026, 3, 15)

    assert get_round_day_offset('C', 32, 'R32', dt) == 0   # Mar 16 Mon
    assert get_round_day_offset('C', 32, 'R16', dt) == 2   # Mar 18 Wed
    assert get_round_day_offset('C', 32, 'QF', dt) == 4    # Mar 20 Fri
    assert get_round_day_offset('C', 32, 'SF', dt) == 5    # Mar 21 Sat
    assert get_round_day_offset('C', 32, 'F', dt) == 6     # Mar 22 Sun

    print("\n  ✓ Zadar CH (verified vs real dates) passed")


def test_infer_draw_size():
    """Draw size inference from event name + level"""
    assert infer_draw_size('Australian Open', 'G') == 128
    assert infer_draw_size('BNP Paribas Open - Indian Wells', 'M') == 96
    assert infer_draw_size('Miami Open', 'M') == 96
    assert infer_draw_size('Mutua Madrid Open', 'M') == 64
    assert infer_draw_size('Internazionali BNL dItalia - Roma', 'M') == 64
    assert infer_draw_size('Monte Carlo Masters', 'M') == 48
    assert infer_draw_size('Paris Masters', 'M') == 48
    assert infer_draw_size('Cincinnati Open', 'M') == 64
    assert infer_draw_size('Some ATP 250', 'A') == 32
    assert infer_draw_size('Some Challenger', 'C') == 32
    print("\n  ✓ Draw size inference passed")


def test_masters_small_monday():
    """M small (48-draw) Monday start: finals on Sunday"""
    dt = datetime(2026, 4, 13)  # Monday
    f_off = get_round_day_offset('M', 48, 'F', dt)
    assert f_off == 6
    finals = dt + timedelta(days=f_off)
    assert finals.weekday() == 6
    # Check round offsets make sense
    r64_off = get_round_day_offset('M', 48, 'R64', dt)
    assert r64_off == 0  # R64 on day 0 (Monday)
    print("\n  ✓ M small Monday start passed")


if __name__ == '__main__':
    test_indian_wells_2026()
    test_miami_2026()
    test_murcia_challenger()
    test_australian_open_2026()
    test_masters_64_monday()
    test_masters_64_nonmonday()
    test_fallback_no_date()
    test_sunday_start()
    test_tour_finals()
    test_davis_cup()
    test_dynamic_qualifier_rounds()
    test_zadar_challenger_real()
    test_infer_draw_size()
    test_masters_small_monday()
    print("\n\n✅ All round offset tests passed!")
