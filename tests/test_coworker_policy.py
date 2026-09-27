"""Attention policy."""

from datetime import datetime
from zoneinfo import ZoneInfo

from pawn_agent.core.coworker.policy import (
    PolicyContext,
    apply_policy,
    fingerprint,
    in_quiet_hours,
    matches_ignore,
)


def _ctx(**kwargs) -> PolicyContext:
    base = dict(
        now=datetime(2026, 9, 27, 10, 0, tzinfo=ZoneInfo("Europe/Rome")),
        timezone_name="Europe/Rome",
        quiet_hours="21:00-08:00",
        max_per_day=2,
        notified_today=0,
        ignore=["status meetings"],
        seen_fingerprints=set(),
        suppressed_fingerprints=set(),
    )
    base.update(kwargs)
    return PolicyContext(**base)


def test_fingerprint_stable():
    assert fingerprint("  Hello   there ", "X") == fingerprint("hello there", "x")


def test_quiet_hours_wrap_midnight():
    night = datetime(2026, 9, 27, 22, 0, tzinfo=ZoneInfo("Europe/Rome"))
    morning = datetime(2026, 9, 27, 7, 0, tzinfo=ZoneInfo("Europe/Rome"))
    day = datetime(2026, 9, 27, 12, 0, tzinfo=ZoneInfo("Europe/Rome"))
    assert in_quiet_hours(night, "21:00-08:00", "Europe/Rome")
    assert in_quiet_hours(morning, "21:00-08:00", "Europe/Rome")
    assert not in_quiet_hours(day, "21:00-08:00", "Europe/Rome")


def test_ignore_list():
    assert matches_ignore("weekly status meetings notes", ["status meetings"])


def test_policy_allows_interrupt():
    decision = apply_policy(
        text="Ship it Friday",
        thread="Project X",
        interrupt=True,
        fingerprint_value="abc",
        ctx=_ctx(),
    )
    assert decision.interrupt


def test_policy_blocks_cap_quiet_and_seen():
    assert not apply_policy(
        text="Ship it",
        thread="Project X",
        interrupt=True,
        fingerprint_value="abc",
        ctx=_ctx(notified_today=2),
    ).interrupt
    night = datetime(2026, 9, 27, 22, 30, tzinfo=ZoneInfo("Europe/Rome"))
    assert not apply_policy(
        text="Ship it",
        thread="Project X",
        interrupt=True,
        fingerprint_value="abc",
        ctx=_ctx(now=night),
    ).interrupt
    assert not apply_policy(
        text="Ship it",
        thread="Project X",
        interrupt=True,
        fingerprint_value="abc",
        ctx=_ctx(seen_fingerprints={"abc"}),
    ).interrupt
    assert not apply_policy(
        text="status meetings recap",
        thread="Project X",
        interrupt=True,
        fingerprint_value="abc",
        ctx=_ctx(),
    ).interrupt
