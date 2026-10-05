"""Goals.md parser."""

from pawn_core.goals import parse_goals, slugify

SAMPLE = """---
pawn: goals
timezone: Europe/Rome
notify:
  max_per_day: 3
  quiet_hours: "21:00-08:00"
ignore:
  - status meetings
---

## Active

### Project X storage
- why: Blocking Friday.
- movement: A chosen option.
- interrupt: A new commitment.

## Parked

### Mobile inbox
- note: [[Ideas/Mobile inbox]]
- do: Link meetings.

## Done recently

### Vault transcripts
- closed: 2026-09-20
"""


def test_slugify():
    assert slugify("Project X storage") == "project-x-storage"


def test_parse_goals_threads_and_policy():
    goals = parse_goals(SAMPLE)
    assert goals.valid
    assert goals.max_per_day == 3
    assert goals.quiet_hours == "21:00-08:00"
    assert goals.timezone == "Europe/Rome"
    assert goals.ignore == ["status meetings"]
    assert [thread.name for thread in goals.active] == ["Project X storage"]
    assert goals.active[0].movement == "A chosen option."
    assert [thread.name for thread in goals.parked] == ["Mobile inbox"]
    assert goals.threads[-1].status == "done"
    assert goals.threads[-1].closed == "2026-09-20"


def test_invalid_pawn_key_is_empty():
    goals = parse_goals("---\npawn: task\n---\n## Active\n### Nope\n")
    assert goals.valid is False
    assert goals.active == []
