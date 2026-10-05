"""Vault keys recovered from agent tool observations."""

from pawn_agent.core.vault_paths import vault_paths_from_steps


def _step(action: str, observation: str) -> dict:
    return {"kind": "action", "tool_calls": [{"action": action, "observation": observation}]}


def test_writes_appends_updates_and_analysis_save() -> None:
    steps = [
        _step("note_read", "wrote Pawn/Decoy.md (etag=nope) is just note text"),
        _step("note_write", "wrote Pawn/Notes/Title.md (etag=abc)"),
        _step("note_append", "appended Pawn/Today.md (etag=def)"),
        _step("task_update", "updated Pawn/Tasks/job.md (etag=ghi)\nstatus='review'"),
        _step(
            "session_analyze",
            "Saved analysis to vault: Pawn/Analyses/sess.md\n\n# body",
        ),
        _step("note_search", "no notes"),
    ]
    assert vault_paths_from_steps(steps) == [
        "Pawn/Notes/Title.md",
        "Pawn/Today.md",
        "Pawn/Tasks/job.md",
        "Pawn/Analyses/sess.md",
    ]


def test_errors_and_duplicate_paths_are_dropped() -> None:
    steps = [
        _step("note_write", "Error: write denied: outside Pawn/"),
        _step("note_write", "wrote Pawn/Notes/A.md (etag=1)"),
        _step("note_append", "appended Pawn/Notes/A.md (etag=2)"),
        _step("session_analyze", "Error performing structured analysis: boom"),
        {"kind": "message"},
        "not-a-step",
    ]
    assert vault_paths_from_steps(steps) == ["Pawn/Notes/A.md"]


def test_keys_with_spaces() -> None:
    steps = [_step("note_write", "wrote Pawn/Notes/My Note.md (etag=abc)")]
    assert vault_paths_from_steps(steps) == ["Pawn/Notes/My Note.md"]


def test_empty_steps() -> None:
    assert vault_paths_from_steps(None) == []
    assert vault_paths_from_steps([]) == []
