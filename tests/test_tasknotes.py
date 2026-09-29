"""TaskNotes pick-lists, tasks, updates, and boards."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from pawn_agent.tools import tasknotes_impl
from pawn_agent.tools.cli import tasknotes_commit, tasknotes_propose
from pawn_agent.utils.config import AgentConfig, CoworkerConfig, TaskNotesConfig
from pawn_core.vault import VaultStore, dump_frontmatter, parse_frontmatter
from tests.test_vault_store import FakeS3Client

NOW = datetime(2026, 9, 29, 8, 30, tzinfo=ZoneInfo("Europe/Rome"))

ITEMS = {
    "title": "Launch",
    "boards": "both",
    "items": [
        {
            "id": "1",
            "title": "Send the contract",
            "assignee": "Davide",
            "due": "2026-10-02",
            "scheduled": "2026-09-30T09:00",
            "project": "Website Redesign",
            "details": "Send it before Friday.",
            "priority": "high",
            "source": "session:abc",
            "source_note": "Pawn/Transcripts/2026-09-28 abc.md",
            "blocked_by": ["2"],
        },
        {
            "id": "2",
            "title": "Review the draft",
            "assignee": "me",
            "due": "2026-10-06",
            "details": "Read it Monday.",
        },
    ],
}


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> VaultStore:
    client = FakeS3Client()
    vault = VaultStore(bucket="vault", client=client, agent_root="Pawn")
    monkeypatch.setattr(tasknotes_impl, "vault_store_from_config", lambda _cfg: vault)
    return vault


def _cfg(**task_kwargs: object) -> AgentConfig:
    return AgentConfig(
        tasknotes=TaskNotesConfig(display_name="Ada", me=["Ada"], **task_kwargs),
        coworker=CoworkerConfig(timezone="Europe/Rome", me=["Ada"]),
    )


def _proposal_key(store: VaultStore) -> str:
    keys = store.list("Pawn/TaskNotes/Proposals")
    assert len(keys) == 1
    return keys[0]


def test_propose_writes_a_checklist_and_no_tasks(store: VaultStore) -> None:
    report = tasknotes_impl.tasknotes_propose_impl(
        _cfg(),
        document=json.dumps(ITEMS),
        now=NOW,
    )
    assert "No tasks created yet" in report
    assert "Ada (1)" in report
    assert "Davide (1)" in report
    assert store.list("Pawn/TaskNotes/Tasks") == []
    text = store.read(_proposal_key(store))
    assert "- [x] `1` Send the contract · assignee Davide · due 2026-10-02" in text
    assert "Send it before Friday." in text
    assert "```tasknotes" in text
    meta, _body = parse_frontmatter(text)
    assert meta["proposal_status"] == "open"
    assert meta["boards"] == "both"


def test_commit_respects_an_edited_checklist(store: VaultStore) -> None:
    tasknotes_impl.tasknotes_propose_impl(_cfg(), document=json.dumps(ITEMS), now=NOW)
    key = _proposal_key(store)
    text = store.read(key).replace("- [x] `2`", "- [ ] `2`", 1)
    text = text.replace("due 2026-10-02", "due 2026-10-05", 1)
    store.write(key, text)

    report = tasknotes_impl.tasknotes_commit_impl(_cfg(), proposal=key, now=NOW)
    assert "Created 1." in report
    assert "Left unchecked 1." in report
    tasks = store.list("Pawn/TaskNotes/Tasks")
    assert tasks == ["Pawn/TaskNotes/Tasks/Send the contract.md"]
    meta, body = parse_frontmatter(store.read(tasks[0]))
    assert meta["due"] == "2026-10-05"
    assert meta["scheduled"] == "2026-09-30T09:00"
    assert meta["assignee"] == "Davide"
    assert meta["priority"] == "high"
    assert meta["tags"] == ["task"]
    assert meta["projects"] == ["[[Website Redesign]]"]
    assert "blockedBy" not in meta or meta["blockedBy"] == []
    assert "Source: [[Pawn/Transcripts/2026-09-28 abc]]" in body
    project = store.read("Pawn/TaskNotes/Projects/Website Redesign.md")
    assert "tags:" in project
    boards = store.list("Pawn/TaskNotes/Views", suffix=".base")
    assert "Pawn/TaskNotes/Views/Davide.base" in boards
    assert any(path.endswith("Website Redesign project.base") for path in boards)
    updated, _body = parse_frontmatter(store.read(key))
    assert updated["proposal_status"] == "open"


def test_pick_overrides_unchecked_lines(store: VaultStore) -> None:
    tasknotes_impl.tasknotes_propose_impl(_cfg(), document=json.dumps(ITEMS), now=NOW)
    key = _proposal_key(store)
    store.write(key, store.read(key).replace("- [x]", "- [ ]"))
    report = tasknotes_impl.tasknotes_commit_impl(_cfg(), proposal=key, pick="2", now=NOW)
    assert "Created 1." in report
    tasks = store.list("Pawn/TaskNotes/Tasks")
    assert tasks == ["Pawn/TaskNotes/Tasks/Review the draft.md"]
    meta, _body = parse_frontmatter(store.read(tasks[0]))
    assert meta["assignee"] == "Ada"


def test_second_commit_skips_the_same_title_and_assignee(store: VaultStore) -> None:
    cfg = _cfg()
    document = json.dumps(ITEMS)
    first = tasknotes_impl.tasknotes_commit_impl(cfg, document=document, boards="none", now=NOW)
    assert "Created 2." in first
    second = tasknotes_impl.tasknotes_commit_impl(cfg, document=document, boards="none", now=NOW)
    assert "Created 0." in second
    assert "Skipped 2" in second
    assert len(store.list("Pawn/TaskNotes/Tasks")) == 2
    forced = tasknotes_impl.tasknotes_commit_impl(
        cfg, document=document, boards="none", force=True, now=NOW
    )
    assert "Created 2." in forced
    assert len(store.list("Pawn/TaskNotes/Tasks")) == 4


def test_external_tasknotes_note_is_not_duplicated_or_rewritten(store: VaultStore) -> None:
    existing = dump_frontmatter(
        {
            "tags": ["task"],
            "title": "Send the contract",
            "assignee": "Davide",
            "status": "open",
        },
        "Kept by the plugin.\n",
    )
    store._client.objects["TaskNotes/Tasks/Send the contract.md"] = {  # type: ignore[attr-defined]
        "body": existing.encode("utf-8"),
        "etag": '"ext"',
        "last_modified": datetime.now(timezone.utc),
    }
    document = json.dumps(
        {"items": [{"id": "1", "title": "Send the contract", "assignee": "Davide"}]}
    )
    report = tasknotes_impl.tasknotes_commit_impl(_cfg(), document=document, now=NOW)
    assert "Skipped 1" in report
    assert store.list("Pawn/TaskNotes/Tasks") == []
    listed = tasknotes_impl.tasknotes_list_impl(_cfg())
    assert "read-only" in listed
    update = tasknotes_impl.tasknotes_update_impl(
        _cfg(),
        ident="TaskNotes/Tasks/Send the contract.md",
        status="done",
        now=NOW,
    )
    assert update.startswith("Error:")
    assert "Kept by the plugin." in store.read("TaskNotes/Tasks/Send the contract.md")


def test_update_keeps_the_filename_and_the_note_body(store: VaultStore) -> None:
    tasknotes_impl.tasknotes_commit_impl(
        _cfg(),
        document=json.dumps(ITEMS),
        boards="none",
        now=NOW,
    )
    path = "Pawn/TaskNotes/Tasks/Review the draft.md"
    before, _body = parse_frontmatter(store.read(path))
    report = tasknotes_impl.tasknotes_update_impl(
        _cfg(),
        ident=before["pawn_id"][:8],
        scheduled="2026-10-06T15:30",
        clear_due=True,
        status="done",
        now=NOW,
    )
    assert path in report or "Review the draft" in report
    meta, body = parse_frontmatter(store.read(path))
    assert "due" not in meta
    assert meta["scheduled"] == "2026-10-06T15:30"
    assert meta["status"] == "done"
    assert meta["completedDate"] == "2026-09-29"
    assert "Read it Monday." in body
    assert store.list("Pawn/TaskNotes/Tasks") == [
        "Pawn/TaskNotes/Tasks/Review the draft.md",
        "Pawn/TaskNotes/Tasks/Send the contract.md",
    ]


def test_duplicate_titles_need_a_path(store: VaultStore) -> None:
    document = json.dumps(
        {
            "items": [
                {"id": "1", "title": "Call", "assignee": "Ada"},
                {"id": "2", "title": "Call", "assignee": "Davide"},
            ]
        }
    )
    tasknotes_impl.tasknotes_commit_impl(_cfg(), document=document, now=NOW)
    report = tasknotes_impl.tasknotes_update_impl(_cfg(), ident="Call", status="done", now=NOW)
    assert report.startswith("Error:")
    assert "more than one task" in report


def test_hand_edited_board_is_left_alone(store: VaultStore) -> None:
    first = tasknotes_impl.tasknotes_board_impl(_cfg(), name="Davide", assignee="Davide")
    assert "Davide.base" in first
    key = "Pawn/TaskNotes/Views/Davide.base"
    store.write(key, "filters:\n  and: []\n")
    second = tasknotes_impl.tasknotes_board_impl(_cfg(), name="Davide", assignee="Ada")
    assert "left" in second
    assert store.read(key) == "filters:\n  and: []\n"


def test_generated_board_filters_on_assignee(store: VaultStore) -> None:
    tasknotes_impl.tasknotes_board_impl(_cfg(), name="Davide", assignee="me")
    text = store.read("Pawn/TaskNotes/Views/Davide.base")
    assert text.startswith("# pawn-tasknotes-board\n")
    loaded = __import__("yaml").safe_load("\n".join(text.splitlines()[4:]))
    assert 'file.hasTag("task")' in loaded["filters"]["and"]
    assert 'assignee == "Ada"' in loaded["filters"]["and"]
    kinds = [view["type"] for view in loaded["views"]]
    assert kinds == ["tasknotesKanban", "tasknotesCalendar"]


def test_bad_date_does_not_drop_the_rest_of_the_list(store: VaultStore) -> None:
    document = json.dumps(
        {
            "items": [
                {"id": "1", "title": "Book the room", "due": "Friday"},
                {"id": "2", "title": "Send the map", "assignee": "Davide"},
            ]
        }
    )
    report = tasknotes_impl.tasknotes_commit_impl(_cfg(), document=document, now=NOW)
    assert "Created 1." in report
    assert "Friday" in report or "YYYY-MM-DD" in report
    assert store.list("Pawn/TaskNotes/Tasks") == ["Pawn/TaskNotes/Tasks/Send the map.md"]


def test_paths_outside_the_agent_root_are_refused(store: VaultStore) -> None:
    report = tasknotes_impl.tasknotes_commit_impl(
        _cfg(tasks_dir="TaskNotes/Tasks"),
        document=json.dumps({"items": [{"title": "Nope"}]}),
        now=NOW,
    )
    assert report.startswith("Error:")
    assert "agent root" in report
    assert store.list("TaskNotes/Tasks") == []


def test_dependency_links_the_other_task_in_the_batch(store: VaultStore) -> None:
    tasknotes_impl.tasknotes_commit_impl(
        _cfg(),
        document=json.dumps(ITEMS),
        boards="none",
        now=NOW,
    )
    meta, _body = parse_frontmatter(store.read("Pawn/TaskNotes/Tasks/Send the contract.md"))
    assert meta["blockedBy"] == ["[[Review the draft]]"]


def test_list_hides_done_and_can_show_undated(store: VaultStore) -> None:
    tasknotes_impl.tasknotes_commit_impl(
        _cfg(),
        document=json.dumps(
            {
                "items": [
                    {"id": "1", "title": "Dated", "assignee": "Davide", "due": "2026-10-02"},
                    {"id": "2", "title": "Loose", "assignee": "Davide"},
                    {"id": "3", "title": "Finished", "assignee": "Davide", "status": "done"},
                ]
            }
        ),
        now=NOW,
    )
    listed = tasknotes_impl.tasknotes_list_impl(_cfg(), assignee="Davide")
    assert "Dated" in listed
    assert "Loose" in listed
    assert "Finished" not in listed
    undated = tasknotes_impl.tasknotes_list_impl(_cfg(), undated=True)
    assert "Loose" in undated
    assert "Dated" not in undated


def test_timezone_suffix_on_scheduled_is_rejected(store: VaultStore) -> None:
    report = tasknotes_impl.tasknotes_commit_impl(
        _cfg(),
        document=json.dumps({"items": [{"title": "Standup", "scheduled": "2026-10-01T09:00Z"}]}),
        now=NOW,
    )
    assert report.startswith("Error:") or "Needs a fix" in report
    assert store.list("Pawn/TaskNotes/Tasks") == []


def test_in_batch_duplicate_is_one_note(store: VaultStore) -> None:
    report = tasknotes_impl.tasknotes_commit_impl(
        _cfg(),
        document=json.dumps(
            {
                "items": [
                    {"id": "1", "title": "Send the contract", "assignee": "Davide"},
                    {"id": "2", "title": "Send the contract", "assignee": "Davide"},
                ]
            }
        ),
        now=NOW,
    )
    assert "Created 1." in report
    assert "Skipped 1" in report
    assert len(store.list("Pawn/TaskNotes/Tasks")) == 1


def test_commit_without_a_path_uses_the_latest_open_pick_list(store: VaultStore) -> None:
    tasknotes_impl.tasknotes_propose_impl(_cfg(), document=json.dumps(ITEMS), now=NOW)
    report = tasknotes_impl.tasknotes_commit_impl(_cfg(), boards="none", now=NOW)
    assert "latest open pick-list" in report
    assert "Created 2." in report


def test_cli_commit_reads_an_items_file(store: VaultStore, tmp_path, monkeypatch) -> None:
    path = tmp_path / "items.json"
    path.write_text(json.dumps({"items": [{"title": "File the receipt", "assignee": "Ada"}]}))
    monkeypatch.setattr(tasknotes_commit, "load_agent_config", lambda _path: _cfg())
    code = tasknotes_commit.main(["--items-file", str(path)])
    assert code == 0
    assert store.list("Pawn/TaskNotes/Tasks") == ["Pawn/TaskNotes/Tasks/File the receipt.md"]


def test_cli_propose_help() -> None:
    with pytest.raises(SystemExit) as exc:
        tasknotes_propose.main(["--help"])
    assert exc.value.code == 0
