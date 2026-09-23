"""Tests for SiYuan @pawn protocol helpers and CliTools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pawn_agent.core.sallm_skills import build_pawn_skills
from pawn_agent.core.sallm_tools import build_pawn_clitools
from pawn_agent.core.siyuan_protocol import (
    approval_checked,
    build_discovery_sql,
    build_result_markdown,
    extract_block_refs,
    instruction_hash,
    is_pawn_mention,
    parse_discovered_rows,
    strip_mention_prefix,
)


def test_is_pawn_mention() -> None:
    assert is_pawn_mention("@pawn do the thing")
    assert is_pawn_mention("  @Pawn follow up")
    assert not is_pawn_mention("talk about @pawn later")
    assert not is_pawn_mention("pawn without at")


def test_instruction_hash_stable() -> None:
    a = instruction_hash("@pawn  Hello   World")
    b = instruction_hash("@pawn hello world")
    assert a == b
    assert len(a) == 32


def test_extract_block_refs() -> None:
    text = 'See ((20260920113000-abc1234 "Aurora")) and ((20260920113000-abc1234))'
    assert extract_block_refs(text) == ["20260920113000-abc1234"]


def test_build_discovery_sql_escapes() -> None:
    stmt = build_discovery_sql(["nb'1"], mention_token="@pawn", watermark="2026", limit=10)
    assert "box IN ('nb''1')" in stmt
    assert "updated > '2026'" in stmt
    assert "LIMIT 10" in stmt


def test_parse_discovered_rows_filters() -> None:
    rows = [
        {
            "id": "b1",
            "parent_id": "p1",
            "root_id": "r1",
            "box": "nb",
            "content": "@pawn analyze",
            "markdown": "@pawn analyze",
            "updated": "1",
        },
        {
            "id": "b2",
            "parent_id": "p2",
            "root_id": "r2",
            "box": "nb",
            "content": "mention @pawn mid",
            "markdown": "mention @pawn mid",
            "updated": "2",
        },
    ]
    found = parse_discovered_rows(rows)
    assert len(found) == 1
    assert found[0].block_id == "b1"


def test_approval_checked() -> None:
    assert approval_checked("- [x] Approve for Pawn memory\n")
    assert approval_checked("* [X] Approve for Pawn memory")
    assert not approval_checked("- [ ] Approve for Pawn memory\n")


def test_build_result_markdown_includes_checklist() -> None:
    md = build_result_markdown("Hello", request_id="req-1")
    assert "Approve for Pawn memory" in md
    assert "req-1" in md
    assert "Hello" in md


def test_strip_mention_prefix() -> None:
    assert strip_mention_prefix("@pawn Do stuff") == "Do stuff"


def test_strip_mention_from_callout_instruction() -> None:
    from pawn_agent.core.siyuan_protocol import strip_mention_from_instruction

    md = "> [!TIP] 🤖 Task\n> @pawn do the thing\n> more lines\n"
    out = strip_mention_from_instruction(md)
    assert "@pawn" not in out.lower()
    assert "do the thing" in out
    assert "more lines" in out


def test_looks_like_tip_callout_and_contains_mention() -> None:
    from pawn_agent.core.siyuan_protocol import (
        contains_mention_token,
        looks_like_tip_callout,
    )

    md = "> [!TIP] 🤖 Hello\n> @pawn body\n"
    assert looks_like_tip_callout(md)
    assert contains_mention_token(md, "@pawn")
    assert not looks_like_tip_callout("@pawn plain")


def test_extract_pawn_prompt_instruction() -> None:
    from pawn_agent.core.siyuan_protocol import extract_pawn_prompt_instruction

    kd = ';;;pawn/prompt\nanalyze the notes\nsecond line\n;;;\n{: id="p1" updated="1"}\n'
    assert extract_pawn_prompt_instruction(kd) == "analyze the notes\nsecond line"
    kept = extract_pawn_prompt_instruction(';;;pawn/prompt\nkeep {: span} here\n;;;\n{: id="p1"}\n')
    assert kept == "keep {: span} here"
    assert extract_pawn_prompt_instruction(";;;pawn/prompt\n;;;\n") == ""
    assert extract_pawn_prompt_instruction("@pawn plain") is None
    assert extract_pawn_prompt_instruction("> [!TIP] 🤖 T\n> @pawn body\n") is None


def test_resolve_pawn_trigger_prefers_prompt_over_callout() -> None:
    from pawn_agent.core.siyuan_protocol import resolve_pawn_trigger

    prompt_kd = ';;;pawn/prompt\nanalyze the notes\n;;;\n{: id="prompt1"}\n'
    callout_kd = "> [!TIP] 🤖 Title\n> @pawn full body\n"

    def query_sql(stmt: str):
        if "prompt1" in stmt:
            return [
                {
                    "id": "prompt1",
                    "parent_id": "callout1",
                    "root_id": "root1",
                    "box": "nb1",
                    "content": "analyze the notes",
                    "markdown": prompt_kd,
                    "updated": "2",
                }
            ]
        if "callout1" in stmt:
            return [
                {
                    "id": "callout1",
                    "parent_id": "parent1",
                    "root_id": "root1",
                    "box": "nb1",
                    "content": "@pawn",
                    "markdown": callout_kd,
                    "updated": "1",
                }
            ]
        return []

    client = MagicMock()
    client.query_sql.side_effect = query_sql
    client.get_block_kramdown.side_effect = lambda bid: (
        prompt_kd if bid == "prompt1" else callout_kd
    )

    resolved = resolve_pawn_trigger(client, "prompt1")
    assert resolved is not None
    assert resolved.trigger_block_id == "prompt1"
    assert resolved.parent_block_id == "callout1"
    assert resolved.instruction_text == "analyze the notes"
    assert ";;;" not in resolved.instruction_text


def test_resolve_pawn_trigger_prefers_callout_ancestor() -> None:
    from pawn_agent.core.siyuan_protocol import resolve_pawn_trigger

    callout_kd = "> [!TIP] 🤖 Title\n> @pawn full body\n> second line\n"
    child_kd = "second line"

    def query_sql(stmt: str):
        if "child1" in stmt:
            return [
                {
                    "id": "child1",
                    "parent_id": "callout1",
                    "root_id": "root1",
                    "box": "nb1",
                    "content": "second line",
                    "markdown": "second line",
                    "updated": "2",
                }
            ]
        if "callout1" in stmt:
            return [
                {
                    "id": "callout1",
                    "parent_id": "parent1",
                    "root_id": "root1",
                    "box": "nb1",
                    "content": "@pawn",
                    "markdown": callout_kd,
                    "updated": "1",
                }
            ]
        return []

    client = MagicMock()
    client.query_sql.side_effect = query_sql
    client.get_block_kramdown.side_effect = lambda bid: (
        child_kd if bid == "child1" else callout_kd
    )

    resolved = resolve_pawn_trigger(client, "child1")
    assert resolved is not None
    assert resolved.trigger_block_id == "callout1"
    assert resolved.parent_block_id == "parent1"
    assert "full body" in resolved.instruction_text
    assert "second line" in resolved.instruction_text


def test_extract_pawn_callout_title() -> None:
    from pawn_agent.core.siyuan_protocol import (
        build_pawn_tip_callout_markdown,
        extract_pawn_callout_title,
    )

    title = extract_pawn_callout_title(
        "@pawn give me followup steps to allow IT to better communicate "
        "-- fetch also the ((abc))"
    )
    assert title.startswith("Give me followup")
    assert "--" not in title
    md = build_pawn_tip_callout_markdown(
        "@pawn give me followup steps",
        title="Add conversation followup",
    )
    assert md.startswith("> [!TIP] 🤖 Add conversation followup\n")
    assert "> @pawn give me followup steps" in md


def test_clitools_include_siyuan_block_tools() -> None:
    tools = build_pawn_clitools()
    assert "siyuan_read" in tools
    assert "siyuan_append" in tools
    assert "siyuan_set_status" in tools


def test_siyuan_tasks_skill_tools() -> None:
    registry = build_pawn_skills()
    assert "siyuan_tasks" in registry.names()
    skill = registry.get("siyuan_tasks")
    assert skill is not None
    assert skill.tools is not None
    assert "siyuan_append" in skill.tools
    assert "session_analyze" in skill.tools


def test_siyuan_append_cli_uses_impl(tmp_path) -> None:
    from pawn_agent.tools.cli import siyuan_append

    note = tmp_path / "note.md"
    note.write_text("body text", encoding="utf-8")
    with (
        patch(
            "pawn_agent.tools.cli.siyuan_append.load_agent_config",
            return_value=MagicMock(),
        ),
        patch(
            "pawn_agent.tools.cli.siyuan_append.siyuan_append_impl",
            return_value="Appended block_id=x",
        ) as impl,
    ):
        code = siyuan_append.main(
            ["--parent-id", "p1", "--content-file", str(note), "--as-result", "--request-id", "r1"]
        )
    assert code == 0
    impl.assert_called_once()
    assert impl.call_args.kwargs["parent_id"] == "p1"
    assert impl.call_args.kwargs["as_result"] is True


def test_siyuan_set_status_impl_rejects_bad_status() -> None:
    from pawn_agent.tools.siyuan_blocks import siyuan_set_status_impl

    out = siyuan_set_status_impl(MagicMock(), block_id="b1", status="explode")
    assert out.startswith("Error:")


def test_siyuan_append_impl_calls_client() -> None:
    from pawn_agent.tools.siyuan_blocks import siyuan_append_impl

    cfg = MagicMock()
    client = MagicMock()
    client.append_block.return_value = "newid"
    with patch(
        "pawn_agent.tools.siyuan_blocks.client_from_agent_config",
        return_value=client,
    ):
        out = siyuan_append_impl(cfg, parent_id="p", content="hi", as_result=True, request_id="r")
    assert "newid" in out
    client.append_block.assert_called_once()
    args = client.append_block.call_args[0]
    assert args[0] == "p"
    assert "Approve for Pawn memory" in args[1]
