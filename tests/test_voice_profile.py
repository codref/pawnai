"""Voice profile (/v1/pawn/voice): tool gating, speech text, endpoint."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from pawn_agent.core import sallm_session
from pawn_agent.core.sallm_skills import build_pawn_skills
from pawn_agent.core.voice_profile import (
    VoiceProfile,
    extract_controls,
    parse_access,
    to_speech_text,
)
from pawn_server.core import api_server
from tests.test_api_jobs import AUTH, _events, env  # noqa: F401  (fixture re-export)


def test_chat_profile_exposes_no_tools() -> None:
    voice = VoiceProfile(access="chat")
    assert voice.tools() == []
    skills = voice.skills()
    stock = build_pawn_skills()
    # Every skill name stays resolvable (persisted skill stacks must not KeyError).
    assert skills.names() == stock.names()
    fake_tools = {"sessions_list": object(), "session_delete": object()}
    for name in skills.names():
        assert skills.get(name).tools == ()
        assert skills.resolve_tools(name, fake_tools) == {}


def test_full_profile_matches_stock_surface() -> None:
    voice = VoiceProfile(access="full")
    assert voice.skills().names() == build_pawn_skills().names()
    assert voice.skills().get("sessions").tools == build_pawn_skills().get("sessions").tools
    assert len(voice.tools()) > 0


def test_system_prompt_mentions_lock_state() -> None:
    assert "LOCKED" in VoiceProfile(access="chat").system_extra()
    assert "unlocked" in VoiceProfile(access="full").system_extra()


def test_parse_access_defaults_to_chat() -> None:
    assert parse_access(None) == "chat"
    assert parse_access("FULL") == "full"
    assert parse_access("admin") == "chat"


def test_access_change_rebuilds_agent(monkeypatch: Any) -> None:
    built: list[Any] = []

    def fake_rebuild(agent: Any, cfg: Any, voice: Any = None) -> Any:
        built.append(voice)
        return SimpleNamespace(session_id=agent.session_id, trace=None)

    monkeypatch.setattr(sallm_session, "rebuild_agent_for_config", fake_rebuild)
    monkeypatch.setattr(sallm_session, "_agent_fingerprint", lambda cfg, voice=None: (
        voice.fingerprint() if voice is not None else None
    ))
    agent = SimpleNamespace(session_id="voice:x", trace=None)
    chat = VoiceProfile(access="chat")
    session = sallm_session.SallmChatSession(agent, cfg=object(), voice=chat)  # type: ignore[arg-type]
    session.apply_config(object(), chat)  # type: ignore[arg-type]
    assert built == []
    session.apply_config(object(), VoiceProfile(access="full"))  # type: ignore[arg-type]
    assert [v.access for v in built] == ["full"]
    assert session.conversation_id == "voice:x"


def test_to_speech_text_flattens_markdown() -> None:
    raw = (
        "## Sessioni\n\n- **Riunione Acme** del 3 marzo\n- `standup` [link](http://x.y)\n\n"
        "Vuoi il riassunto? Posso anche altro. E altro ancora. E basta."
    )
    out = to_speech_text(raw)
    assert "#" not in out and "*" not in out and "`" not in out and "http" not in out
    assert out.startswith("Sessioni. Riunione Acme del 3 marzo. standup link.")
    assert out.count(".") + out.count("?") <= 4


def test_to_speech_text_caps_length() -> None:
    out = to_speech_text("parola " * 200, max_chars=80)
    assert len(out) <= 81 and out.endswith("…")


def test_voice_endpoint_streams_and_passes_profile(env: SimpleNamespace) -> None:  # noqa: F811
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/pawn/voice",
            headers=AUTH,
            json={"conversation": "voice:abc", "message": "ciao", "access": "full"},
        )
        events = _events(resp.text)
        names = [e[0] for e in events]
        assert "answer" in names and names[-1] == "done"
        answer = dict(events)["answer"]
        assert answer == {"content": "the answer.", "access": "full", "controls": []}
        call = env.prompts[0]
        assert call["session_id"] == "voice:abc"
        assert call["source"] == "voice"
        assert call["voice"].access == "full"


def test_voice_endpoint_rejects_foreign_conversation(env: SimpleNamespace) -> None:  # noqa: F811
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/pawn/voice",
            headers=AUTH,
            json={"conversation": "note:Projects/Roadmap.md", "message": "ciao"},
        )
        assert resp.status_code == 400
        assert env.prompts == []


def test_voice_endpoint_requires_token(env: SimpleNamespace) -> None:  # noqa: F811
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post("/v1/pawn/voice", json={"conversation": "voice:a", "message": "x"})
        assert resp.status_code == 401


def test_extract_controls() -> None:
    assert extract_controls("Ok, a dopo. [[END]]") == ("Ok, a dopo.", ["end"])
    assert extract_controls("Fatto [[ lock ]] e chiudo [[END]] [[END]]") == (
        "Fatto  e chiudo",
        ["lock", "end"],
    )
    assert extract_controls("Niente da fare.") == ("Niente da fare.", [])
    assert "[[END]]" in VoiceProfile().system_extra()


def test_voice_answer_body_never_returns_the_tool_trail() -> None:
    from pawn_agent.core.voice_profile import voice_answer_body

    trail = "[tool] sessions_list → Found 3 session(s): sean-demo-1124 segments: 540"
    assert voice_answer_body(trail) == ""
    assert voice_answer_body(trail + "\n\nHai tre sessioni.") == "Hai tre sessioni."
    assert voice_answer_body("Ciao.") == "Ciao."


def test_split_voice_reply_separates_speech_screen_and_controls() -> None:
    from pawn_agent.core.voice_profile import split_voice_reply

    speech, screen, controls = split_voice_reply(
        "L'ultima sessione è la demo con Sean, circa un'ora. [[SCREEN]]\n- **Durata**: 1h 02m\n- 540 segmenti [[END]]"
    )
    assert speech == "L'ultima sessione è la demo con Sean, circa un'ora."
    assert screen.startswith("- **Durata**") and "[[END]]" not in screen
    assert controls == ["end"]
    # Screen only: the spoken part points there instead of staying silent.
    speech, screen, _ = split_voice_reply("[[SCREEN]]\n| a | b |", "it")
    assert speech == "Ti ho messo i dettagli sullo schermo." and screen == "| a | b |"


def test_voice_prompt_is_voice_first() -> None:
    extra = VoiceProfile(access="full").system_extra()
    assert "[[SCREEN]]" in extra and "not to recite" in extra


def test_voice_endpoint_nudges_once_when_only_tools_ran(
    tmp_path: Any, monkeypatch: Any, env: SimpleNamespace  # noqa: F811
) -> None:
    from pawn_agent.core.agent_runner import AgentRunResult

    replies = iter([
        "[tool] sessions_list → Found 3 session(s): sean-demo-1124 segments: 540",
        "Hai tre sessioni, l'ultima è la demo con Sean. [[SCREEN]]\n- sean-demo-1124: 1h 02m",
    ])
    prompts: list[str] = []

    async def fake_turn(**kwargs: Any) -> AgentRunResult:
        prompts.append(kwargs["prompt"])
        return AgentRunResult(run_id="run-x", response=next(replies))

    monkeypatch.setattr(api_server, "run_agent_turn", fake_turn)
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/pawn/voice",
            headers=AUTH,
            json={"conversation": "voice:n", "message": "analizza l'ultima sessione", "access": "full"},
        )
    answer = dict(_events(resp.text))["answer"]
    assert len(prompts) == 2 and "Promemoria" in prompts[1]
    assert answer["content"] == "Hai tre sessioni, l'ultima è la demo con Sean."
    assert answer["display"] == "- sean-demo-1124: 1h 02m"
    assert "[tool]" not in answer["content"]
