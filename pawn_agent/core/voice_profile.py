"""Voice-assistant profile for sallm agents (``/v1/pawn/voice``).

Owns: the spoken-reply system prompt, the per-access skill/tool surface, and
the ``to_speech_text`` post-processor.

``access="chat"`` keeps every skill *name* (so a skill stack persisted from an
unlocked turn still resolves) but strips their tools and builds the Agent with
an empty tool registry: nothing is executable. ``access="full"`` is the stock
pawn surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

from sallm import SkillRegistry

from pawn_agent.core.sallm_skills import build_pawn_skills
from pawn_agent.core.sallm_tools import build_pawn_clitools

Access = Literal["chat", "full"]

_LANG_NAMES = {"it": "Italian", "en": "English"}


@dataclass(frozen=True)
class VoiceProfile:
    access: Access = "chat"
    lang: str = "it"

    def fingerprint(self) -> tuple:
        return ("voice", self.access, self.lang)

    def system_extra(self) -> str:
        lang = _LANG_NAMES.get(self.lang, self.lang)
        lines = [
            "VOICE MODE: your answer is spoken aloud by a home voice assistant. Voice comes first: "
            "the user is listening, maybe not looking at a screen.",
            f"- Reply in {lang} unless the user clearly speaks another language.",
            "- Say the answer itself in one or two short sentences (under 40 words), the way a "
            "person would say it. Never use lists, tables, markdown, emoji, code, file paths or URLs "
            "in the spoken part.",
            "- Tool results are for you to read, not to recite: interpret them and say what they "
            "mean. Never read out ids, slugs, segment counts, durations lists, timestamps or raw "
            "numbers; round and summarise (\"about an hour\", \"three sessions this week\").",
            "- After using tools you must always end with a spoken answer, never with tool output.",
            "- Say numbers, dates and times the way a person would say them aloud.",
            "- Ask at most one short question, only if you really need it.",
            "- If there are details worth seeing (names, a list, figures, a longer analysis), you may "
            "add them AFTER a line containing only [[SCREEN]], as short markdown for the phone's "
            "screen. The spoken part before it must still answer on its own; you may say the "
            "details are on the screen. Never put the answer only on the screen.",
        ]
        if self.access == "chat":
            lines.append(
                "- Tools are LOCKED in this conversation. You cannot read sessions, notes, "
                "calendars or any stored data. If the user asks for that, say briefly that "
                "they must unlock full access first (\"sblocca le sessioni\")."
            )
        else:
            lines.append(
                "- Full access is unlocked: you may use tools. Summarise results in one or "
                "two spoken sentences; never read out raw listings."
            )
        lines.append(
            "- The voice device obeys two control tags. End your reply with [[END]] when the "
            "user wants to stop talking, end the conversation or stop you listening. End it "
            "with [[LOCK]] when the user wants to lock, revoke or close full access again. "
            "Never claim you cannot do these: the tag does it. Never use the tags otherwise."
        )
        return "\n".join(lines)

    def skills(self) -> SkillRegistry:
        registry = build_pawn_skills()
        if self.access == "full":
            return registry
        locked = SkillRegistry(
            [replace(registry.get(name), tools=()) for name in registry.names()]
        )
        return locked

    def tools(self) -> list:
        if self.access == "full":
            return build_pawn_clitools()
        return []


def parse_access(raw: str | None) -> Access:
    return "full" if (raw or "").strip().lower() == "full" else "chat"


CONTROLS = ("end", "lock")
_SCREEN_RE = re.compile(r"\[\[\s*SCREEN\s*\]\]", re.I)

# Asked once when a voice turn ended on tool results with no answer to speak.
VOICE_NUDGE = {
    "it": "(Promemoria del dispositivo vocale) Rispondi ora alla mia ultima richiesta in una o due "
    "frasi da dire a voce, interpretando i risultati degli strumenti che hai già ottenuto. Metti "
    "eventuali dettagli dopo [[SCREEN]].",
    "en": "(Voice device reminder) Answer my last request now in one or two sentences to be spoken "
    "aloud, interpreting the tool results you already have. Put any details after [[SCREEN]].",
}
_ON_SCREEN = {"it": "Ti ho messo i dettagli sullo schermo.", "en": "I've put the details on the screen."}


def voice_answer_body(response: str) -> str:
    """The agent's answer without the ``[tool] …`` trail; empty when there is only the trail.

    Unlike ``strip_tool_trail`` this never falls back to the trail: tool output must not be spoken.
    """
    text = (response or "").strip()
    if not text.startswith("[tool]"):
        return text
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("[tool]")).strip()


def split_voice_reply(body: str, lang: str = "it") -> tuple[str, str, list[str]]:
    """Split an answer into (speech, screen markdown, controls).

    The speech is flattened by :func:`to_speech_text`; the screen part is what came after
    ``[[SCREEN]]``. Control tags are honoured wherever they appear.
    """
    m = _SCREEN_RE.search(body or "")
    spoken, screen = (body[: m.start()], body[m.end():]) if m else (body or "", "")
    spoken, c1 = extract_controls(spoken)
    screen, c2 = extract_controls(screen)
    controls = list(dict.fromkeys(c1 + c2))
    speech = to_speech_text(spoken)
    screen = screen.strip()
    if not speech and screen:
        speech = _ON_SCREEN.get(lang, _ON_SCREEN["en"])
    return speech, screen, controls
_CONTROL_RE = re.compile(r"\[\[\s*(END|LOCK)\s*\]\]", re.I)


def extract_controls(text: str) -> tuple[str, list[str]]:
    """Strip ``[[END]]`` / ``[[LOCK]]`` tags; return (clean text, controls in order)."""
    found: list[str] = []
    for m in _CONTROL_RE.finditer(text or ""):
        name = m.group(1).lower()
        if name not in found:
            found.append(name)
    return _CONTROL_RE.sub("", text or "").strip(), found


_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_WIKI_LINK_RE = re.compile(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]")
_URL_RE = re.compile(r"https?://\S+")
_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M)
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*", re.M)
_EMPH_RE = re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1")
_SENT_RE = re.compile(r"(?<=[.!?…])\s+")


def to_speech_text(text: str, *, max_sentences: int = 3, max_chars: int = 400) -> str:
    """Flatten markdown-ish agent output into a short speakable string."""
    s = text or ""
    s = _FENCE_RE.sub(" ", s)
    s = _WIKI_LINK_RE.sub(r"\1", s)
    s = _MD_LINK_RE.sub(r"\1", s)
    s = _URL_RE.sub("", s)
    s = _INLINE_CODE_RE.sub(r"\1", s)
    s = _HEADING_RE.sub("", s)
    s = _BULLET_RE.sub("", s)
    s = _EMPH_RE.sub(r"\2", s)
    s = s.replace("|", " ")
    # Lines without terminal punctuation (ex-list items) become sentences.
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    lines = [ln if ln[-1] in ".!?…:;," else ln + "." for ln in lines]
    s = re.sub(r"\s+", " ", " ".join(lines)).strip()
    sentences = [p for p in _SENT_RE.split(s) if p]
    out = " ".join(sentences[:max_sentences])
    if len(out) > max_chars:
        cut = out[:max_chars].rsplit(" ", 1)[0].rstrip(",;:")
        out = cut + "…"
    return out
