"""Versioned prompts for the observation passes.

Bump a version whenever its prompt text changes: observations record
``extractor = "{type}.v{n}"`` so later label metrics compare like with like.
"""

from __future__ import annotations

from dataclasses import dataclass

TRIAGE_VERSION = 1

SYSTEM = (
    "You read meeting transcripts and notes for a manager. Reply with JSON "
    "only. Never invent people, dates, quotes, or facts."
)

TRIAGE_PROMPT = """\
Score how likely the SOURCE below contains each kind of statement.
Return JSON with a number from 0 to 1 for every key:
{{{keys}}}

Kinds:
{definitions}

Small talk, greetings and tool chatter score 0.
{part}
SOURCE:
{source}
"""


@dataclass(frozen=True)
class TypeSpec:
    name: str
    version: int
    definition: str
    fields: dict[str, str]
    rules: tuple[str, ...] = ()

    @property
    def extractor(self) -> str:
        return f"{self.name}.v{self.version}"


SPECS: dict[str, TypeSpec] = {
    spec.name: spec
    for spec in (
        TypeSpec(
            "decision",
            2,
            "a choice between options that was made or agreed, not one merely "
            "discussed: a direction, a priority, an owner, a design.",
            {
                "choice": "what was chosen, one sentence",
                "rationale": "why, from the source, or empty",
                "assumptions": "list of things that must stay true for this to "
                "be right, stated or clearly implied in the source",
                "owner": "who carries it out, or empty",
                "alternatives": "list of options rejected, or empty list",
            },
            (
                "A proposal nobody agreed to is not a decision.",
                "One person saying they will do something ('I'll share the doc') "
                "is a commitment, not a decision. Skip it here.",
                "Assumptions must come from the source; do not speculate.",
            ),
        ),
        TypeSpec(
            "commitment",
            1,
            "someone saying they will do something, or being asked and agreeing.",
            {
                "debtor": "who will do it",
                "creditor": "who it is for, or empty",
                "what": "the deliverable, one sentence",
                "due": "YYYY-MM-DD when a date is said or clearly implied, else empty",
            },
            ("Vague intentions ('we should look at') are not commitments.",),
        ),
        TypeSpec(
            "question",
            3,
            "a question someone actually asks, or an explicitly open unknown "
            "('we don't know yet whether…'), that matters to the work and is left "
            "unresolved in the source.",
            {
                "question": "the question, as a question",
                "asked_by": "who raised it, or empty",
                "blocking": "true when progress waits on the answer",
            },
            (
                "Skip questions answered later in the source.",
                "Skip clarifying back-and-forth between participants ('what is the "
                "timeline?', 'do you deploy everything?') that gets answered in the "
                "next exchanges. Keep questions still open at the end that matter "
                "beyond this conversation.",
                "Opinions, criticism and worries are not questions, even when "
                "they imply one. Skip them here.",
                "text must itself be phrased as a question ending with '?'.",
            ),
        ),
        TypeSpec(
            "block",
            2,
            "a specific piece of work that is stopped or waiting right now, "
            "and the concrete thing it waits on (a decision, a person, access, "
            "a missing skill, another team's deliverable).",
            {
                "blocked": "the specific work that is stopped, named",
                "by": "the concrete thing it waits on",
                "owner": "who can unblock it, or empty",
            },
            (
                "Both 'blocked' and 'by' must be concrete and stated in the source; "
                "if you cannot name the stopped work, it is not a block.",
                "General risks, complaints, frustrations, weaknesses of a tool or "
                "model, and organizational problems ('politics beat evidence', "
                "'too many side initiatives') are concerns, not blocks. Skip them.",
                "Something that might cause trouble later is a concern, not a block.",
            ),
        ),
        TypeSpec(
            "contradiction",
            1,
            "two statements in the source that cannot both be true.",
            {
                "claim_a": "first statement",
                "claim_b": "conflicting statement",
            },
            ("Disagreement between people counts when it is about facts.",),
        ),
        TypeSpec(
            "prediction",
            1,
            "a forecast about the future that can later be checked.",
            {
                "claim": "what is predicted",
                "who": "who predicted it",
                "by_when": "YYYY-MM-DD or a phrase such as 'end of Q1', or empty",
                "confidence_words": "hedging words used ('pretty sure'), or empty",
            },
            ("Plans and commitments are not predictions.",),
        ),
        TypeSpec(
            "concern",
            1,
            "a worry, frustration or risk someone raises.",
            {
                "who": "who raised it",
                "about": "what it is about",
                "intensity": "low | medium | high",
            },
        ),
        TypeSpec(
            "fact",
            1,
            "a concrete status, number, date or state of affairs worth remembering.",
            {
                "about": "the subject",
                "value": "the fact",
            },
            ("Skip facts that only matter inside this meeting.",),
        ),
    )
}

TYPED_PROMPT = """\
Find every {name} in the SOURCE below.
A {name} is {definition}

Return JSON:
{{"observations": [{{
  "text": "one-line claim a stranger understands",
  "context": "2-3 self-contained sentences: who, about what, what is at stake",
  "speaker": "who said it, or empty",
  "quote": "verbatim passage copied from the source that supports it",
  "confidence": 0.0,
{fields}
}}]}}

Rules:
- Only what the source says. If unknown, leave the field empty.
- quote must be copied exactly from the source. Drop the observation
  rather than invent or paraphrase a quote.
- confidence is your probability (0-1) that this really is a {name}.
- Prefer fewer, richer observations. Use an empty list when none qualify.
{rules}{examples}{part}
SOURCE:
{source}
"""


def triage_prompt(types: list[str], source: str, part: str) -> str:
    keys = ", ".join(f'"{t}": 0.0' for t in types)
    definitions = "\n".join(f"- {t}: {SPECS[t].definition}" for t in types)
    return TRIAGE_PROMPT.format(keys=keys, definitions=definitions, part=part, source=source)


def examples_block(name: str, examples: dict[str, list[dict[str, str]]] | None) -> str:
    """The user's own past verdicts on this kind, nearest to the window."""
    if not examples or not (examples.get("accept") or examples.get("reject")):
        return ""
    lines = [
        "",
        f"How this user judged past {name} observations (calibrate to them; "
        "they are not from this SOURCE):",
    ]
    if examples.get("accept"):
        lines.append("Wanted:")
        lines.extend(f'- "{e["text"]}"' for e in examples["accept"])
    if examples.get("reject"):
        lines.append("Not wanted:")
        lines.extend(f'- "{e["text"]}"' for e in examples["reject"])
    return "\n".join(lines) + "\n"


def typed_prompt(
    name: str,
    source: str,
    part: str,
    examples: dict[str, list[dict[str, str]]] | None = None,
) -> str:
    spec = SPECS[name]
    fields = ",\n".join(f'  "{key}": "{desc}"' for key, desc in spec.fields.items())
    rules = "".join(f"- {rule}\n" for rule in spec.rules)
    return TYPED_PROMPT.format(
        name=spec.name,
        definition=spec.definition,
        fields=fields,
        rules=rules,
        examples=examples_block(name, examples),
        part=part,
        source=source,
    )


def part_note(index: int, total: int, annotations: str) -> str:
    """Context block placed before SOURCE: window position and annotations."""
    lines: list[str] = []
    if total > 1:
        lines.append(
            f"This SOURCE is part {index + 1} of {total} of a longer source, with "
            "a small overlap with its neighbours. Use only what this part says, "
            "and do not mention parts."
        )
    if annotations:
        lines.append(
            "User annotations about the whole source (background, not "
            f"observations):\n{annotations}"
        )
    if not lines:
        return ""
    return "\n" + "\n\n".join(lines) + "\n"
