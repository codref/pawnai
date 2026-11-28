"""Board-chat profile for sallm agents (a portal chat docked beside a board).

Same tools and skills as the stock pawn surface (``board_read`` and
``board_propose`` are always registered); the profile only adds a system
fragment that names the board and states the rule: Pawn proposes, the user
decides. Threaded like :class:`VoiceProfile` (the ``voice=`` slot of
``run_agent_turn`` is the per-conversation profile).
"""

from __future__ import annotations

from dataclasses import dataclass

from sallm import SkillRegistry

from pawn_agent.core.sallm_skills import build_pawn_skills
from pawn_agent.core.sallm_tools import build_pawn_clitools


@dataclass(frozen=True)
class BoardProfile:
    board: str

    def fingerprint(self) -> tuple:
        return ("board", self.board)

    def system_extra(self) -> str:
        return "\n".join(
            [
                f"BOARD MODE: this conversation is docked beside the user's corkboard "
                f"`{self.board}`. Each user turn carries a snapshot of the board (cards with "
                "ids like item:…, subject:…, obs:…, note:…, and the strings between them); the "
                "cards the user SELECTED come first and are what the question is about.",
                "- Use the board skill. Refer to cards by their ids and quote their titles.",
                "- You cannot change the board. To suggest a change, call board_propose "
                f"(--board {self.board}): a string between two cards, a note card, a group "
                "of cards under a new thread, or a card to pin. It appears as a ghost the "
                "user accepts or burns. Never claim a change was made; say you proposed it.",
                "- Propose only what the evidence supports, at most a few per turn, each "
                "with a one-line reason. Say plainly when the board does not answer the "
                "question, and what would.",
                "- board_read --node ID shows one card's full evidence; tools like "
                "session_transcript and note_search find what is not on the board.",
            ]
        )

    def skills(self) -> SkillRegistry:
        return build_pawn_skills()

    def tools(self) -> dict:
        return build_pawn_clitools()
