"""Core agent module for pawn-agent.

:class:`ConversationAgent` wraps a DSPy ``ReAct`` module, orchestrating
tool selection and invocation in response to a user prompt.

Usage::

    import dspy
    from pawn_agent.core.lm import build_lm
    from pawn_agent.core.tools import build_tools
    from pawn_agent.core.agent import ConversationAgent
    from pawn_agent.utils.config import load_config

    cfg = load_config()
    lm = build_lm(cfg)
    dspy.configure(lm=lm)
    tools = build_tools(cfg)
    agent = ConversationAgent(tools=tools)
    print(agent.run("Summarise session abc123 and save it to SiYuan"))
"""

from __future__ import annotations

from typing import Callable, List

import dspy


class ConversationAgent(dspy.Module):
    """DSPy agent that selects and invokes tools to fulfil a user prompt.

    Uses ``dspy.ReAct`` with up to *max_iters* thought–action–observation
    cycles.  The agent receives a plain-text prompt and returns a plain-text
    response.

    Args:
        tools: List of callable tools (from :func:`~pawn_agent.core.tools.build_tools`).
        max_iters: Maximum ReAct iterations before returning a final answer.
    """

    def __init__(self, tools: List[Callable], max_iters: int = 6) -> None:
        super().__init__()
        self.react = dspy.ReAct(
            "user_prompt -> response",
            tools=tools,
            max_iters=max_iters,
        )

    def forward(self, user_prompt: str) -> str:
        """Run the agent on a user prompt and return its response.

        Args:
            user_prompt: Free-text instruction from the user.

        Returns:
            Agent's final response string.
        """
        result = self.react(user_prompt=user_prompt)
        return result.response
