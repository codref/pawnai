"""Atomic tool executor functions wrapping pawnai core modules.

Each public function in the sub-modules has the signature::

    async def execute(params: dict, cfg: AgentConfig) -> dict

They are registered in :mod:`pawn_agent.executors.registration` and looked up
by the string ``executor`` key declared in each tool YAML file.
"""
