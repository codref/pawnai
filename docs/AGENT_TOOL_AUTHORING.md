# Agent tool authoring

This guide is superseded by [TOOLS.md](TOOLS.md).

Summary: add an `*_impl` module, a `cli/*.py` argparse entrypoint, register a
`CliTool` in `pawn_agent/core/sallm_tools.py`, and optionally a skill in
`sallm_skills.py`. Do not add PydanticAI `build(cfg)` discovery modules.
