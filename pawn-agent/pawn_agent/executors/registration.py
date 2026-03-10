"""Boot-time registration of all tool executor callables.

Call :func:`build_executor_registry` once during startup to get a fully
populated :class:`~pawn_agent.skills.executor_registry.ExecutorRegistry`.

Adding a new tool executor requires:
1. Create a module under ``pawn_agent/executors/`` with an async ``execute``-style function.
2. Add a ``registry.register("your.key", module.function)`` line below.
3. Create a matching ``tools/your_tool.yaml`` with ``function: your.key``.
"""

from __future__ import annotations

from pawn_agent.skills.executor_registry import ExecutorRegistry

from . import analysis, combined, diarization, s3, siyuan, transcription


def build_executor_registry() -> ExecutorRegistry:
    """Create and return an :class:`ExecutorRegistry` with all built-in executors.

    Returns:
        A fully populated registry ready for use by :class:`~pawn_agent.skills.runner.SkillRunner`.
    """
    registry = ExecutorRegistry()

    # S3 / file operations
    registry.register("s3.download", s3.download)
    registry.register("s3.cleanup", s3.cleanup)

    # Audio processing
    registry.register("transcription.run", transcription.run)
    registry.register("diarization.run", diarization.run)
    registry.register("combined.run", combined.run)

    # Analysis
    registry.register("analysis.run", analysis.run)

    # Integrations
    registry.register("siyuan.sync", siyuan.sync)

    return registry
