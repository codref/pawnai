"""pawn-agent — DSPy-powered agentic queue listener for PawnAI.

Consumes natural-language requests from a dedicated pawn-queue topic,
produces an explicit execution plan (plan-then-execute via DSPy ChainOfThought),
and runs each step by invoking loaded skill pipelines composed of atomic tools.

Architecture overview::

    Queue message (free-text request)
        │
        ▼
    DSPy Planner  ←── skill YAML descriptors (name + description)
        │
        ▼ explicit JSON plan (logged before any execution)
        │
        ▼
    Plan Executor
        │   for each plan step:
        ▼
    SkillRunner   ←── skill YAML tool-steps + template expressions
        │
        ▼
    ExecutorRegistry  ←── tool YAML → registered Python callable
        │
        ▼
    pawnai core modules
"""

__version__ = "0.1.0"
