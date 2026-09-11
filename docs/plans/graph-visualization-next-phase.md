# Graph Visualization Next Phase

## Summary

Evolve the current minimal graph-run viewer from an inline SVG debug page into
a durable internal inspection tool. Keep the current PostgreSQL event model and
`pawn-server` REST API as the contract, but improve correctness, usability,
retention, and integration readiness.

The next phase should still avoid a full observability stack. The goal is to
make graph execution visualization reliable enough for day-to-day debugging and
structured enough to become a React Flow app, Grafana panel, or OpenTelemetry
bridge later.

## Current Baseline

- Graph events are captured best-effort through `GraphEventRecorder`.
- The recorder writes `run_start`, `node_start`, `node_end`,
  `router_decision`, `edge_taken`, `error`, and `run_end`.
- `agent_runs.id` is reused as the graph execution `run_id`.
- PostgreSQL stores `graph_run_events` and `graph_topologies`.
- `pawn-server` exposes read APIs under `/api/agent-runs`.
- `/graph-viewer` is a single inline HTML page using vanilla JS and SVG.
- Event writes are synchronous and best-effort; failures are logged and
  swallowed.

## Phase Goals

- Make the graph API contract explicit and tested.
- Move the viewer out of inline FastAPI code into a small static asset module.
- Improve path reconstruction so topology edges and execution edges agree.
- Add filtering, retention, and payload-size safety.
- Preserve the backend API shape so future UI/plugin work is low-risk.

## Key Changes

### Backend API Hardening

- Add Pydantic response models for:
  - run summaries
  - run details
  - graph events
  - graph response payloads
- Move raw SQL helpers out of `pawn_server/core/api_server.py` into a focused
  module, for example `pawn_server/core/graph_runs_api.py` or
  `pawn_agent/core/graph_queries.py`.
- Keep endpoints unchanged:
  - `GET /api/agent-runs`
  - `GET /api/agent-runs/{run_id}`
  - `GET /api/agent-runs/{run_id}/events`
  - `GET /api/agent-runs/{run_id}/graph`
- Add v1 filters to `GET /api/agent-runs`:
  - `status`
  - `source`
  - `session_id`
  - `created_after`
  - `created_before`
  - `limit`
  - `offset`
- Return stable error shapes for not-found and invalid query parameters.
- Add tests for auth, pagination bounds, filters, not-found behavior, and graph
  response assembly.

### Event Capture Improvements

- Add a small payload sanitizer before writing event payloads:
  - drop `None` values
  - cap strings to a configurable max length
  - cap serialized payload JSON to a configurable max byte size
  - redact obvious token-like fields such as `api_key`, `token`,
    `authorization`, and `secret`
- Add config fields under a new optional section:

```yaml
graph_events:
  enabled: true
  capture_payloads: true
  max_payload_bytes: 8192
  max_string_length: 2048
```

- Keep `PAWN_GRAPH_EVENTS_ENABLED=0` as an emergency override.
- Record `edge_taken` for deterministic edges as well as conditional edges so
  the event stream fully describes the path without server-side inference.
- Add `event_schema_version` to payloads or as a column only if the JSON shape
  starts to vary. For this phase, prefer a payload field to avoid a migration.
- Keep all event writes best-effort and non-fatal.

### Topology Contract

- Replace the hand-maintained topology constant with a single helper that is
  used by both:
  - graph builder registration
  - topology snapshot persistence
- Keep node IDs identical to LangGraph node names.
- Include edge metadata:
  - `conditional: bool`
  - `label` for router choices where useful
  - `source_kind` and `target_kind` only if the UI needs them
- Add a unit test that compares topology nodes/edges to the graph builder calls.

### Viewer Extraction

- Move the inline HTML into static files:

```text
pawn_server/static/graph_viewer/
  index.html
  graph-viewer.js
  graph-viewer.css
```

- Serve it from `pawn-server` with `StaticFiles` or a small explicit
  `HTMLResponse` loader.
- Keep `/graph-viewer` as the entrypoint.
- Keep the UI dependency-free for this phase unless a React Flow migration is
  explicitly started.
- Improve the vanilla SVG viewer:
  - run search/filter box
  - status/source/session filters
  - selected edge details
  - router-decision badges on router nodes
  - error-first timeline affordance
  - copyable run ID
  - refresh button that preserves the selected run
- Add CSS responsive behavior so the viewer remains usable on narrow laptop
  windows.

### Graph Layout And Path Accuracy

- Replace the current fixed kind-column layout with a simple layered layout:
  - `__start__`
  - linear pre-router nodes
  - router nodes
  - tool/response nodes
  - `__end__`
- Use actual topology edges to render the background graph.
- Use `edge_taken` events to mark executed edges.
- If a run has old events without deterministic `edge_taken`, preserve the
  current server-side fallback path inference.
- Represent repeated visits to the same node in the timeline even if the graph
  node is rendered once.
- Surface aggregate node data:
  - visit count
  - total duration
  - latest status
  - latest error

### Runtime And Retention

- Add retention utilities but do not auto-delete by default.
- Provide a CLI command or helper function for cleanup:

```bash
pawn-server graph-runs prune --older-than-days 30
```

- Pruning should delete old `agent_runs` rows only if that is already
  acceptable for run history. Otherwise, prune only `graph_run_events` and leave
  `agent_runs` intact.
- Add indexes only when query patterns prove they need them. Existing
  `run_id`, `run_id + sequence`, `run_id + timestamp`, `event_type`, and
  `node_name` are enough for phase two.

## Implementation Milestones

### Milestone 1: API And Query Refactor

Objective: make graph-run read behavior testable without growing
`api_server.py`.

Likely files:

- `pawn_server/core/api_server.py`
- `pawn_server/core/graph_runs.py`
- `tests/test_graph_runs_api.py`

Expected output:

- Existing endpoints keep the same URLs and response shape.
- Query logic lives outside `api_server.py`.
- Pydantic response models document the API contract.

Acceptance criteria:

- Tests cover list, detail, events, graph response, and 404 cases.
- `uv run pytest --no-cov tests/test_graph_runs_api.py` passes.
- Existing API tests still pass.

### Milestone 2: Static Viewer Files

Objective: remove the large inline HTML string from FastAPI.

Likely files:

- `pawn_server/static/graph_viewer/index.html`
- `pawn_server/static/graph_viewer/graph-viewer.js`
- `pawn_server/static/graph_viewer/graph-viewer.css`
- `pawn_server/core/api_server.py`

Expected output:

- `/graph-viewer` serves the same viewer from static assets.
- The viewer still loads runs, graph, and events from the same APIs.

Acceptance criteria:

- Manual browser test shows the same graph as before.
- Static files are served behind the same auth behavior as `/graph-viewer`.
- `api_server.py` becomes materially smaller.

### Milestone 3: Event Payload Safety

Objective: prevent accidental large or sensitive event payloads.

Likely files:

- `pawn_agent/core/graph_events.py`
- `pawn_agent/utils/config.py`
- `tests/test_graph_events.py`

Expected output:

- Payloads are capped and redacted before writes.
- Capture can be disabled through config or env.

Acceptance criteria:

- Tests cover redaction, string truncation, payload size capping, and disabled
  capture.
- Event logging failures still do not fail agent runs.

### Milestone 4: Full Edge Capture

Objective: make execution path reconstruction event-driven instead of inferred.

Likely files:

- `pawn_agent/core/graph_events.py`
- `pawn_agent/core/langgraph_chat.py`
- `pawn_server/core/graph_runs.py`
- `tests/test_langgraph_chat.py`
- `tests/test_graph_runs_api.py`

Expected output:

- Deterministic edges emit `edge_taken`.
- Conditional edges continue to emit `router_decision` and `edge_taken`.
- `/api/agent-runs/{run_id}/graph` highlights path from event rows first.

Acceptance criteria:

- A normal run includes path edges from `__start__` through `__end__`.
- Repeated dispatch loops are visible in the timeline.
- Existing graph runs without deterministic edge events still render.

### Milestone 5: Viewer Usability Pass

Objective: make the internal UI comfortable for daily debugging.

Likely files:

- `pawn_server/static/graph_viewer/graph-viewer.js`
- `pawn_server/static/graph_viewer/graph-viewer.css`
- `tests/test_graph_runs_api.py`

Expected output:

- Run filters.
- Better selected-node and selected-edge panels.
- Error-focused timeline.
- Visit counts and total duration per node.

Acceptance criteria:

- A failed run makes the failed node and error event obvious.
- Router decisions are visible without reading raw JSON.
- A run with multiple tool loops is readable.

### Milestone 6: Retention Command

Objective: keep local databases from growing indefinitely.

Likely files:

- `pawn_server/cli/commands.py`
- `pawn_agent/utils/db.py` or a new graph-run service module
- `tests/test_graph_run_retention.py`

Expected output:

- `pawn-server graph-runs prune --older-than-days N` removes old graph events.
- Dry-run mode reports counts without deleting.

Acceptance criteria:

- Tests cover dry run, deletion, and boundary timestamps.
- The command never deletes `pawnai.yaml` or any config/secrets.

## Test Plan

- Unit tests:
  - graph event sanitizer
  - recorder disabled behavior
  - deterministic and conditional edge capture
  - topology helper output
  - graph response assembly from synthetic event rows
- API tests:
  - auth enforcement
  - list pagination
  - list filters
  - run detail
  - events ordering
  - graph response shape
  - 404 responses
- UI smoke test:
  - serve `pawn-server`
  - create one run
  - open `/graph-viewer`
  - verify run list, graph, timeline, and details panel render
- Regression tests:
  - `uv run pytest --no-cov tests/test_graph_events.py tests/test_agent_runner_graph_events.py tests/test_langgraph_chat.py tests/test_agent_queue_listener.py`
  - Add new focused graph API and retention tests as milestones land.

## Non-Goals For This Phase

- Do not add Phoenix, Tempo, Loki, or Grafana as required runtime services.
- Do not replace PostgreSQL as the event store.
- Do not store full LangGraph state snapshots by default.
- Do not build a full observability product.
- Do not add a Node/Vite/React build unless the viewer extraction is already
  complete and the plain SVG UI becomes a clear bottleneck.

## Future Migration Notes

- React Flow migration:
  - keep `/api/agent-runs/{run_id}/graph` as the data contract
  - replace only the static viewer implementation
  - map backend `nodes`, `edges`, `path`, `node_status`, and
    `router_decisions` into React Flow node/edge props
- Grafana panel:
  - reuse the graph response shape as the panel query response
  - keep `run_id`, `thread_id`, `graph_name`, and `graph_version` stable
- OpenTelemetry:
  - map `run_start/run_end` to a root span
  - map `node_start/node_end/error` to child spans
  - map `router_decision` and `edge_taken` to span events
  - store generated `trace_id` in `graph_run_events.trace_id`

## Assumptions

- The current v1 graph event implementation has already been migrated with
  `alembic upgrade head`.
- `agent_runs` remains the run summary source of truth.
- `pawn-server` continues to be the only HTTP surface for the internal viewer.
- Local/internal users are the primary audience for the next phase.
- Best-effort event capture remains preferable to any logging behavior that can
  fail the agent run.
