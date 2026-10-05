# Queue Producer Agent Tool

## Summary

Add a safe v1 queue-publishing tool for pawn-agent using `pawn-queue`. The tool
will publish command-envelope JSON messages to named queue targets configured in
`pawnai.yaml`, and the LangGraph chat planner will be able to route user
requests to it. Add a client-side consumer guide in `docs/`.

## Key Changes

- Add config support in `pawn_agent.utils.config` with a new
  `queue_producers: dict[str, QueueProducerConfig]` section.
- Each producer target has `topic`, `bucket_name`, optional `producer_name`,
  optional `polling`, and optional `concurrency`.
- Continue reading S3 credentials from the top-level `s3:` section, matching
  the existing queue listeners.
- Add a new auto-discovered tool at `pawn_agent/tools/push_queue_message.py`.
- Tool name: `push_queue_message`.
- Tool args: `target: str`, `command: str`, `payload: dict[str, Any]`.
- Publish `{ "command": command, **payload }` to the named target.
- Create the topic idempotently, register a producer, publish, and return a
  short receipt with `target`, `topic`, and `message_id`.
- Reject missing target, unknown target, missing command, non-object payload, or
  payloads that already contain `command`.
- Wire the tool into LangGraph with `tool_push_queue_message` in
  `VALID_ACTIONS`, the planner prompt, dispatch routing, graph nodes, and
  graph edges.
- Add a LangGraph tool adapter in `pawn_agent/core/langgraph_tools.py` that
  calls the shared tool implementation.
- Make the planner use this tool only when the user explicitly asks to enqueue,
  publish, push, schedule, or hand work to a queue.
- Add `docs/QUEUE_PRODUCER_CLIENT_GUIDE.md`.
- Document producer config, tool payload examples, the client consumer contract,
  ack/nack expectations, idempotency notes, and a minimal `pawn-queue` consumer
  example.
- Document the message envelope as `{ "command": "...", ... }` and recommend
  that consumers reject unknown commands to the dead-letter queue.

## Test Plan

- Unit-test config parsing for `queue_producers` with named targets.
- Unit-test `push_queue_message_impl` with a mocked `PawnQueueBuilder`.
- Cover the happy path: topic creation, producer registration, and publishing
  the expected envelope.
- Cover unknown target errors.
- Cover missing S3 config errors.
- Cover rejection when `payload` contains `command`.
- Update LangGraph tests so the graph registers `tool_push_queue_message`.
- Update LangGraph dispatch tests so `tool_push_queue_message` routes correctly.
- Confirm planner valid actions include the new tool.
- Run targeted tests:

```bash
pytest tests/test_langgraph_chat.py tests/test_agent_queue_listener.py tests/test_pawn_agent_cli.py --no-cov
```

- Add and run a focused test module for the queue producer tool.

## Assumptions

- The user’s "langchain agent" means the repo’s active LangGraph chat agent
  path.
- V1 uses named producer targets for guardrails instead of allowing raw
  bucket/topic values from prompts.
- Messages use the command-envelope contract: the tool receives `command`
  separately and merges it into the published payload.
- This change does not add a new consumer implementation in the app; it adds
  documentation for implementing the external/client-side consumer.
