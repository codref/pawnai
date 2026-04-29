---
description: "Create or improve a stdix standard YAML file. Use when you want to encode implementation requirements, coding conventions, or project practices as rules that AI coding assistants can follow."
name: "New stdix Standard"
argument-hint: "Describe the standard to create or improve, and include any examples or current rule file."
agent: "agent"
tools: ["create_file", "read_file", "run_in_terminal"]
---

Create or improve a stdix standard YAML file that can be injected into AI coding assistants such as Copilot, Claude, Cursor, and AGENTS.md-style instruction files.

The output must be self-contained: a coding assistant that only reads the generated YAML should understand when the standard applies, which APIs or patterns to use, which behavior to avoid, and how to implement the subject correctly.

## Input

The user will provide:
- A short description of the standard, including the language, framework, library, or workflow.
- Optionally, an existing standard YAML file to improve.
- Optionally, example code, documentation snippets, or conventions to encode.

If no input is attached, ask the user for a description before proceeding.

## File Location

- If the user points to an existing YAML file, edit that file in place.
- If creating a new file, place it in the project's stdix standards directory when one exists.
- Use a lowercase, descriptive filename based on the topic.
- Do not reference repository-specific example paths in the generated prompt or standard unless the user explicitly asks for them.

## Schema

```yaml
id: <language-or-shared>.<topic> # dot-namespaced, lowercase, unique
title: <Human Readable Title>
version: 1.0.0
language: <python|go|node|...>   # omit when language-agnostic
tags:
  - <tag1>
  - <tag2>
applies_when:
  - <specific situation where this standard must be applied>
rules:
  - <actionable implementation rule>
outputs:
  agents: true
  claude: true
  copilot: true
  cursor: true
```

**Required:** `id`, `title`, `version` (semver `x.y.z`), `rules`
**Optional:** `language`, `tags`, `applies_when`, `outputs`

## Writing Guidelines

- Write for an implementation agent, not for a human reader browsing documentation.
- Make `applies_when` precise enough that an assistant can decide when to inject the standard.
- Prefer 12–24 high-signal rules for implementation-heavy standards; use fewer only when the subject is genuinely small.
- Write one concrete behavior per rule.
- Use imperative mood: "Use X", "Validate Y", "Do not Z".
- Name the exact public APIs, configuration keys, lifecycle methods, files, payload fields, or command names the assistant should use.
- Include the important negative constraints, especially APIs or shortcuts the assistant must not use.
- Explain reliability requirements such as validation, idempotency, retries, transaction boundaries, logging, cleanup, concurrency, and shutdown when they matter.
- Include operational requirements such as secrets, environment variables, resource limits, and deployment/scaling patterns when they affect correct implementation.
- Keep rules independent and self-contained; a rule should still make sense when copied into a generated instruction file.
- Avoid vague advice such as "follow best practices", "be robust", or "handle errors properly" unless the rule says exactly how.
- Avoid project-local paths, temporary paths, test fixture paths, and links to example documents in the standard body.
- Quote YAML strings that contain `:` or other syntax that could be parsed incorrectly.

## Deriving Rules

When examples or existing code are provided:

1. Identify the public contract a future assistant must reproduce.
2. Extract concrete lifecycle steps, configuration conventions, data shapes, and forbidden patterns.
3. Convert implicit behavior into explicit rules.
4. Prefer the current public API over older compatibility wrappers.
5. Remove incidental implementation details that are not required in other projects.

When improving an existing standard:

1. Preserve the `id` unless the topic is clearly wrong.
2. Bump the patch or minor version when the rule set materially changes.
3. Replace outdated API names, storage assumptions, and vague instructions.
4. Keep outputs enabled unless the user asks to target only specific assistants.

## Steps

1. Read any existing standard, examples, or nearby code the user provided.
2. Determine the `id`, `title`, `language`, `tags`, and `applies_when` from the subject.
3. Draft rules that are specific enough for an assistant to implement the behavior without additional documentation.
4. Save the YAML in the target file.
5. Validate that the YAML parses.
6. If the project provides a stdix validation or registry rebuild command, run it when available.
7. Report the file changed, the validation performed, and any assumptions or follow-up publishing step the user still needs.
