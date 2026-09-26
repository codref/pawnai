import type { App } from "obsidian";
import type { AskJobRequest, Job, JobStatus } from "./api";
import { newId } from "./chat/conversations";
import { createTaskNote, listTaskNotes, ParsedTaskNote } from "./tasks";

/** Map a task note to a job so offline / synced tasks show up in the jobs list. */
export function jobFromTaskNote(task: ParsedTaskNote): Job {
  const status: JobStatus = task.approved && task.status === "review" ? "done" : task.status;
  return {
    id: task.id,
    kind: "ask",
    status,
    title: task.instruction.split(/\r?\n/)[0]?.slice(0, 120) || task.id,
    instruction: task.instruction,
    conversation: task.conversation,
    note_path: task.note ?? null,
    task_key: task.path,
    result: task.result || null,
    error_code: null,
    approved: task.approved,
    via: "vault",
    offline: true,
  };
}

/** Server unreachable: write a todo task note; the vault watcher runs it after sync. */
export async function queueOffline(app: App, agentRoot: string, req: AskJobRequest): Promise<Job> {
  const id = req.id ?? newId();
  const contextParts: string[] = [];
  if (req.selection?.trim()) contextParts.push(`Selected text:\n\n${req.selection.trim()}`);
  if (req.context_paths?.length) {
    contextParts.push(
      "Context notes:\n" +
        req.context_paths.map((p) => `- [[${p.replace(/\.md$/i, "")}]]`).join("\n"),
    );
  }
  if (req.context?.trim()) contextParts.push(req.context.trim());
  const path = await createTaskNote(app, agentRoot, {
    id,
    instruction: req.instruction,
    context: contextParts.join("\n\n"),
    notePath: req.note_path,
    conversation: req.conversation,
  });
  return {
    id,
    kind: "ask",
    status: "todo",
    title: req.instruction.split(/\r?\n/)[0]?.slice(0, 120) || id,
    instruction: req.instruction,
    conversation: req.conversation ?? "",
    note_path: req.note_path ?? null,
    task_key: path,
    result: null,
    approved: false,
    via: "vault",
    offline: true,
  };
}

export async function offlineJobs(app: App, agentRoot: string): Promise<Job[]> {
  return (await listTaskNotes(app, agentRoot)).map(jobFromTaskNote);
}
