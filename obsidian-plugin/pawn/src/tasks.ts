import type { App, Editor, TFile } from "obsidian";
import type { PawnSettings } from "./settings";

export interface TaskMeta {
  id: string;
  status: string;
  approved: boolean;
  note: string;
  conversation: string;
  path: string;
  instruction: string;
  result: string;
}

function yamlEscape(value: string): string {
  if (/[:#{}[\],&*?|>!%@`]/.test(value) || value.includes("\n")) {
    return JSON.stringify(value);
  }
  return value;
}

export function taskPathForId(agentRoot: string, id: string): string {
  return `${agentRoot.replace(/\/+$/, "")}/Tasks/${id}.md`;
}

export function conversationForNote(notePath: string): string {
  return `note:${notePath.replace(/^\/+/, "")}`;
}

export function renderTaskNote(opts: {
  id: string;
  status: string;
  instruction: string;
  context?: string;
  result?: string;
  notePath?: string;
  conversation?: string;
  approved?: boolean;
}): string {
  const lines = [
    "---",
    "pawn: task",
    `id: ${opts.id}`,
    `status: ${opts.status}`,
    `approved: ${opts.approved ? "true" : "false"}`,
  ];
  if (opts.conversation) {
    lines.push(`conversation: ${yamlEscape(opts.conversation)}`);
  }
  if (opts.notePath) {
    const wiki = opts.notePath.replace(/\.md$/i, "");
    lines.push(`note: "[[${wiki}]]"`);
  }
  lines.push("---", "", "## Instruction", opts.instruction.trim(), "");
  if (opts.context?.trim()) {
    lines.push("## Context", opts.context.trim(), "");
  }
  if (opts.result?.trim()) {
    lines.push("## Result", opts.result.trim(), "");
  }
  return lines.join("\n");
}

function getSection(body: string, name: string): string {
  const re = new RegExp(
    `^##\\s+${name}\\s*\\n([\\s\\S]*?)(?=^##\\s+|$)`,
    "m",
  );
  const m = body.match(re);
  return m ? m[1].trim() : "";
}

export function parseTaskFile(path: string, text: string): TaskMeta | null {
  if (!text.startsWith("---")) return null;
  const end = text.indexOf("\n---", 3);
  if (end < 0) return null;
  const fm = text.slice(3, end).trim();
  const body = text.slice(end + 4);
  if (!/^pawn:\s*task\b/m.test(fm)) return null;
  const id = (fm.match(/^id:\s*(.+)$/m)?.[1] || "").trim();
  const status = (fm.match(/^status:\s*(.+)$/m)?.[1] || "todo").trim();
  const approved = /^approved:\s*(true|yes|1)\b/im.test(fm);
  const conversation = (fm.match(/^conversation:\s*(.+)$/m)?.[1] || "")
    .trim()
    .replace(/^["']|["']$/g, "");
  let note = (fm.match(/^note:\s*(.+)$/m)?.[1] || "").trim();
  note = note.replace(/^["']|["']$/g, "");
  const wiki = note.match(/^\[\[([^\]|]+)/);
  if (wiki) note = wiki[1].trim();
  return {
    id,
    status,
    approved,
    note,
    conversation,
    path,
    instruction: getSection(body, "Instruction"),
    result: getSection(body, "Result"),
  };
}

export async function createTaskNote(
  app: App,
  settings: PawnSettings,
  opts: {
    id: string;
    instruction: string;
    notePath: string;
    context?: string;
    conversation?: string;
  },
): Promise<string> {
  const path = taskPathForId(settings.agentRoot, opts.id);
  const folder = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "";
  if (folder && !app.vault.getAbstractFileByPath(folder)) {
    await app.vault.createFolder(folder).catch(() => undefined);
  }
  const content = renderTaskNote({
    id: opts.id,
    status: "todo",
    instruction: opts.instruction,
    context: opts.context,
    notePath: opts.notePath,
    conversation: opts.conversation || conversationForNote(opts.notePath),
    approved: false,
  });
  const existing = app.vault.getAbstractFileByPath(path);
  if (existing) {
    await app.vault.modify(existing as TFile, content);
  } else {
    await app.vault.create(path, content);
  }
  return path;
}

export async function writeTaskResult(
  app: App,
  taskPath: string,
  result: string,
  status: string,
): Promise<void> {
  const file = app.vault.getAbstractFileByPath(taskPath);
  if (!file || !("extension" in file)) return;
  const text = await app.vault.read(file as TFile);
  const meta = parseTaskFile(taskPath, text);
  if (!meta) return;
  const content = renderTaskNote({
    id: meta.id,
    status,
    instruction: meta.instruction,
    context: getSection(
      text.includes("---")
        ? text.slice(text.indexOf("\n---", 3) + 4)
        : text,
      "Context",
    ),
    result,
    notePath: meta.note,
    conversation: meta.conversation,
    approved: false,
  });
  await app.vault.modify(file as TFile, content);
}

export async function setApproved(
  app: App,
  taskPath: string,
  approved: boolean,
): Promise<void> {
  const file = app.vault.getAbstractFileByPath(taskPath);
  if (!file || !("extension" in file)) return;
  await app.fileManager.processFrontMatter(file as TFile, (fm) => {
    fm.approved = approved;
    if (approved) fm.status = "done";
  });
}

export async function setTaskStatus(
  app: App,
  taskPath: string,
  status: string,
): Promise<void> {
  const file = app.vault.getAbstractFileByPath(taskPath);
  if (!file || !("extension" in file)) return;
  await app.fileManager.processFrontMatter(file as TFile, (fm) => {
    fm.status = status;
  });
}

export async function appendInstructionFollowUp(
  app: App,
  taskPath: string,
  followUp: string,
): Promise<void> {
  const file = app.vault.getAbstractFileByPath(taskPath);
  if (!file || !("extension" in file)) return;
  const text = await app.vault.read(file as TFile);
  const meta = parseTaskFile(taskPath, text);
  if (!meta) return;
  const bodyStart = text.indexOf("\n---", 3) + 4;
  const body = text.slice(bodyStart);
  const context = getSection(body, "Context");
  const instruction =
    meta.instruction.trim() +
    "\n\n### Follow-up\n" +
    followUp.trim();
  const content = renderTaskNote({
    id: meta.id,
    status: "todo",
    instruction,
    context,
    result: meta.result,
    notePath: meta.note,
    conversation: meta.conversation,
    approved: false,
  });
  await app.vault.modify(file as TFile, content);
}

export async function listTasksForNote(
  app: App,
  notePath: string,
  agentRoot: string,
): Promise<TaskMeta[]> {
  const folder = `${agentRoot.replace(/\/+$/, "")}/Tasks`;
  const files = app.vault.getMarkdownFiles().filter((f) =>
    f.path.startsWith(folder + "/")
  );
  const noteKey = notePath.replace(/\.md$/i, "");
  const out: TaskMeta[] = [];
  for (const file of files) {
    const text = await app.vault.read(file);
    const meta = parseTaskFile(file.path, text);
    if (!meta) continue;
    const linked = meta.note.replace(/\.md$/i, "");
    if (
      linked === noteKey ||
      meta.conversation === conversationForNote(notePath) ||
      meta.conversation === `note:${notePath}`
    ) {
      out.push(meta);
    }
  }
  return out;
}

export function insertCallout(editor: Editor, taskPath: string, title: string): void {
  const wiki = taskPath.replace(/\.md$/i, "");
  const short = (title || "Pawn task").replace(/\n/g, " ").slice(0, 60);
  const callout = `> [!pawn] [[${wiki}|${short}]]\n`;
  const cursor = editor.getCursor();
  editor.replaceRange(callout, cursor);
}

export function uuid4(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
