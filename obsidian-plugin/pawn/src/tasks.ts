import { App, TFile, normalizePath } from "obsidian";
import { noteConversationId } from "./chat/conversations";

/** Task note statuses (Pawn/Tasks/*.md frontmatter). */
export type TaskStatus = "todo" | "running" | "review" | "done" | "blocked";

export interface ParsedTaskNote {
  id: string;
  status: TaskStatus;
  note?: string;
  conversation: string;
  approved: boolean;
  instruction: string;
  context: string;
  result: string;
  path: string;
}

const STATUSES: TaskStatus[] = ["todo", "running", "review", "done", "blocked"];

export function tasksFolder(agentRoot: string): string {
  return normalizePath(`${agentRoot.replace(/\\/g, "/").replace(/\/+$/, "")}/Tasks`);
}

export function taskPathForId(agentRoot: string, id: string): string {
  return normalizePath(`${tasksFolder(agentRoot)}/${id}.md`);
}

function parseSimpleYaml(yaml: string): Record<string, string | boolean> {
  const out: Record<string, string | boolean> = {};
  for (const line of yaml.split(/\r?\n/)) {
    const m = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!m) continue;
    let val = m[2].trim();
    if (val === "true") out[m[1]] = true;
    else if (val === "false") out[m[1]] = false;
    else {
      if (
        (val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))
      ) {
        val = val.slice(1, -1);
      }
      out[m[1]] = val;
    }
  }
  return out;
}

function splitFrontmatter(text: string): { meta: Record<string, string | boolean>; body: string } {
  const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!m) return { meta: {}, body: text };
  return { meta: parseSimpleYaml(m[1]), body: m[2] };
}

function getSection(body: string, name: string): string {
  const lines = (body || "").split(/\r?\n/);
  const start = lines.findIndex((l) => new RegExp(`^##\\s+${name}\\s*$`).test(l));
  if (start < 0) return "";
  const rest = lines.slice(start + 1);
  const end = rest.findIndex((l) => /^##\s+/.test(l));
  return (end < 0 ? rest : rest.slice(0, end)).join("\n").replace(/\s+$/, "");
}

function noteFromMeta(raw: unknown): string | undefined {
  if (typeof raw !== "string") return undefined;
  let s = raw.trim();
  if (s.startsWith("[[") && s.endsWith("]]")) s = s.slice(2, -2).split("|")[0].trim();
  return s || undefined;
}

export function parseTaskNote(text: string, path: string): ParsedTaskNote | null {
  const { meta, body } = splitFrontmatter(text);
  if (meta.pawn !== "task") return null;
  const id = String(meta.id ?? "").trim();
  if (!id) return null;
  let status = String(meta.status ?? "todo").toLowerCase() as TaskStatus;
  if (!STATUSES.includes(status)) status = "todo";
  return {
    id,
    status,
    note: noteFromMeta(meta.note),
    conversation: String(meta.conversation ?? "").trim(),
    approved: meta.approved === true,
    instruction: getSection(body, "Instruction"),
    context: getSection(body, "Context"),
    result: getSection(body, "Result"),
    path,
  };
}

function renderTaskNote(p: {
  id: string;
  instruction: string;
  context: string;
  conversation: string;
  notePath?: string;
  model?: string;
}): string {
  const meta = [
    "pawn: task",
    `id: ${p.id}`,
    "status: todo",
    "approved: false",
    `conversation: ${p.conversation}`,
  ];
  if (p.model) meta.push(`model: ${JSON.stringify(p.model)}`);
  if (p.notePath) {
    const link = `[[${p.notePath.replace(/\.md$/i, "")}]]`;
    meta.push(`note: ${p.notePath.includes(" ") ? `"${link}"` : link}`);
  }
  let body = `## Instruction\n${p.instruction.trim()}\n`;
  if (p.context.trim()) body += `\n## Context\n${p.context.trim()}\n`;
  return `---\n${meta.join("\n")}\n---\n\n${body}`;
}

/** Offline fallback: a todo task note the vault watcher runs once it syncs to S3. */
export async function createTaskNote(
  app: App,
  agentRoot: string,
  p: {
    id: string;
    instruction: string;
    context?: string;
    notePath?: string;
    conversation?: string;
    model?: string;
  },
): Promise<string> {
  const path = taskPathForId(agentRoot, p.id);
  const folder = tasksFolder(agentRoot);
  if (!app.vault.getAbstractFileByPath(folder)) {
    await app.vault.createFolder(folder).catch(() => undefined);
  }
  const content = renderTaskNote({
    id: p.id,
    instruction: p.instruction,
    context: p.context ?? "",
    conversation:
      p.conversation ?? (p.notePath ? noteConversationId(p.notePath) : noteConversationId(path)),
    notePath: p.notePath,
    model: p.model,
  });
  const existing = app.vault.getAbstractFileByPath(path);
  if (existing instanceof TFile) await app.vault.modify(existing, content);
  else await app.vault.create(path, content);
  return path;
}

export async function setApproved(app: App, taskPath: string): Promise<void> {
  const file = app.vault.getAbstractFileByPath(taskPath);
  if (!(file instanceof TFile)) return;
  await app.fileManager.processFrontMatter(file, (fm) => {
    fm.approved = true;
  });
}

/** Close without indexing: status done, approved false (watcher / dismiss API). */
export async function setDismissed(app: App, taskPath: string): Promise<void> {
  const file = app.vault.getAbstractFileByPath(taskPath);
  if (!(file instanceof TFile)) return;
  await app.fileManager.processFrontMatter(file, (fm) => {
    fm.status = "done";
    fm.approved = false;
  });
}

export async function listTaskNotes(app: App, agentRoot: string): Promise<ParsedTaskNote[]> {
  const prefix = `${tasksFolder(agentRoot)}/`;
  const out: ParsedTaskNote[] = [];
  for (const file of app.vault.getMarkdownFiles()) {
    if (!file.path.startsWith(prefix)) continue;
    const parsed = parseTaskNote(await app.vault.cachedRead(file), file.path);
    if (parsed) out.push(parsed);
  }
  return out;
}
