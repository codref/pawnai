import { App, Notice, TFile, normalizePath } from "obsidian";

/** Idea notes live here. The Inbox lists those whose status is ``inbox``. */
export const IDEAS_FOLDER = "Ideas";
export const GOALS_PATH = "Goals.md";

const UNSAFE = /[\\/:*?"<>|\u0000-\u001f]/g;
const TITLE_MAX = 60;

export interface IdeaCard {
  path: string;
  title: string;
  line: string;
  file: TFile;
}

export function ideaFilenameTitle(title: string): string {
  const line = (title || "").split("\n")[0] ?? "";
  const cleaned = line.replace(UNSAFE, "").trim().replace(/^#+/, "").trim();
  return cleaned.slice(0, TITLE_MAX).trim() || "idea";
}

export function renderIdeaNote(line: string): string {
  const heading = ideaFilenameTitle(line);
  return `---\ntags: [idea]\nstatus: inbox\n---\n\n# ${heading}\n\n${line.trim()}\n`;
}

export function ideaNotePath(line: string, day: string): string {
  return normalizePath(`${IDEAS_FOLDER}/${day} ${ideaFilenameTitle(line)}.md`);
}

function frontmatterBody(text: string): { meta: string; rest: string } | null {
  if (!text.startsWith("---\n")) return null;
  const end = text.indexOf("\n---", 3);
  if (end < 0) return null;
  return { meta: text.slice(4, end), rest: text.slice(end + 4) };
}

export function frontmatterStatus(text: string): string {
  const parsed = frontmatterBody(text);
  if (!parsed) return "";
  const match = parsed.meta.match(/^status:\s*(.*)$/m);
  return match ? match[1].trim() : "";
}

export function ideaTitle(text: string, fallback: string): string {
  const parsed = frontmatterBody(text);
  const body = (parsed ? parsed.rest : text).trim();
  const heading = body.split("\n").find((line) => line.startsWith("# "));
  if (heading) return heading.replace(/^#\s+/, "").trim() || fallback;
  return fallback;
}

export function ideaLine(text: string): string {
  const parsed = frontmatterBody(text);
  const body = (parsed ? parsed.rest : text).trim();
  const lines = body.split("\n");
  const rest = lines[0]?.startsWith("#") ? lines.slice(1).join("\n").trim() : body;
  return rest;
}

export function withStatus(text: string, status: string): string {
  const parsed = frontmatterBody(text);
  if (!parsed) {
    return `---\ntags: [idea]\nstatus: ${status}\n---\n\n${text}`;
  }
  let meta = parsed.meta;
  if (/^status:\s*/m.test(meta)) {
    meta = meta.replace(/^status:.*$/m, `status: ${status}`);
  } else {
    if (meta.length && !meta.endsWith("\n")) meta += "\n";
    meta += `status: ${status}\n`;
  }
  if (!meta.endsWith("\n")) meta += "\n";
  return `---\n${meta}---${parsed.rest}`;
}

async function ensureFolder(app: App, path: string): Promise<void> {
  const parts = normalizePath(path).split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (!app.vault.getAbstractFileByPath(current)) {
      await app.vault.createFolder(current);
    }
  }
}

export async function captureIdea(app: App, line: string): Promise<string> {
  const raw = line.trim();
  if (!raw) throw new Error("Write something first.");
  const day = window.moment().format("YYYY-MM-DD");
  const path = ideaNotePath(raw, day);
  if (app.vault.getAbstractFileByPath(path)) {
    throw new Error(`Already captured at ${path}. Left the existing note in place.`);
  }
  await ensureFolder(app, IDEAS_FOLDER);
  await app.vault.create(path, renderIdeaNote(raw));
  return path;
}

export async function setIdeaStatus(app: App, file: TFile, status: string): Promise<void> {
  const text = await app.vault.read(file);
  await app.vault.modify(file, withStatus(text, status));
}

export async function listInboxIdeas(app: App): Promise<IdeaCard[]> {
  const files = app.vault
    .getMarkdownFiles()
    .filter((file) => file.path.startsWith(`${IDEAS_FOLDER}/`));
  const cards: IdeaCard[] = [];
  for (const file of files) {
    const cache = app.metadataCache.getFileCache(file);
    const status = cache?.frontmatter?.status;
    if (status !== undefined && String(status).trim() !== "inbox") continue;
    // Only read body for notes that look like inbox (or lack cache).
    if (status === undefined) {
      const text = await app.vault.cachedRead(file);
      if (frontmatterStatus(text) !== "inbox") continue;
      cards.push({
        path: file.path,
        title: ideaTitle(text, file.basename),
        line: ideaLine(text),
        file,
      });
      continue;
    }
    const text = await app.vault.cachedRead(file);
    cards.push({
      path: file.path,
      title: ideaTitle(text, file.basename),
      line: ideaLine(text),
      file,
    });
  }
  cards.sort((a, b) => b.path.localeCompare(a.path));
  return cards;
}

function pawnValue(text: string): string {
  const parsed = frontmatterBody(text);
  if (!parsed) return "";
  const match = parsed.meta.match(/^pawn:\s*(.*)$/m);
  return match ? match[1].trim() : "";
}

function hasThread(text: string, title: string): boolean {
  const needle = title.trim().toLowerCase();
  return text.split("\n").some((line) => {
    const match = line.trim().match(/^###\s+(.+)$/);
    return match ? match[1].trim().toLowerCase() === needle : false;
  });
}

function spliceParked(existing: string, block: string): string {
  const lines = existing.split("\n");
  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    if (/^##\s+parked\s*$/i.test(lines[i].trim())) {
      start = i;
      break;
    }
  }
  const chunk = block.trim();
  if (start < 0) {
    let insertAt = lines.length;
    for (let i = 0; i < lines.length; i++) {
      if (/^##\s+done/i.test(lines[i].trim())) {
        insertAt = i;
        break;
      }
    }
    const head = lines.slice(0, insertAt).join("\n").trimEnd();
    const tail = lines.slice(insertAt).join("\n").replace(/^\n+/, "");
    const section = `## Parked\n\n${chunk}\n`;
    const merged = `${head ? `${head}\n\n` : ""}${section}${tail ? `\n${tail}` : ""}`;
    return merged.endsWith("\n") ? merged : `${merged}\n`;
  }
  let end = lines.length;
  for (let i = start + 1; i < lines.length; i++) {
    if (/^##\s+/.test(lines[i].trim())) {
      end = i;
      break;
    }
  }
  const head = lines.slice(0, end).join("\n").trimEnd();
  const tail = lines.slice(end).join("\n");
  const merged = `${head}\n\n${chunk}\n${tail}`;
  return merged.endsWith("\n") ? merged : `${merged}\n`;
}

/** Append a Parked thread that links this idea. Does not change the idea note. */
export async function parkIdea(app: App, ideaPath: string, title: string, line: string): Promise<string> {
  const name = title.trim().slice(0, 120);
  if (!name) throw new Error("A goal thread needs a name.");
  const link = `[[${ideaPath.replace(/\.md$/, "")}]]`;
  const block = `### ${name}\n- note: ${link}\n- do: ${line.replace(/\s+/g, " ").trim()}`;
  const existing = app.vault.getAbstractFileByPath(GOALS_PATH);
  if (existing instanceof TFile) {
    const current = await app.vault.read(existing);
    const pawn = pawnValue(current);
    if (pawn && pawn !== "goals") {
      throw new Error("Goals.md is not a goals note, so it was left unchanged.");
    }
    if (hasThread(current, name)) {
      return `Goals.md already has a thread named '${name}'.`;
    }
    await app.vault.modify(existing, spliceParked(current, block));
    return `Parked '${name}' in ${GOALS_PATH}.`;
  }
  if (existing) throw new Error(`${GOALS_PATH} is not a note.`);
  await app.vault.create(GOALS_PATH, `---\npawn: goals\n---\n\n## Parked\n\n${block}\n`);
  return `Parked '${name}' in ${GOALS_PATH}.`;
}

function taskStem(title: string): string {
  let cleaned = title.replace(UNSAFE, " ").replace(/\s+/g, " ").trim().replace(/\.+$/, "");
  if (cleaned.length > 80) cleaned = cleaned.slice(0, 80).trim();
  return cleaned || "task";
}

/** Write one TaskNotes note and return its path. */
export async function createTaskFromIdea(
  app: App,
  agentRoot: string,
  ideaPath: string,
  title: string,
  line: string,
): Promise<string> {
  const root = agentRoot.replace(/\\/g, "/").replace(/\/+$/, "") || "Pawn";
  const folder = normalizePath(`${root}/TaskNotes/Tasks`);
  await ensureFolder(app, folder);
  const stem = taskStem(title);
  let path = normalizePath(`${folder}/${stem}.md`);
  let n = 2;
  while (app.vault.getAbstractFileByPath(path)) {
    path = normalizePath(`${folder}/${stem} ${n}.md`);
    n += 1;
  }
  const link = `[[${ideaPath.replace(/\.md$/, "")}]]`;
  const body =
    `---\ntags: [task]\ntitle: ${JSON.stringify(title)}\nstatus: open\npriority: normal\n---\n\n` +
    `Source: ${link}\n\n${line.trim()}\n`;
  await app.vault.create(path, body);
  return path;
}

export function noticeError(error: unknown): void {
  new Notice(error instanceof Error ? error.message : String(error), 6000);
}
