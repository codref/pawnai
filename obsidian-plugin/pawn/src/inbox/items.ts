import { App, TFile, normalizePath } from "obsidian";
import type { InboxItem } from "../api";

export const OPEN_STATUSES = ["new", "notified", "snoozed"] as const;
export const CLOSED_STATUSES = ["filed", "task", "dismissed"] as const;
export type ItemsScope = "open" | "closed" | "all";
export const PAGE_SIZE = 50;

const KIND_LABEL: Record<string, string> = {
  decision: "Decision",
  commitment: "Commitment",
  open_question: "Question",
  block: "Block",
  contradiction: "Contradiction",
  proposal: "Research",
  people_update: "People",
  schedule_proposal: "Schedule",
};

export function kindLabel(kind: string): string {
  return KIND_LABEL[kind] || kind || "Item";
}

/** CSS modifier for kind chip coloring. */
export function kindClass(kind: string): string {
  const key = (kind || "item").replace(/_/g, "-");
  return `is-kind-${key}`;
}

export function statusesForScope(scope: ItemsScope): string | undefined {
  if (scope === "open") return OPEN_STATUSES.join(",");
  if (scope === "closed") return CLOSED_STATUSES.join(",");
  return "all";
}

export function statusAllowed(status: string, scope: ItemsScope): boolean {
  const s = status.toLowerCase();
  if (scope === "all") return true;
  if (scope === "open") return (OPEN_STATUSES as readonly string[]).includes(s);
  return (CLOSED_STATUSES as readonly string[]).includes(s);
}

export function dayKey(iso?: string | null): string {
  if (!iso) return "unknown";
  const m = window.moment(iso);
  if (!m.isValid()) return "unknown";
  return m.format("YYYY-MM-DD");
}

export function dayLabel(key: string): string {
  if (key === "unknown") return "Unknown date";
  const today = window.moment().format("YYYY-MM-DD");
  const yesterday = window.moment().subtract(1, "day").format("YYYY-MM-DD");
  if (key === today) return "Today";
  if (key === yesterday) return "Yesterday";
  return window.moment(key, "YYYY-MM-DD").format("ddd D MMM YYYY");
}

export function groupByDay(items: InboxItem[]): { key: string; label: string; items: InboxItem[] }[] {
  const map = new Map<string, InboxItem[]>();
  for (const item of items) {
    const key = dayKey(item.created_at);
    const list = map.get(key) ?? [];
    list.push(item);
    map.set(key, list);
  }
  return Array.from(map.entries())
    .sort((a, b) => b[0].localeCompare(a[0]))
    .map(([key, group]) => ({ key, label: dayLabel(key), items: group }));
}

export type ItemsGroupBy = "day" | "session";

export interface SessionItemGroup {
  key: string;
  label: string;
  items: InboxItem[];
}

export interface DaySessionGroup {
  key: string;
  label: string;
  sessions: SessionItemGroup[];
}

/** Session origin key for grouping; non-session sources → ``other``. */
export function itemSessionKey(item: InboxItem): string {
  if ((item.source_kind || "").trim() === "session") {
    const ref = (item.source_ref || "").trim();
    return ref || "other";
  }
  return "other";
}

/**
 * Nest items as day → session (source_ref). *titles* maps session_id → display title.
 */
export function groupBySession(
  items: InboxItem[],
  titles?: Map<string, string> | Record<string, string>,
): DaySessionGroup[] {
  const titleOf = (ref: string): string => {
    if (!ref || ref === "other") return "Other";
    if (titles instanceof Map) return titles.get(ref) || ref;
    if (titles && typeof titles === "object") return titles[ref] || ref;
    return ref;
  };
  const byDay = new Map<string, Map<string, InboxItem[]>>();
  for (const item of items) {
    const d = dayKey(item.created_at);
    const sk = itemSessionKey(item);
    const dayMap = byDay.get(d) ?? new Map<string, InboxItem[]>();
    const list = dayMap.get(sk) ?? [];
    list.push(item);
    dayMap.set(sk, list);
    byDay.set(d, dayMap);
  }
  return Array.from(byDay.entries())
    .sort((a, b) => b[0].localeCompare(a[0]))
    .map(([key, sessionMap]) => ({
      key,
      label: dayLabel(key),
      sessions: Array.from(sessionMap.entries())
        .sort((a, b) => {
          if (a[0] === "other") return 1;
          if (b[0] === "other") return -1;
          return titleOf(a[0]).localeCompare(titleOf(b[0]));
        })
        .map(([sk, group]) => ({
          key: sk,
          label: sk === "other" ? "Other" : titleOf(sk),
          items: group,
        })),
    }));
}

function frontmatterBody(text: string): { meta: string; rest: string } | null {
  if (!text.startsWith("---\n")) return null;
  const end = text.indexOf("\n---", 3);
  if (end < 0) return null;
  return { meta: text.slice(4, end), rest: text.slice(end + 4) };
}

function metaValue(meta: string, key: string): string {
  const match = meta.match(new RegExp(`^${key}:\\s*(.*)$`, "m"));
  return match ? match[1].trim() : "";
}

function bodyText(rest: string): { text: string; context: string; quote: string } {
  const lines = rest.trim().split("\n");
  const textLines: string[] = [];
  const contextLines: string[] = [];
  const quoteLines: string[] = [];
  let section: "text" | "context" | "quote" | "done" = "text";
  for (const line of lines) {
    if (section === "done") break;
    if (line.startsWith("## Context")) {
      section = "context";
      continue;
    }
    if (line.startsWith("Source:") || (line.startsWith("## ") && !line.startsWith("## Context"))) {
      section = "done";
      continue;
    }
    if (line.startsWith(">")) {
      section = "quote";
      quoteLines.push(line.replace(/^>\s?/, ""));
      continue;
    }
    if (section === "quote") {
      if (!line.trim()) {
        section = "done";
        continue;
      }
      // Non-blockquote after quote ends the quote block.
      section = "done";
      continue;
    }
    if (section === "context") {
      contextLines.push(line);
      continue;
    }
    if (section === "text") textLines.push(line);
  }
  return {
    text: textLines.join("\n").trim() || "(empty)",
    context: contextLines.join("\n").trim(),
    quote: quoteLines.join("\n").trim(),
  };
}

/** Parse a vault markdown file into an InboxItem when frontmatter is ``pawn: item``. */
export function parseItemNote(
  path: string,
  text: string,
  mtime?: number,
  scope: ItemsScope = "open",
): InboxItem | null {
  const parsed = frontmatterBody(text);
  if (!parsed) return null;
  if (metaValue(parsed.meta, "pawn") !== "item") return null;
  const status = (metaValue(parsed.meta, "status") || "new").toLowerCase();
  if (!statusAllowed(status, scope)) return null;
  const id = metaValue(parsed.meta, "id") || path;
  const shortId = metaValue(parsed.meta, "short_id") || id.slice(0, 8);
  const { text: body, context, quote } = bodyText(parsed.rest);
  const created =
    mtime != null ? window.moment(mtime).toISOString() : window.moment().toISOString();
  const openedAt = metaValue(parsed.meta, "opened_at") || null;
  const sourceKind = metaValue(parsed.meta, "source_kind") || null;
  const sourceRef = metaValue(parsed.meta, "source_ref") || null;
  return {
    id,
    short_id: shortId,
    kind: metaValue(parsed.meta, "kind") || "item",
    text: body,
    thread: metaValue(parsed.meta, "thread") || null,
    status,
    interrupt: false,
    reason: null,
    quote: quote || null,
    context: context || null,
    note_key: path,
    created_at: created,
    opened_at: openedAt,
    source_kind: sourceKind,
    source_ref: sourceRef,
  };
}

export function countVaultItemNotes(app: App, itemsDir: string): number {
  const root = normalizePath(itemsDir.replace(/\/+$/, ""));
  const prefix = root.endsWith("/") ? root : `${root}/`;
  return app.vault.getMarkdownFiles().filter((f) => f.path.startsWith(prefix)).length;
}

export interface ItemNoteRef {
  path: string;
  id: string;
  short_id: string;
  status: string;
}

/** List every ``pawn: item`` note under Items/ (lightweight frontmatter only). */
export async function listVaultItemRefs(app: App, itemsDir: string): Promise<ItemNoteRef[]> {
  const root = normalizePath(itemsDir.replace(/\/+$/, ""));
  const prefix = root.endsWith("/") ? root : `${root}/`;
  const files = app.vault.getMarkdownFiles().filter((f) => f.path.startsWith(prefix));
  const refs: ItemNoteRef[] = [];
  for (const file of files) {
    const cache = app.metadataCache.getFileCache(file);
    const fm = cache?.frontmatter;
    if (fm && String(fm.pawn || "") === "item") {
      refs.push({
        path: file.path,
        id: String(fm.id || "").trim(),
        short_id: String(fm.short_id || "").trim(),
        status: String(fm.status || "new").trim().toLowerCase(),
      });
      continue;
    }
    const text = await app.vault.cachedRead(file);
    const parsed = frontmatterBody(text);
    if (!parsed || metaValue(parsed.meta, "pawn") !== "item") continue;
    refs.push({
      path: file.path,
      id: metaValue(parsed.meta, "id"),
      short_id: metaValue(parsed.meta, "short_id"),
      status: (metaValue(parsed.meta, "status") || "new").toLowerCase(),
    });
  }
  return refs;
}

/**
 * Trash item notes that are safe to flush:
 * - status is explicitly closed, or
 * - note is not among the live open items (orphan / stale note).
 * Notes matching *keep* (open id / short_id / note_key) are never touched.
 */
export async function flushFlushableVaultNotes(
  app: App,
  itemsDir: string,
  keep: Set<string>,
  onProgress?: (done: number, total: number) => void,
): Promise<number> {
  const refs = await listVaultItemRefs(app, itemsDir);
  const toTrash: TFile[] = [];
  for (const ref of refs) {
    const kept =
      (ref.id && keep.has(ref.id)) ||
      (ref.short_id && keep.has(ref.short_id)) ||
      keep.has(ref.path);
    if (kept) continue;
    const file = app.vault.getAbstractFileByPath(ref.path);
    if (file instanceof TFile) toTrash.push(file);
  }
  let removed = 0;
  for (const file of toTrash) {
    await app.fileManager.trashFile(file);
    removed += 1;
    onProgress?.(removed, toTrash.length);
  }
  return removed;
}

/** @deprecated use flushFlushableVaultNotes */
export async function flushClosedVaultNotes(
  app: App,
  itemsDir: string,
  onProgress?: (done: number, total: number) => void,
): Promise<number> {
  return flushFlushableVaultNotes(app, itemsDir, new Set(), onProgress);
}

/**
 * Offline list of items. Prefer metadataCache frontmatter so we avoid
 * reading hundreds of note bodies on every refresh.
 */
export async function listItemNotes(
  app: App,
  itemsDir: string,
  scope: ItemsScope = "open",
): Promise<InboxItem[]> {
  const root = normalizePath(itemsDir.replace(/\/+$/, ""));
  const prefix = root.endsWith("/") ? root : `${root}/`;
  const files = app.vault.getMarkdownFiles().filter((f) => f.path.startsWith(prefix));
  const items: InboxItem[] = [];
  for (const file of files) {
    const cache = app.metadataCache.getFileCache(file);
    const fm = cache?.frontmatter;
    if (fm && String(fm.pawn || "") === "item") {
      const status = String(fm.status || "new").trim().toLowerCase();
      if (!statusAllowed(status, scope)) continue;
      const id = String(fm.id || file.path);
      const shortId = String(fm.short_id || id.slice(0, 8));
      const kind = String(fm.kind || "item");
      const thread = fm.thread != null && String(fm.thread).trim() ? String(fm.thread) : null;
      const base = file.basename
        .replace(/^[0-9]{4}-[0-9]{2}-[0-9]{2}-/, "")
        .replace(/-[a-f0-9]{8}$/i, "");
      const label = base.replace(/-/g, " ").trim() || shortId;
      const openedRaw = fm.opened_at != null ? String(fm.opened_at).trim() : "";
      const sourceKind =
        fm.source_kind != null && String(fm.source_kind).trim()
          ? String(fm.source_kind).trim()
          : null;
      const sourceRef =
        fm.source_ref != null && String(fm.source_ref).trim()
          ? String(fm.source_ref).trim()
          : null;
      items.push({
        id,
        short_id: shortId,
        kind,
        text: label,
        thread,
        status,
        interrupt: false,
        note_key: file.path,
        created_at: window.moment(file.stat.mtime).toISOString(),
        opened_at: openedRaw || null,
        source_kind: sourceKind,
        source_ref: sourceRef,
      });
      continue;
    }
    const text = await app.vault.cachedRead(file);
    const item = parseItemNote(file.path, text, file.stat.mtime, scope);
    if (item) items.push(item);
  }
  items.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
  return items;
}

/** @deprecated use listItemNotes */
export async function listOpenItemNotes(app: App, itemsDir: string): Promise<InboxItem[]> {
  return listItemNotes(app, itemsDir, "open");
}

function setMetaLine(meta: string, key: string, value: string): string {
  const line = `${key}: ${value}`;
  const re = new RegExp(`^${key}:\\s*.*$`, "m");
  if (re.test(meta)) return meta.replace(re, line);
  let next = meta;
  if (next.length && !next.endsWith("\n")) next += "\n";
  return `${next}${line}\n`;
}

function clearMetaLine(meta: string, key: string): string {
  return meta.replace(new RegExp(`^${key}:\\s*.*\\n?`, "m"), "");
}

/** Set ``action:`` on an item note for the vault watcher (offline triage). */
export async function setItemNoteAction(
  app: App,
  noteKey: string,
  action: string,
  arg?: string,
): Promise<void> {
  const file = app.vault.getAbstractFileByPath(noteKey);
  if (!(file instanceof TFile)) throw new Error(`Note not found: ${noteKey}`);
  const text = await app.vault.read(file);
  const parsed = frontmatterBody(text);
  if (!parsed) throw new Error("Item note has no frontmatter.");
  let meta = setMetaLine(parsed.meta, "action", action);
  if (action === "todo" && arg?.trim()) {
    meta = setMetaLine(meta, "todo_project", arg.trim());
  } else {
    meta = clearMetaLine(meta, "todo_project");
  }
  if (!meta.endsWith("\n")) meta += "\n";
  await app.vault.modify(file, `---\n${meta}---${parsed.rest}`);
}

/** Persist ``opened_at`` when the user first opens an item note. */
export async function setItemNoteOpenedAt(
  app: App,
  noteKey: string,
  openedAt: string,
): Promise<void> {
  const file = app.vault.getAbstractFileByPath(noteKey);
  if (!(file instanceof TFile)) return;
  const text = await app.vault.read(file);
  const parsed = frontmatterBody(text);
  if (!parsed || metaValue(parsed.meta, "pawn") !== "item") return;
  if (metaValue(parsed.meta, "opened_at")) return;
  let meta = setMetaLine(parsed.meta, "opened_at", openedAt);
  if (!meta.endsWith("\n")) meta += "\n";
  await app.vault.modify(file, `---\n${meta}---${parsed.rest}`);
}

/** Read ``opened_at`` from vault metadata for a note path. */
export function readOpenedAt(app: App, noteKey: string | null | undefined): string | null {
  if (!noteKey) return null;
  const file = app.vault.getAbstractFileByPath(noteKey);
  if (!(file instanceof TFile)) return null;
  const cache = app.metadataCache.getFileCache(file);
  const raw = cache?.frontmatter?.opened_at;
  if (raw != null && String(raw).trim()) return String(raw).trim();
  return null;
}

export function isItemUnread(item: InboxItem): boolean {
  const status = (item.status || "").toLowerCase();
  if (!(OPEN_STATUSES as readonly string[]).includes(status)) return false;
  return !item.opened_at;
}

export function isItemNotePath(path: string, itemsDir: string): boolean {
  const root = normalizePath(itemsDir.replace(/\/+$/, ""));
  return path === root || path.startsWith(`${root}/`);
}
