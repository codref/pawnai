import { App, TFile, normalizePath } from "obsidian";
import type { InboxItem } from "../api";

export const OPEN_STATUSES = ["new", "notified", "snoozed"] as const;
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

function bodyText(rest: string): { text: string; quote: string } {
  const lines = rest.trim().split("\n");
  const textLines: string[] = [];
  const quoteLines: string[] = [];
  let inQuote = false;
  for (const line of lines) {
    if (line.startsWith(">")) {
      inQuote = true;
      quoteLines.push(line.replace(/^>\s?/, ""));
      continue;
    }
    if (inQuote && !line.trim()) break;
    if (line.startsWith("Source:") || line.startsWith("## ")) break;
    if (!inQuote) textLines.push(line);
  }
  return {
    text: textLines.join("\n").trim() || "(empty)",
    quote: quoteLines.join("\n").trim(),
  };
}

/** Parse a vault markdown file into an InboxItem when frontmatter is ``pawn: item``. */
export function parseItemNote(path: string, text: string, mtime?: number): InboxItem | null {
  const parsed = frontmatterBody(text);
  if (!parsed) return null;
  if (metaValue(parsed.meta, "pawn") !== "item") return null;
  const status = (metaValue(parsed.meta, "status") || "new").toLowerCase();
  if (!(OPEN_STATUSES as readonly string[]).includes(status)) return null;
  const id = metaValue(parsed.meta, "id") || path;
  const shortId = metaValue(parsed.meta, "short_id") || id.slice(0, 8);
  const { text: body, quote } = bodyText(parsed.rest);
  const created =
    mtime != null ? window.moment(mtime).toISOString() : window.moment().toISOString();
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
    note_key: path,
    created_at: created,
  };
}

/**
 * Offline list of open items. Prefer metadataCache frontmatter so we avoid
 * reading hundreds of note bodies on every refresh; fall back to a short
 * cachedRead only when status/kind are not in the cache yet.
 */
export async function listOpenItemNotes(app: App, itemsDir: string): Promise<InboxItem[]> {
  const root = normalizePath(itemsDir.replace(/\/+$/, ""));
  const prefix = root.endsWith("/") ? root : `${root}/`;
  const files = app.vault.getMarkdownFiles().filter((f) => f.path.startsWith(prefix));
  const items: InboxItem[] = [];
  for (const file of files) {
    const cache = app.metadataCache.getFileCache(file);
    const fm = cache?.frontmatter;
    if (fm && String(fm.pawn || "") === "item") {
      const status = String(fm.status || "new").trim().toLowerCase();
      if (!(OPEN_STATUSES as readonly string[]).includes(status)) continue;
      const id = String(fm.id || file.path);
      const shortId = String(fm.short_id || id.slice(0, 8));
      const kind = String(fm.kind || "item");
      const thread = fm.thread != null && String(fm.thread).trim() ? String(fm.thread) : null;
      // Body text is not in frontmatter; use basename slug as a cheap label.
      const base = file.basename.replace(/^[0-9]{4}-[0-9]{2}-[0-9]{2}-/, "").replace(/-[a-f0-9]{8}$/i, "");
      const label = base.replace(/-/g, " ").trim() || shortId;
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
      });
      continue;
    }
    // Uncached / not yet indexed — one cachedRead.
    const text = await app.vault.cachedRead(file);
    const item = parseItemNote(file.path, text, file.stat.mtime);
    if (item) items.push(item);
  }
  items.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
  return items;
}

/** Set ``action:`` on an item note for the vault watcher (offline triage). */
export async function setItemNoteAction(app: App, noteKey: string, action: string): Promise<void> {
  const file = app.vault.getAbstractFileByPath(noteKey);
  if (!(file instanceof TFile)) throw new Error(`Note not found: ${noteKey}`);
  const text = await app.vault.read(file);
  const parsed = frontmatterBody(text);
  if (!parsed) throw new Error("Item note has no frontmatter.");
  let meta = parsed.meta;
  if (/^action:\s*/m.test(meta)) {
    meta = meta.replace(/^action:.*$/m, `action: ${action}`);
  } else {
    if (meta.length && !meta.endsWith("\n")) meta += "\n";
    meta += `action: ${action}\n`;
  }
  if (!meta.endsWith("\n")) meta += "\n";
  await app.vault.modify(file, `---\n${meta}---${parsed.rest}`);
}

export function isItemNotePath(path: string, itemsDir: string): boolean {
  const root = normalizePath(itemsDir.replace(/\/+$/, ""));
  return path === root || path.startsWith(`${root}/`);
}
