import { App, TFile, normalizePath } from "obsidian";
import { dayKey, dayLabel } from "./items";

/** Default research inbox folder (matches capture.inbox_dir default). */
export const CAPTURES_INBOX_FOLDER = "Pawn/Captures/Inbox";

export interface CaptureCard {
  path: string;
  title: string;
  status: string;
  collection: string;
  entity: string;
  hint: string;
  type: string;
  snippet_kind: string;
  source_url: string;
  caption: string;
  created: string;
  captured_at: string;
  file: TFile;
}

function frontmatterStatus(text: string): string {
  const match = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return "";
  const status = match[1].match(/^status:\s*(.*)$/m);
  return status ? status[1].trim().replace(/^["']|["']$/g, "") : "";
}

function fmString(cache: Record<string, unknown> | undefined, key: string): string {
  if (!cache) return "";
  const raw = cache[key];
  if (raw === undefined || raw === null) return "";
  return String(raw).trim().replace(/^["']|["']$/g, "");
}

function captureTitle(text: string, basename: string): string {
  for (const line of text.split("\n")) {
    if (line.startsWith("# ")) return line.slice(2).trim() || basename;
  }
  return basename;
}

export function captureTypeLabel(type: string, snippetKind: string): string {
  const t = (type || "").trim();
  if (t) return t.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const sk = (snippetKind || "").trim().toLowerCase();
  if (sk === "image") return "Image";
  if (sk === "text") return "Text";
  return "Capture";
}

export function captureTypeClass(type: string, snippetKind: string): string {
  const t = (type || "").trim().toLowerCase().replace(/_/g, "-");
  if (t) return `is-capture-${t}`;
  const sk = (snippetKind || "").trim().toLowerCase();
  if (sk === "image") return "is-capture-image";
  if (sk === "text") return "is-capture-text";
  return "is-capture-inbox";
}

export function captureCreatedAt(card: CaptureCard): string {
  return card.created || card.captured_at || "";
}

export function groupCapturesByDay(
  items: CaptureCard[],
): { key: string; label: string; items: CaptureCard[] }[] {
  const map = new Map<string, CaptureCard[]>();
  for (const item of items) {
    const key = dayKey(captureCreatedAt(item));
    const list = map.get(key) ?? [];
    list.push(item);
    map.set(key, list);
  }
  return Array.from(map.entries())
    .sort((a, b) => b[0].localeCompare(a[0]))
    .map(([key, group]) => ({ key, label: dayLabel(key), items: group }));
}

export async function listInboxCaptures(
  app: App,
  folder: string = CAPTURES_INBOX_FOLDER,
): Promise<CaptureCard[]> {
  const prefix = normalizePath(folder).replace(/\/+$/, "") + "/";
  const files = app.vault
    .getMarkdownFiles()
    .filter((file) => file.path.startsWith(prefix) || file.path === folder);
  const cards: CaptureCard[] = [];
  for (const file of files) {
    if (file.path.includes("/assets/")) continue;
    const cache = app.metadataCache.getFileCache(file);
    const fm = cache?.frontmatter as Record<string, unknown> | undefined;
    let status = fmString(fm, "status");
    const text = await app.vault.cachedRead(file);
    if (!status) status = frontmatterStatus(text);
    if (status !== "inbox" && status !== "proposed") continue;
    cards.push({
      path: file.path,
      title: captureTitle(text, file.basename),
      status,
      collection: fmString(fm, "collection"),
      entity: fmString(fm, "entity"),
      hint: fmString(fm, "hint"),
      type: fmString(fm, "type"),
      snippet_kind: fmString(fm, "kind"),
      source_url: fmString(fm, "source_url"),
      caption: fmString(fm, "caption"),
      created: fmString(fm, "created"),
      captured_at: fmString(fm, "captured_at"),
      file,
    });
  }
  cards.sort((a, b) => {
    const ca = captureCreatedAt(a);
    const cb = captureCreatedAt(b);
    if (ca && cb) return cb.localeCompare(ca);
    return b.path.localeCompare(a.path);
  });
  return cards;
}
