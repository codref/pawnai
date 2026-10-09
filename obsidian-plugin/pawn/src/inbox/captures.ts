import { App, TFile, normalizePath } from "obsidian";

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
  return String(raw).trim();
}

function captureTitle(text: string, basename: string): string {
  for (const line of text.split("\n")) {
    if (line.startsWith("# ")) return line.slice(2).trim() || basename;
  }
  return basename;
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
    if (!status) {
      const text = await app.vault.cachedRead(file);
      status = frontmatterStatus(text);
    }
    if (status !== "inbox" && status !== "proposed") continue;
    const text = await app.vault.cachedRead(file);
    cards.push({
      path: file.path,
      title: captureTitle(text, file.basename),
      status,
      collection: fmString(fm, "collection"),
      entity: fmString(fm, "entity"),
      hint: fmString(fm, "hint"),
      type: fmString(fm, "type"),
      file,
    });
  }
  cards.sort((a, b) => b.path.localeCompare(a.path));
  return cards;
}
