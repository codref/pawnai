/** Image files and note embeds the chat composer can send to a vision model. */

const IMAGE_EXT = new Set(["png", "jpg", "jpeg", "gif", "webp"]);

const WIKI_EMBED = /!\[\[([^\]|#|]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]/g;
const WIKI_EXCALIDRAW = /!\[\[([^\]|#|]+)(#[^\]|]+)?(?:\|[^\]]*)?\]\]/g;
const MD_EMBED = /!\[[^\]]*\]\(([^)]+)\)/g;
const MD_TITLE = /\s+["'].*["']\s*$/;

export interface ExcalidrawEmbed {
  /** Link path without a heading or alias, vault-relative or bare. */
  target: string;
  /** Heading or block ref, including the leading `#`, or empty. */
  fragment: string;
}

export function isExcalidrawName(name: string): boolean {
  const base = name.split(/[/\\]/).pop()?.toLowerCase() ?? "";
  return base.endsWith(".excalidraw") || base.endsWith(".excalidraw.md");
}

export function isImageName(name: string, mediaType = ""): boolean {
  const mime = mediaType.split(";", 1)[0].trim().toLowerCase();
  if (mime === "image/png" || mime === "image/jpeg" || mime === "image/gif" || mime === "image/webp") {
    return true;
  }
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  return IMAGE_EXT.has(ext);
}

export function mimeForName(name: string): string {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  if (ext === "png") return "image/png";
  if (ext === "gif") return "image/gif";
  if (ext === "webp") return "image/webp";
  if (ext === "jpg" || ext === "jpeg") return "image/jpeg";
  return "application/octet-stream";
}

/** Embed targets in document order. Remote URLs are skipped. */
export function noteImageTargets(markdown: string): string[] {
  const found: string[] = [];
  const seen = new Set<string>();
  const located: { index: number; raw: string }[] = [];
  for (const match of markdown.matchAll(WIKI_EMBED)) {
    located.push({ index: match.index ?? 0, raw: match[1] ?? "" });
  }
  for (const match of markdown.matchAll(MD_EMBED)) {
    located.push({ index: match.index ?? 0, raw: match[1] ?? "" });
  }
  located.sort((a, b) => a.index - b.index);
  const add = (raw: string) => {
    const target = cleanEmbedTarget(raw);
    if (!target || !isImageName(target)) return;
    if (seen.has(target)) return;
    seen.add(target);
    found.push(target);
  };
  for (const item of located) add(item.raw);
  return found;
}

/** Excalidraw wiki embeds in document order, including an optional `#^frame`. */
export function noteExcalidrawTargets(markdown: string): ExcalidrawEmbed[] {
  const found: ExcalidrawEmbed[] = [];
  const seen = new Set<string>();
  for (const match of markdown.matchAll(WIKI_EXCALIDRAW)) {
    const cleaned = cleanEmbedTarget(match[1] ?? "");
    if (!cleaned || !isExcalidrawName(cleaned)) continue;
    const fragment = (match[2] ?? "").trim();
    const key = `${cleaned}\0${fragment}`;
    if (seen.has(key)) continue;
    seen.add(key);
    found.push({ target: cleaned, fragment });
  }
  return found;
}

function cleanEmbedTarget(raw: string): string {
  let target = raw.trim();
  if (target.startsWith("<") && target.endsWith(">") && target.length > 2) {
    target = target.slice(1, -1).trim();
  }
  target = target.replace(MD_TITLE, "").trim().replace(/^\.\//, "");
  if (!target || /^(https?:|data:|mailto:|#)/i.test(target)) return "";
  try {
    target = decodeURIComponent(target);
  } catch {
    // Keep the raw target when it is not percent-encoded.
  }
  return target.replace(/\\/g, "/");
}

export function bytesToBase64(data: ArrayBuffer): string {
  const bytes = new Uint8Array(data);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}
