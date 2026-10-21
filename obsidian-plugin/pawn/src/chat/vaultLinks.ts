import { App, Keymap, TFile } from "obsidian";

/** Vault-relative path with `/` and a note-like extension (optional quotes). */
const PATH_RE =
  /["']?([A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_ .'-]+)+\.(?:md|base|canvas))["']?/gi;

const PLACEHOLDER_RE = /§§PAWNLINK(\d+)§§/g;

/** Char before a path must not continue a longer token (URL, dotted id). */
function isStartBoundary(ch: string | undefined): boolean {
  if (!ch) return true;
  return !/[\w./-]/.test(ch);
}

/**
 * Char after the extension may be sentence punctuation (e.g. `foo.md.`).
 * Still reject word chars / `/` / `-` so we do not swallow the next token.
 */
function isEndBoundary(ch: string | undefined): boolean {
  if (!ch) return true;
  return !/[\w/-]/.test(ch);
}

type Span = { start: number; end: number };

/** Ranges that must not be linkified (code, existing links). */
function protectedSpans(md: string): Span[] {
  const spans: Span[] = [];
  const push = (re: RegExp) => {
    re.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(md))) {
      spans.push({ start: m.index, end: m.index + m[0].length });
    }
  };
  push(/```[\s\S]*?```/g);
  push(/~~~[\s\S]*?~~~/g);
  push(/`[^`\n]+`/g);
  push(/!?\[\[[^\]]*\]\]/g);
  push(/\[[^\]]*\]\([^)]+\)/g);
  return spans.sort((a, b) => a.start - b.start);
}

function inSpan(spans: Span[], index: number, end: number): boolean {
  for (const s of spans) {
    if (s.end <= index) continue;
    if (s.start >= end) break;
    return true;
  }
  return false;
}

function asVaultPath(path: string): string {
  return (path || "").trim().replace(/\\/g, "/").replace(/^\/+/, "");
}

export interface LinkifyResult {
  markdown: string;
  paths: string[];
}

/**
 * Replace bare vault paths with opaque placeholders so MarkdownRenderer cannot
 * italicize underscores or strip anchors. Call {@link materializeVaultLinks}
 * after render to turn placeholders into real `a.internal-link` elements.
 */
export function linkifyVaultPaths(md: string): LinkifyResult {
  if (!md) return { markdown: md, paths: [] };
  const spans = protectedSpans(md);
  const paths: string[] = [];
  const out: string[] = [];
  let last = 0;
  PATH_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = PATH_RE.exec(md))) {
    const full = m[0];
    const path = asVaultPath(m[1]);
    const start = m.index;
    const end = start + full.length;
    if (!isStartBoundary(md[start - 1]) || !isEndBoundary(md[end])) continue;
    if (inSpan(spans, start, end)) continue;
    const idx = paths.length;
    paths.push(path);
    out.push(md.slice(last, start));
    out.push(`§§PAWNLINK${idx}§§`);
    last = end;
  }
  out.push(md.slice(last));
  return { markdown: out.join(""), paths };
}

function makeInternalLink(path: string): HTMLAnchorElement {
  const a = document.createElement("a");
  a.className = "internal-link";
  a.textContent = path;
  a.setAttribute("data-href", path);
  a.setAttribute("data-path", path);
  a.setAttribute("href", path);
  return a;
}

/**
 * Parse text that is entirely a vault path (optional quotes / trailing sentence punct).
 * Used for inline-code spans like `` `Pawn/Notes/foo.md` ``.
 */
export function parseVaultPathText(raw: string): { path: string; trailing: string } | null {
  let text = (raw || "").trim();
  if (!text) return null;
  if (
    (text.startsWith('"') && text.endsWith('"')) ||
    (text.startsWith("'") && text.endsWith("'"))
  ) {
    text = text.slice(1, -1).trim();
  }
  let trailing = "";
  const withPunct = text.match(/^(.*\.(?:md|base|canvas))([.,;:!?)]+)$/i);
  if (withPunct) {
    text = withPunct[1];
    trailing = withPunct[2];
  }
  const path = asVaultPath(text);
  if (!/^[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_ .'-]+)+\.(?:md|base|canvas)$/i.test(path)) {
    return null;
  }
  return { path, trailing };
}

/** Swap placeholder text nodes for clickable vault links after MarkdownRenderer. */
export function materializeVaultLinks(container: HTMLElement, paths: string[]): void {
  if (!paths.length) return;
  const walk = (node: Node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.nodeValue ?? "";
      if (!text.includes("§§PAWNLINK")) return;
      const frag = document.createDocumentFragment();
      let last = 0;
      PLACEHOLDER_RE.lastIndex = 0;
      let m: RegExpExecArray | null;
      while ((m = PLACEHOLDER_RE.exec(text))) {
        const idx = Number(m[1]);
        const path = paths[idx];
        if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        if (path) frag.appendChild(makeInternalLink(path));
        else frag.appendChild(document.createTextNode(m[0]));
        last = m.index + m[0].length;
      }
      if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
      node.parentNode?.replaceChild(frag, node);
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    const el = node as HTMLElement;
    if (el.tagName === "CODE" || el.tagName === "PRE" || el.tagName === "A") return;
    for (const child of Array.from(node.childNodes)) walk(child);
  };
  walk(container);
}

/**
 * Agents often wrap paths in backticks. Those render as `<code>`, which we
 * previously skipped — turn path-only inline code into internal links.
 * Leaves fenced `<pre><code>` and non-path code (e.g. CLI args) alone.
 */
export function linkifyInlineCodePaths(container: HTMLElement): void {
  for (const code of Array.from(container.querySelectorAll("code"))) {
    if (code.closest("pre")) continue;
    if (code.closest("a")) continue;
    const parsed = parseVaultPathText(code.textContent ?? "");
    if (!parsed) continue;
    const frag = document.createDocumentFragment();
    frag.appendChild(makeInternalLink(parsed.path));
    if (parsed.trailing) frag.appendChild(document.createTextNode(parsed.trailing));
    code.replaceWith(frag);
  }
}

function resolveLinkTarget(app: App, href: string, sourcePath: string): TFile | null {
  let target = (href || "").trim();
  if (!target) return null;
  if (target.startsWith("obsidian://")) {
    try {
      const url = new URL(target);
      target = url.searchParams.get("file") ?? "";
    } catch {
      return null;
    }
  }
  target = decodeURIComponent(target).split("#")[0].split("|")[0].trim();
  target = asVaultPath(target);
  if (!target) return null;
  const direct = app.vault.getAbstractFileByPath(target);
  if (direct instanceof TFile) return direct;
  if (!/\.[a-z0-9]+$/i.test(target)) {
    const withMd = app.vault.getAbstractFileByPath(`${target}.md`);
    if (withMd instanceof TFile) return withMd;
  }
  return (
    app.metadataCache.getFirstLinkpathDest(target.replace(/\.md$/i, ""), sourcePath) ?? null
  );
}

/** Delegated click: open vault notes from rendered internal links in a chat/job body. */
export function wireInternalLinks(app: App, container: HTMLElement, sourcePath: string): void {
  container.addEventListener("click", (ev) => {
    const el = (ev.target as HTMLElement | null)?.closest?.("a");
    if (!el || !container.contains(el)) return;
    const href =
      el.getAttribute("data-href") ||
      el.getAttribute("data-path") ||
      el.getAttribute("href") ||
      "";
    if (!href || /^https?:\/\//i.test(href) || href.startsWith("mailto:")) return;
    if (!el.classList.contains("internal-link") && !href.startsWith("obsidian://")) {
      if (!/\.(md|base|canvas)$/i.test(href) && !href.includes("/")) return;
    }
    ev.preventDefault();
    ev.stopPropagation();
    const file = resolveLinkTarget(app, href, sourcePath);
    if (file) {
      void app.workspace.getLeaf(Keymap.isModEvent(ev)).openFile(file);
      return;
    }
    void app.workspace.openLinkText(href, sourcePath, Keymap.isModEvent(ev));
  });
}
