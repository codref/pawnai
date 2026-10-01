/** Export an Excalidraw drawing to PNG through the installed Excalidraw plugin. */

import { App, TFile } from "obsidian";

import { isExcalidrawName } from "./noteImages";

const PLUGIN_ID = "obsidian-excalidraw-plugin";
const MAX_BYTES = 4 * 1024 * 1024;

interface ExportApi {
  reset: () => void;
  isExcalidrawFile: (file: TFile) => boolean;
  createPNGBase64: (
    templatePath?: string,
    scale?: number,
    exportSettings?: unknown,
    loader?: unknown,
    theme?: string,
  ) => Promise<string>;
  getExportSettings: (withBackground: boolean, withTheme: boolean) => unknown;
  getEmbeddedFilesLoader: (isDark?: boolean) => unknown;
}

interface ExcalidrawHost {
  ea?: ExportApi & { getAPI?: () => ExportApi };
}

export type DiagramExport =
  | { status: "ok"; filename: string; dataBase64: string }
  | { status: "missing" }
  | { status: "too-large"; name: string }
  | { status: "failed"; name: string };

interface CacheEntry {
  fingerprint: string;
  dataBase64: string;
}

const cache = new Map<string, CacheEntry>();

/** Hash of the drawing scene. The whole note is used when `## Drawing` is absent. */
export function drawingFingerprint(markdown: string): string {
  const at = markdown.indexOf("## Drawing");
  const scene = at >= 0 ? markdown.slice(at) : markdown;
  let h1 = 0x811c9dc5;
  let h2 = 0x811c9dc5 ^ scene.length;
  for (let i = 0; i < scene.length; i++) {
    const code = scene.charCodeAt(i);
    h1 = Math.imul(h1 ^ code, 0x01000193);
    h2 = Math.imul(h2 ^ code, 0x01000193);
  }
  return `${h1 >>> 0}-${h2 >>> 0}`;
}

export function exportFilename(file: TFile): string {
  const stem = file.name.replace(/\.md$/i, "");
  return isExcalidrawName(stem) || isExcalidrawName(file.name) ? `${stem}.png` : `${file.basename}.png`;
}

function excalidrawHost(app: App): ExcalidrawHost | null {
  const manager = (
    app as unknown as {
      plugins?: {
        plugins?: Record<string, ExcalidrawHost>;
        getPlugin?: (id: string) => ExcalidrawHost | null;
      };
    }
  ).plugins;
  const host = manager?.plugins?.[PLUGIN_ID] ?? manager?.getPlugin?.(PLUGIN_ID) ?? null;
  return host?.ea ? host : null;
}

export function excalidrawApi(app: App): ExportApi | null {
  const ea = excalidrawHost(app)?.ea;
  if (!ea?.getAPI) return null;
  try {
    // getAPI uses `this.plugin`. A detached call throws and looks like a missing plugin.
    return ea.getAPI();
  } catch {
    return null;
  }
}

export function fileIsExcalidraw(app: App, file: TFile): boolean {
  if (isExcalidrawName(file.name)) return true;
  const ea = excalidrawHost(app)?.ea;
  if (!ea) return false;
  try {
    return ea.isExcalidrawFile(file);
  } catch {
    return false;
  }
}

/**
 * PNG for one drawing. The same scene reuses the previous bytes so the
 * caption digest stays stable. A changed scene exports again.
 */
export async function exportExcalidrawImage(
  app: App,
  file: TFile,
  fragment = "",
): Promise<DiagramExport> {
  const name = exportFilename(file);
  if (!excalidrawHost(app)) return { status: "missing" };
  const api = excalidrawApi(app);
  if (!api) return { status: "failed", name };
  const key = `${file.path}\0${fragment}`;
  const fingerprint = await sceneFingerprint(app, file, fragment);
  if (fingerprint) {
    const cached = cache.get(key);
    if (cached && cached.fingerprint === fingerprint) {
      return { status: "ok", filename: name, dataBase64: cached.dataBase64 };
    }
  }
  const template = fragment
    ? `${file.path}${fragment.startsWith("#") ? fragment : `#${fragment}`}`
    : file.path;
  try {
    let encoded = await renderPng(api, template, 2);
    if (base64ByteLength(encoded) > MAX_BYTES) {
      encoded = await renderPng(api, template, 1);
    }
    if (!encoded || base64ByteLength(encoded) > MAX_BYTES) {
      return { status: "too-large", name };
    }
    if (fingerprint) cache.set(key, { fingerprint, dataBase64: encoded });
    return { status: "ok", filename: name, dataBase64: encoded };
  } catch {
    return { status: "failed", name };
  }
}

async function sceneFingerprint(app: App, file: TFile, fragment: string): Promise<string | null> {
  try {
    const body = await app.vault.cachedRead(file);
    return `${fragment}\n${drawingFingerprint(body)}`;
  } catch {
    return null;
  }
}

async function renderPng(api: ExportApi, template: string, scale: number): Promise<string> {
  api.reset();
  const data = await api.createPNGBase64(
    template,
    scale,
    api.getExportSettings(true, false),
    api.getEmbeddedFilesLoader(false),
    "light",
  );
  return pngBase64(data);
}

function pngBase64(value: string): string {
  const text = (value || "").trim();
  const comma = text.indexOf(",");
  const payload = text.startsWith("data:") && comma >= 0 ? text.slice(comma + 1) : text;
  return payload.replace(/\s/g, "");
}

function base64ByteLength(data: string): number {
  if (!data) return 0;
  const padding = data.endsWith("==") ? 2 : data.endsWith("=") ? 1 : 0;
  return Math.floor((data.length * 3) / 4) - padding;
}
