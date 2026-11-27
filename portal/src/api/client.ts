// HTTP client for pawn-server. streamFetch is ported from
// obsidian-plugin/pawn/src/api.ts (desktop fetch path, no requestUrl fallback).

import { SseParser, parseSseText, type SseEvent } from "./sse";
import {
  HttpError,
  type Annotation,
  type AnnotationCreate,
  type CaptureAtom,
  type CaptureCollection,
  type CaptureFileBody,
  type Idea,
  type IdeaActionKind,
  type IdeaActionResult,
  type InboxCounts,
  type ItemAction,
  type ItemActionResult,
  type ItemListParams,
  type ItemsDeleteBody,
  type ItemsPage,
  type SettingsPayload,
  type SettingsStatus,
  type AnnotationListParams,
  type AnnotationPatch,
  type Chat,
  type ChatCreate,
  type ChatDetail,
  type ChatListParams,
  type ChatPatch,
  type ChatStreamHandlers,
  type Clip,
  type ClipCreate,
  type ClipPatch,
  type Fork,
  type FileContent,
  type FileListing,
  type FileStat,
  type FileTags,
  type ReindexResult,
  type Folder,
  type FolderCreate,
  type FolderPatch,
  type FolderTree,
  type MessageCreate,
  type ModelCatalog,
  type PortalApi,
  type SearchResults,
  type TagTrees,
  type VaultEvents,
  type Workspace,
} from "./types";

export type TokenProvider = () => string;
export type UnauthorizedHandler = () => void;

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const out = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    out.set(k, String(v));
  }
  const s = out.toString();
  return s ? `?${s}` : "";
}

async function errorDetail(resp: Response): Promise<string> {
  const text = await resp.text().catch(() => "");
  try {
    const detail = (JSON.parse(text) as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (detail !== undefined) return JSON.stringify(detail).slice(0, 500);
  } catch {
    /* not json */
  }
  return text.slice(0, 500) || resp.statusText;
}

/** Route one SSE event of a portal chat turn to its handler. */
export function dispatchChatEvent(ev: SseEvent, h: ChatStreamHandlers): void {
  let data: Record<string, unknown> = {};
  try {
    data = JSON.parse(ev.data) as Record<string, unknown>;
  } catch {
    return;
  }
  switch (ev.event) {
    case "progress":
      h.onProgress?.(String(data.text ?? ""), data.kind ? String(data.kind) : undefined);
      break;
    case "answer":
      h.onAnswer?.(String(data.content ?? ""));
      break;
    case "message":
      h.onMessage?.({
        id: String(data.id ?? ""),
        seq: Number(data.seq ?? 0),
        role: String(data.role ?? ""),
      });
      break;
    case "files":
      h.onFiles?.(
        Array.isArray(data.paths) ? data.paths.map(String) : [],
        data.run_id ? String(data.run_id) : null,
      );
      break;
    case "job":
      h.onJob?.(data);
      break;
    case "chat":
      h.onChat?.({
        id: String(data.id ?? ""),
        title: String(data.title ?? ""),
        tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
      });
      break;
    case "error":
      h.onError?.(String(data.message ?? "Unknown error"));
      break;
    case "done":
      h.onDone?.(data);
      break;
  }
}

export class HttpApi implements PortalApi {
  readonly mock = false;

  constructor(
    private token: TokenProvider,
    private onUnauthorized: UnauthorizedHandler = () => {},
    private base = "",
  ) {}

  private headers(json: boolean): Record<string, string> {
    const h: Record<string, string> = {};
    const tok = this.token();
    if (tok) h.Authorization = `Bearer ${tok}`;
    if (json) h["Content-Type"] = "application/json";
    return h;
  }

  private async raw(method: string, path: string, body?: unknown, signal?: AbortSignal) {
    const isForm = body instanceof FormData;
    const resp = await fetch(`${this.base}${path}`, {
      method,
      headers: this.headers(body !== undefined && !isForm),
      body: body === undefined ? undefined : isForm ? body : JSON.stringify(body),
      signal,
    });
    if (resp.status === 401) this.onUnauthorized();
    if (!resp.ok) throw new HttpError(resp.status, await errorDetail(resp));
    return resp;
  }

  private async json<T>(method: string, path: string, body?: unknown, signal?: AbortSignal) {
    const resp = await this.raw(method, path, body, signal);
    return (await resp.json()) as T;
  }

  /** POST *body* and feed each SSE event to *onEvent* as it arrives. */
  async streamFetch(
    path: string,
    body: unknown,
    onEvent: (ev: SseEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const resp = await fetch(`${this.base}${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: { ...this.headers(body !== undefined), Accept: "text/event-stream" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
    if (resp.status === 401) this.onUnauthorized();
    if (!resp.ok) {
      throw new HttpError(resp.status, await errorDetail(resp));
    }
    if (!resp.body) {
      for (const ev of parseSseText(await resp.text())) onEvent(ev);
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    const parser = new SseParser();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const ev of parser.push(decoder.decode(value, { stream: true }))) onEvent(ev);
    }
    for (const ev of parser.flush()) onEvent(ev);
  }

  async health(): Promise<boolean> {
    try {
      const resp = await fetch(`${this.base}/health`);
      return resp.ok;
    } catch {
      return false;
    }
  }

  async checkToken(): Promise<void> {
    await this.raw("GET", "/v1/pawn/models");
  }

  // ── chats ──────────────────────────────────────────────────────────────

  async listChats(params: ChatListParams = {}): Promise<Chat[]> {
    const body = await this.json<{ chats: Chat[] }>("GET", `/v1/portal/chats${qs({ ...params })}`);
    return body.chats ?? [];
  }

  createChat(body: ChatCreate = {}): Promise<Chat> {
    return this.json<Chat>("POST", "/v1/portal/chats", body);
  }

  getChat(id: string): Promise<ChatDetail> {
    return this.json<ChatDetail>("GET", `/v1/portal/chats/${encodeURIComponent(id)}`);
  }

  patchChat(id: string, body: ChatPatch): Promise<Chat> {
    return this.json<Chat>("PATCH", `/v1/portal/chats/${encodeURIComponent(id)}`, body);
  }

  async deleteChat(id: string): Promise<void> {
    await this.raw("DELETE", `/v1/portal/chats/${encodeURIComponent(id)}`);
  }

  async moveChats(ids: string[], folderId: string | null): Promise<void> {
    await this.raw("POST", "/v1/portal/chats/move", { ids, folder_id: folderId });
  }

  async sendMessage(
    chatId: string,
    body: MessageCreate,
    handlers: ChatStreamHandlers,
    signal?: AbortSignal,
  ): Promise<void> {
    await this.streamFetch(
      `/v1/portal/chats/${encodeURIComponent(chatId)}/messages`,
      body,
      (ev) => dispatchChatEvent(ev, handlers),
      signal,
    );
  }

  // ── folders / tags / search ─────────────────────────────────────────────

  listFolders(): Promise<FolderTree> {
    return this.json<FolderTree>("GET", "/v1/portal/folders");
  }

  createFolder(body: FolderCreate): Promise<Folder> {
    return this.json<Folder>("POST", "/v1/portal/folders", body);
  }

  patchFolder(id: string, body: FolderPatch): Promise<Folder> {
    return this.json<Folder>("PATCH", `/v1/portal/folders/${encodeURIComponent(id)}`, body);
  }

  async deleteFolder(id: string): Promise<void> {
    await this.raw("DELETE", `/v1/portal/folders/${encodeURIComponent(id)}`);
  }

  listTags(): Promise<TagTrees> {
    return this.json<TagTrees>("GET", "/v1/portal/tags");
  }

  async renameTag(from: string, to: string): Promise<number> {
    const body = await this.json<{ changed: number }>("POST", "/v1/portal/tags/rename", {
      from,
      to,
    });
    return body.changed ?? 0;
  }

  search(q: string, limit = 30): Promise<SearchResults> {
    return this.json<SearchResults>("GET", `/v1/portal/search${qs({ q, limit })}`);
  }

  forkChat(chatId: string, messageId: string, title?: string): Promise<Fork> {
    return this.json<Fork>("POST", `/v1/portal/chats/${encodeURIComponent(chatId)}/fork`, {
      message_id: messageId,
      title: title ?? null,
    });
  }

  async listClips(params: { tag?: string; chat_id?: string } = {}): Promise<Clip[]> {
    const body = await this.json<{ clips: Clip[] }>("GET", `/v1/portal/clips${qs(params)}`);
    return body.clips ?? [];
  }

  getClip(id: string): Promise<Clip> {
    return this.json<Clip>("GET", `/v1/portal/clips/${encodeURIComponent(id)}`);
  }

  createClip(body: ClipCreate): Promise<Clip> {
    return this.json<Clip>("POST", "/v1/portal/clips", body);
  }

  patchClip(id: string, body: ClipPatch): Promise<Clip> {
    return this.json<Clip>("PATCH", `/v1/portal/clips/${encodeURIComponent(id)}`, body);
  }

  async deleteClip(id: string): Promise<void> {
    await this.raw("DELETE", `/v1/portal/clips/${encodeURIComponent(id)}`);
  }

  saveClip(id: string, path?: string): Promise<FileStat> {
    return this.json<FileStat>("POST", `/v1/portal/clips/${encodeURIComponent(id)}/save`, {
      path: path ?? null,
    });
  }

  // ── files ──────────────────────────────────────────────────────────────

  async listAnnotations(params: AnnotationListParams = {}): Promise<Annotation[]> {
    const out = new URLSearchParams();
    if (params.target_kind) out.set("target_kind", params.target_kind);
    const refs = Array.isArray(params.target_ref) ? params.target_ref : params.target_ref ? [params.target_ref] : [];
    for (const r of refs) out.append("target_ref", r);
    if (params.tag) out.set("tag", params.tag);
    const q = out.toString();
    const body = await this.json<{ annotations: Annotation[] }>("GET", `/v1/portal/annotations${q ? `?${q}` : ""}`);
    return body.annotations ?? [];
  }

  createAnnotation(body: AnnotationCreate): Promise<Annotation> {
    return this.json<Annotation>("POST", "/v1/portal/annotations", body);
  }

  patchAnnotation(id: string, body: AnnotationPatch): Promise<Annotation> {
    return this.json<Annotation>("PATCH", `/v1/portal/annotations/${encodeURIComponent(id)}`, body);
  }

  async deleteAnnotation(id: string): Promise<void> {
    await this.raw("DELETE", `/v1/portal/annotations/${encodeURIComponent(id)}`);
  }

  listFiles(prefix = ""): Promise<FileListing> {
    return this.json<FileListing>("GET", `/v1/portal/files${qs({ prefix })}`);
  }

  async listAllFiles(prefix = ""): Promise<string[]> {
    const body = await this.json<{ files: string[] }>(
      "GET",
      `/v1/portal/files${qs({ prefix, recursive: true })}`,
    );
    return body.files ?? [];
  }

  readFile(key: string): Promise<FileContent> {
    return this.json<FileContent>("GET", `/v1/portal/files/content${qs({ key })}`);
  }

  async readFileBlob(key: string): Promise<Blob> {
    const resp = await this.raw("GET", `/v1/portal/files/content${qs({ key, raw: true })}`);
    return resp.blob();
  }

  writeFile(key: string, content: string, etag: string): Promise<FileStat> {
    return this.json<FileStat>("PUT", "/v1/portal/files/content", { key, content, etag });
  }

  moveFile(src: string, dst: string, overwrite = false): Promise<FileStat> {
    return this.json<FileStat>("POST", "/v1/portal/files/move", { src, dst, overwrite });
  }

  async deleteFile(key: string): Promise<void> {
    await this.raw("DELETE", `/v1/portal/files${qs({ key })}`);
  }

  uploadFile(file: File, folder = ""): Promise<FileStat> {
    const form = new FormData();
    form.append("file", file, file.name);
    if (folder) form.append("folder", folder);
    return this.json<FileStat>("POST", "/v1/portal/files/upload", form);
  }

  getFileTags(key: string): Promise<FileTags> {
    return this.json<FileTags>("GET", `/v1/portal/files/tags${qs({ key })}`);
  }

  setFileTags(key: string, tags: string[]): Promise<FileTags> {
    return this.json<FileTags>("PUT", "/v1/portal/files/tags", { key, tags });
  }

  reindexFiles(): Promise<ReindexResult> {
    return this.json<ReindexResult>("POST", "/v1/portal/files/reindex");
  }

  // ── workspace / models / events ─────────────────────────────────────────

  getWorkspace(name = "default"): Promise<Workspace> {
    return this.json<Workspace>("GET", `/v1/portal/workspace${qs({ name })}`);
  }

  putWorkspace(layout: Record<string, unknown>, name = "default"): Promise<Workspace> {
    return this.json<Workspace>("PUT", "/v1/portal/workspace", { layout, name });
  }

  listModels(): Promise<ModelCatalog> {
    return this.json<ModelCatalog>("GET", "/v1/pawn/models");
  }

  vaultEvents(since: number, timeout: number, signal?: AbortSignal): Promise<VaultEvents> {
    return this.json<VaultEvents>(
      "GET",
      `/v1/vault/events${qs({ since, timeout })}`,
      undefined,
      signal,
    );
  }

  // ── inbox ───────────────────────────────────────────────────────────────

  inboxCounts(): Promise<InboxCounts> {
    return this.json<InboxCounts>("GET", "/v1/portal/inbox/counts");
  }

  listItems(params: ItemListParams = {}): Promise<ItemsPage> {
    return this.json<ItemsPage>("GET", `/v1/items${qs({ ...params })}`);
  }

  itemAction(id: string, action: ItemAction, arg?: string): Promise<ItemActionResult> {
    return this.json<ItemActionResult>("POST", `/v1/items/${encodeURIComponent(id)}/action`, { action, arg });
  }

  deleteItems(body: ItemsDeleteBody): Promise<{ deleted: number; receipts: string[] }> {
    return this.json("POST", "/v1/items/delete", body);
  }

  async listIdeas(status = "inbox"): Promise<Idea[]> {
    return (await this.json<{ ideas: Idea[] }>("GET", `/v1/portal/inbox/ideas${qs({ status })}`)).ideas;
  }

  async captureIdea(line: string): Promise<string> {
    return (await this.json<{ receipt: string }>("POST", "/v1/portal/inbox/ideas", { line })).receipt;
  }

  ideaAction(path: string, action: IdeaActionKind, title?: string): Promise<IdeaActionResult> {
    return this.json<IdeaActionResult>("POST", "/v1/portal/inbox/ideas/action", { path, action, title });
  }

  async listCaptureAtoms(scope: "inbox" | "all" = "inbox"): Promise<CaptureAtom[]> {
    return (await this.json<{ captures: CaptureAtom[] }>("GET", `/v1/portal/inbox/captures${qs({ scope })}`))
      .captures;
  }

  async listCaptureCollections(): Promise<CaptureCollection[]> {
    return (await this.json<{ data: CaptureCollection[] }>("GET", "/v1/captures/collections")).data;
  }

  fileCapture(body: CaptureFileBody): Promise<Record<string, unknown>> {
    return this.json("POST", "/v1/captures/file", body);
  }

  // ── settings ────────────────────────────────────────────────────────────

  getSettings(): Promise<SettingsPayload> {
    return this.json<SettingsPayload>("GET", "/v1/portal/settings");
  }

  patchSettings(values: Record<string, unknown>): Promise<SettingsPayload> {
    return this.json<SettingsPayload>("PATCH", "/v1/portal/settings", values);
  }

  resetSetting(path: string): Promise<SettingsPayload> {
    return this.json<SettingsPayload>("DELETE", `/v1/portal/settings/${encodeURIComponent(path)}`);
  }

  settingsStatus(): Promise<SettingsStatus> {
    return this.json<SettingsStatus>("GET", "/v1/portal/settings/status");
  }
}
