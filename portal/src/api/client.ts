// HTTP client for pawn-server. streamFetch is ported from
// obsidian-plugin/pawn/src/api.ts (desktop fetch path, no requestUrl fallback).

import { SseParser, parseSseText, type SseEvent } from "./sse";
import {
  HttpError,
  type Annotation,
  type Board,
  type BoardCreate,
  type BoardPatch,
  type BoardProposal,
  type BrainEdge,
  type BrainGraph,
  type BrainGraphParams,
  type BrainLassoBody,
  type BrainLassoResult,
  type BrainLayout,
  type BrainNode,
  type BrainTieResult,
  type EmergenceStatus,
  type EmergenceSubject,
  type EmergenceSubjectDetail,
  type AssistantCommand,
  type AssistantCommandResult,
  type AssistantKind,
  type AssistantRegistration,
  type AssistantSummary,
  type AssistantsOverview,
  type WakeModel,
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
  type SchedulerDiagnostics,
  type ConfigDiagnostics,
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
  type BulkReidentifyBody,
  type GallerySpeaker,
  type Job,
  type ListSessionsParams,
  type PlannerDay,
  type PlannerEvent,
  type PlannerEventCreate,
  type PlannerEventPatch,
  type PlannerRange,
  type SessionAudioUrls,
  type SessionDetail,
  type SessionEnrollBody,
  type SessionExtractResult,
  type SessionReprocessBody,
  type SessionTranscript,
  type SessionsPage,
  type SpeakerDetail,
  type SpeakerMergeResult,
  type SpeakerPatch,
  type SpeakerPeopleResult,
  type SpeakerSummary,
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
    case "board":
      h.onBoard?.({
        board: String(data.board ?? ""),
        proposals: Array.isArray(data.proposals) ? (data.proposals as BoardProposal[]) : [],
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

  settingsScheduler(): Promise<SchedulerDiagnostics> {
    return this.json<SchedulerDiagnostics>("GET", "/v1/portal/settings/scheduler");
  }

  settingsConfig(): Promise<ConfigDiagnostics> {
    return this.json<ConfigDiagnostics>("GET", "/v1/portal/settings/config");
  }

  // ── sessions / day / speakers ───────────────────────────────────────────

  listSessions(params: ListSessionsParams = {}): Promise<SessionsPage> {
    return this.json<SessionsPage>("GET", `/v1/sessions${qs({ ...params })}`);
  }

  getSession(id: string): Promise<SessionDetail> {
    return this.json<SessionDetail>("GET", `/v1/sessions/${encodeURIComponent(id)}`);
  }

  sessionTranscript(id: string, words = false): Promise<SessionTranscript> {
    return this.json<SessionTranscript>(
      "GET",
      `/v1/sessions/${encodeURIComponent(id)}/transcript${qs({ words: words ? 1 : undefined })}`,
    );
  }

  sessionAudioUrl(id: string): Promise<SessionAudioUrls> {
    return this.json<SessionAudioUrls>("POST", `/v1/sessions/${encodeURIComponent(id)}/audio-url`);
  }

  async sessionAudio(id: string, index: number, range?: string): Promise<Response> {
    const headers = this.headers(false);
    if (range) headers.Range = range;
    const resp = await fetch(
      `${this.base}/v1/sessions/${encodeURIComponent(id)}/audio/${index}`,
      { headers },
    );
    if (resp.status === 401) this.onUnauthorized();
    if (resp.status !== 200 && resp.status !== 206) {
      throw new HttpError(resp.status, await errorDetail(resp.clone()));
    }
    return resp;
  }

  createSpeaker(displayName: string): Promise<GallerySpeaker> {
    return this.json<GallerySpeaker>("POST", "/v1/speakers", { display_name: displayName });
  }

  enrollSession(id: string, body: SessionEnrollBody): Promise<Record<string, unknown>> {
    return this.json("POST", `/v1/sessions/${encodeURIComponent(id)}/enroll`, body);
  }

  reidentifySession(id: string): Promise<Record<string, unknown>> {
    return this.json("POST", `/v1/sessions/${encodeURIComponent(id)}/reidentify`);
  }

  relabelSession(id: string, body: { from: string; to: string }): Promise<Record<string, unknown>> {
    return this.json("POST", `/v1/sessions/${encodeURIComponent(id)}/relabel`, body);
  }

  reprocessSession(id: string, body: SessionReprocessBody): Promise<Record<string, unknown>> {
    return this.json("POST", `/v1/sessions/${encodeURIComponent(id)}/reprocess`, body);
  }

  ignoreSessionSpeaker(id: string, body: { from: string; ignore?: boolean }): Promise<Record<string, unknown>> {
    return this.json("POST", `/v1/sessions/${encodeURIComponent(id)}/ignore`, body);
  }

  setSessionTrained(id: string, trained = true): Promise<Record<string, unknown>> {
    return this.json("POST", `/v1/sessions/${encodeURIComponent(id)}/speakers-trained`, { trained });
  }

  extractSession(id: string, body: { items?: boolean; people?: boolean } = {}): Promise<SessionExtractResult> {
    return this.json("POST", `/v1/sessions/${encodeURIComponent(id)}/extract`, body);
  }

  reidentifySessions(body: BulkReidentifyBody = {}): Promise<{ job_id: string }> {
    return this.json("POST", "/v1/sessions/reidentify", body);
  }

  plannerDay(date = ""): Promise<PlannerDay> {
    return this.json<PlannerDay>("GET", `/v1/planner/day${qs({ date })}`);
  }

  plannerRange(from: string, to: string): Promise<PlannerRange> {
    return this.json<PlannerRange>("GET", `/v1/planner/range${qs({ from, to })}`);
  }

  async listPlannerEvents(from = "", to = ""): Promise<PlannerEvent[]> {
    const body = await this.json<{ data: PlannerEvent[] }>("GET", `/v1/planner/events${qs({ from, to })}`);
    return body.data ?? [];
  }

  upsertPlannerEvent(body: PlannerEventCreate): Promise<PlannerEvent> {
    return this.json<PlannerEvent>("POST", "/v1/planner/events", body);
  }

  patchPlannerEvent(id: string, body: PlannerEventPatch): Promise<PlannerEvent> {
    return this.json<PlannerEvent>("PATCH", `/v1/planner/events/${encodeURIComponent(id)}`, body);
  }

  linkPlannerEvent(id: string, sessionId: string | null): Promise<PlannerEvent> {
    return this.json<PlannerEvent>("POST", `/v1/planner/events/${encodeURIComponent(id)}/link`, {
      session_id: sessionId,
    });
  }

  deletePlannerEvent(id: string): Promise<{ deleted: boolean }> {
    return this.json("DELETE", `/v1/planner/events/${encodeURIComponent(id)}`);
  }

  prepPlannerEvent(id: string): Promise<Job> {
    return this.json<Job>("POST", `/v1/planner/events/${encodeURIComponent(id)}/prep`);
  }

  async listSpeakers(params: { q?: string; include_inactive?: boolean } = {}): Promise<SpeakerSummary[]> {
    const body = await this.json<{ data: SpeakerSummary[] }>("GET", `/v1/speakers${qs({ ...params })}`);
    return body.data ?? [];
  }

  getSpeaker(id: string): Promise<SpeakerDetail> {
    return this.json<SpeakerDetail>("GET", `/v1/speakers/${encodeURIComponent(id)}`);
  }

  patchSpeaker(
    id: string,
    body: SpeakerPatch,
  ): Promise<SpeakerDetail & { referenced_sessions?: number; job_id?: string | null }> {
    return this.json("PATCH", `/v1/speakers/${encodeURIComponent(id)}`, body);
  }

  deactivateSpeaker(id: string): Promise<SpeakerSummary> {
    return this.json<SpeakerSummary>("DELETE", `/v1/speakers/${encodeURIComponent(id)}`);
  }

  async deleteEnrollment(speakerId: string, enrollmentId: string): Promise<void> {
    await this.raw(
      "DELETE",
      `/v1/speakers/${encodeURIComponent(speakerId)}/enrollments/${encodeURIComponent(enrollmentId)}`,
    );
  }

  mergeSpeakers(id: string, into: string, reidentify = false): Promise<SpeakerMergeResult> {
    return this.json("POST", `/v1/speakers/${encodeURIComponent(id)}/merge`, { into, reidentify });
  }

  ensureSpeakerNote(id: string): Promise<SpeakerPeopleResult> {
    return this.json("POST", `/v1/speakers/${encodeURIComponent(id)}/people`);
  }

  assistantsOverview(): Promise<AssistantsOverview> {
    return this.json<AssistantsOverview>("GET", "/v1/portal/assistants");
  }

  registerAssistant(name: string, kind: AssistantKind): Promise<AssistantRegistration> {
    return this.json("POST", "/v1/devices/register", { name, kind });
  }

  renameAssistant(id: string, name: string): Promise<AssistantSummary> {
    return this.json("PUT", `/v1/devices/${encodeURIComponent(id)}`, { name });
  }

  revokeAssistant(id: string): Promise<{ deleted: string; events_removed: number }> {
    return this.json("DELETE", `/v1/devices/${encodeURIComponent(id)}`);
  }

  runAssistantCommand(id: string, tool: string, args: Record<string, string>): Promise<AssistantCommandResult> {
    return this.json("POST", `/v1/devices/${encodeURIComponent(id)}/commands`, { tool, args, timeout_s: 30 });
  }

  async assistantCommands(id: string, limit = 50): Promise<AssistantCommand[]> {
    const body = await this.json<{ commands: AssistantCommand[] }>(
      "GET",
      `/v1/portal/assistants/${encodeURIComponent(id)}/commands${qs({ limit })}`,
    );
    return body.commands ?? [];
  }

  publishWakeModel(file: File, opts: { version?: string; notes?: string } = {}): Promise<WakeModel> {
    const form = new FormData();
    form.append("file", file, file.name);
    if (opts.version) form.append("version", opts.version);
    if (opts.notes) form.append("notes", opts.notes);
    return this.json("POST", "/v1/assistants/wake/models", form);
  }

  activateWakeModel(version: string): Promise<WakeModel> {
    return this.json("POST", `/v1/assistants/wake/models/${encodeURIComponent(version)}/activate`);
  }

  brainmap(board: string, params: BrainGraphParams = {}): Promise<BrainGraph> {
    return this.json<BrainGraph>("GET", `/v1/portal/brainmaps/${encodeURIComponent(board)}${qs({ ...params })}`);
  }

  brainmapSpill(board: string, subjectId: string): Promise<{ nodes: BrainNode[]; edges: BrainEdge[] }> {
    return this.json(
      "GET",
      `/v1/portal/brainmaps/${encodeURIComponent(board)}/spill/${encodeURIComponent(subjectId)}`,
    );
  }

  brainmapTie(board: string, from: string, to: string, role: string): Promise<BrainTieResult> {
    return this.json<BrainTieResult>("POST", `/v1/portal/brainmaps/${encodeURIComponent(board)}/strings`, {
      from,
      to,
      role,
    });
  }

  brainmapCut(board: string, edgeId: string): Promise<{ receipt: string }> {
    return this.json(
      "DELETE",
      `/v1/portal/brainmaps/${encodeURIComponent(board)}/strings/${encodeURIComponent(edgeId)}`,
    );
  }

  brainmapSuggestion(
    board: string,
    id: string,
    verdict: "confirm" | "snip",
  ): Promise<{ edge?: BrainEdge | null; receipt: string }> {
    return this.json(
      "POST",
      `/v1/portal/brainmaps/${encodeURIComponent(board)}/suggestions/${encodeURIComponent(id)}`,
      { verdict },
    );
  }

  brainmapLasso(board: string, body: BrainLassoBody): Promise<BrainLassoResult> {
    return this.json<BrainLassoResult>("POST", `/v1/portal/brainmaps/${encodeURIComponent(board)}/lasso`, body);
  }

  getBrainmapLayout(board: string): Promise<BrainLayout> {
    return this.json<BrainLayout>("GET", `/v1/portal/brainmaps/${encodeURIComponent(board)}/layout`);
  }

  putBrainmapLayout(board: string, layout: BrainLayout): Promise<BrainLayout> {
    return this.json<BrainLayout>("PUT", `/v1/portal/brainmaps/${encodeURIComponent(board)}/layout`, layout);
  }

  async listBoards(archived = false): Promise<Board[]> {
    const res = await this.json<{ boards: Board[] }>("GET", `/v1/portal/boards${qs({ archived: archived || undefined })}`);
    return res.boards;
  }

  createBoard(body: BoardCreate): Promise<Board> {
    return this.json<Board>("POST", "/v1/portal/boards", body);
  }

  getBoard(name: string): Promise<Board> {
    return this.json<Board>("GET", `/v1/portal/boards/${encodeURIComponent(name)}`);
  }

  patchBoard(name: string, patch: BoardPatch): Promise<Board> {
    return this.json<Board>("PATCH", `/v1/portal/boards/${encodeURIComponent(name)}`, patch);
  }

  deleteBoard(name: string): Promise<{ deleted: string }> {
    return this.json("DELETE", `/v1/portal/boards/${encodeURIComponent(name)}`);
  }

  pinToBoard(name: string, nodeIds: string[]): Promise<Board> {
    return this.json<Board>("POST", `/v1/portal/boards/${encodeURIComponent(name)}/pins`, { node_ids: nodeIds });
  }

  removeFromBoard(name: string, nodeIds: string[]): Promise<Board> {
    return this.json<Board>("POST", `/v1/portal/boards/${encodeURIComponent(name)}/remove`, { node_ids: nodeIds });
  }

  boardSeen(name: string): Promise<Board> {
    return this.json<Board>("POST", `/v1/portal/boards/${encodeURIComponent(name)}/seen`);
  }

  boardChat(name: string): Promise<Chat> {
    return this.json<Chat>("POST", `/v1/portal/boards/${encodeURIComponent(name)}/chat`);
  }

  decideBoardProposal(
    name: string,
    id: string,
    verdict: "accept" | "reject",
  ): Promise<{ receipt: string; proposal: BoardProposal }> {
    return this.json(
      "POST",
      `/v1/portal/boards/${encodeURIComponent(name)}/proposals/${encodeURIComponent(id)}`,
      { verdict },
    );
  }

  emergenceSubject(id: string): Promise<EmergenceSubjectDetail> {
    return this.json<EmergenceSubjectDetail>("GET", `/v1/emergence/subjects/${encodeURIComponent(id)}`);
  }

  emergenceStatus(id: string, status: EmergenceStatus): Promise<{ subject: EmergenceSubject }> {
    return this.json("POST", `/v1/emergence/subjects/${encodeURIComponent(id)}/status`, {
      status,
      surface: "brainmap",
    });
  }

  emergenceCouncil(id: string): Promise<{ started: string }> {
    return this.json("POST", `/v1/emergence/subjects/${encodeURIComponent(id)}/council`);
  }

  getJob(id: string): Promise<Job> {
    return this.json<Job>("GET", `/v1/jobs/${encodeURIComponent(id)}`);
  }

  jobEvents(onJob: (job: Job) => void, signal?: AbortSignal): Promise<void> {
    return this.streamFetch(
      "/v1/jobs/events",
      undefined,
      (ev) => {
        if (ev.event !== "job") return;
        try {
          onJob(JSON.parse(ev.data) as Job);
        } catch {
          /* keep-alive or a non-json frame */
        }
      },
      signal,
    );
  }
}
