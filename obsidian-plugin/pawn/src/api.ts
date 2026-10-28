import { Platform, requestUrl, RequestUrlResponse } from "obsidian";
import type { PawnSettings } from "./settings";
import { parseSseText, SseEvent, SseParser } from "./sse";

export type JobKind = "ask" | "push_note" | "upload" | "capture_enrich";
export type JobStatus =
  | "todo"
  | "queued"
  | "claimed"
  | "running"
  | "review"
  | "done"
  | "blocked"
  | "deleted";

export interface VaultEvent {
  seq: number;
  paths: string[];
  source: string;
  run_id?: string | null;
}

export interface VaultEventsPage {
  seq: number;
  resync: boolean;
  events: VaultEvent[];
}

export interface Job {
  id: string;
  kind: JobKind;
  status: JobStatus;
  title: string;
  instruction: string;
  conversation: string;
  note_path?: string | null;
  task_key?: string | null;
  result?: string | null;
  error_code?: string | null;
  approved: boolean;
  via?: string;
  payload?: Record<string, unknown>;
  created_at?: string | null;
  updated_at?: string | null;
  /** Local only: created as a task note because the server was unreachable. */
  offline?: boolean;
}

export interface NoteContext {
  path: string;
  content?: string;
  /** false = path pointer only; server must not vault-fill the body */
  inline?: boolean;
}

export interface ChatRequest {
  conversation: string;
  message: string;
  active_note?: NoteContext;
  selection?: string;
  context?: NoteContext[];
  background?: boolean;
  /** Catalog id ``provider@model``. Omitted uses the server background default. */
  model?: string;
  /** OpenRouter reasoning effort: none, low, medium, high. */
  reasoning?: string;
  /** OpenRouter route: balanced, nitro, floor, exacto. */
  route?: string;
  images?: ChatImage[];
  /** Recaption images this conversation has already seen. */
  force_caption?: boolean;
  /** When false, the server does not read images embedded in notes. */
  include_note_images?: boolean;
}

export interface ChatImage {
  filename: string;
  media_type: string;
  data_base64: string;
  role: "question" | "context";
}

export interface ModelChoice {
  id: string;
  provider: string;
  model: string;
  reasoning?: string[];
  reasoning_default?: string;
  routes?: string[];
  route_default?: string;
  /** True when pawnai.yaml flags this catalog model as vision-ready. */
  vision?: boolean;
}

export interface ModelCatalog {
  default: string;
  background: string;
  models: ModelChoice[];
}

export interface ChatHandlers {
  onProgress?: (text: string) => void;
  onAnswer?: (content: string) => void;
  onJob?: (job: Job) => void;
  onError?: (message: string) => void;
}

export interface AskJobRequest {
  id?: string;
  instruction: string;
  conversation?: string;
  note_path?: string;
  selection?: string;
  context_paths?: string[];
  context?: string;
  /** Catalog id inherited from the chat composer. */
  model?: string;
  reasoning?: string;
  route?: string;
}

export interface InboxItem {
  id: string;
  short_id: string;
  kind: string;
  text: string;
  thread?: string | null;
  status: string;
  interrupt: boolean;
  reason?: string | null;
  quote?: string | null;
  context?: string | null;
  note_key?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  /** ISO timestamp when the user first opened the item note (vault frontmatter). */
  opened_at?: string | null;
  source_kind?: string | null;
  source_ref?: string | null;
}

export interface ListItemsParams {
  status?: string;
  statuses?: string;
  kind?: string;
  q?: string;
  source_kind?: string;
  source_ref?: string;
  limit?: number;
  offset?: number;
}

export interface SessionExtractResult {
  session_id: string;
  items: { source_ref?: string; items?: number; interrupts?: number } | null;
  people: Record<string, unknown> | null;
}

export interface ListItemsResult {
  items: InboxItem[];
  total: number;
}

export interface SessionSummary {
  session_id: string;
  title: string;
  updated_at?: string | null;
  created_at?: string | null;
  segments?: number | null;
  duration_seconds?: number | null;
  summary?: string | null;
  transcript_path?: string | null;
  speakers?: string[];
  unmapped_count?: number;
  needs_speakers?: boolean;
  speakers_trained?: boolean;
  tags?: string[];
  topics?: string[];
  sentiment?: string | null;
  has_analysis?: boolean;
}

export interface SessionSpeakerFacet {
  name: string;
  count: number;
}

export interface SessionsPage {
  data: SessionSummary[];
  facets: {
    speakers: SessionSpeakerFacet[];
    needs_speakers: number;
  };
}

export interface SessionLabel {
  local_label: string;
  duration_s: number;
  sample_text: string;
  mapped_speaker_id?: string | null;
  mapped_name?: string | null;
  ignored?: boolean;
  has_centroid: boolean;
}

export interface GallerySpeaker {
  id: string;
  display_name: string;
  enrollments: number;
}

export interface SessionDetail extends SessionSummary {
  labels: SessionLabel[];
  gallery: GallerySpeaker[];
}

export interface ListSessionsParams {
  q?: string;
  limit?: number;
  needs_speakers?: boolean;
  speaker?: string;
}

export interface PlannerEvent {
  id: string;
  title: string;
  start?: string;
  end?: string;
  source?: string;
  source_url?: string;
  attendees?: string[];
  topics?: string[];
  organizer?: string;
  location?: string;
  meeting_url?: string;
  session_id?: string;
  fingerprint?: string;
  prep_job_id?: string;
  prep_at?: string;
  path?: string;
  body?: string;
  prep?: string;
  day?: string;
  clip_count?: number;
  image_count?: number;
  has_prep?: boolean;
}

export interface PlannerSuggestion {
  event_id: string;
  event_path?: string;
  session_id: string;
  session_title?: string;
  score?: number;
}

export interface PlannerBlock {
  kind: "event" | "session";
  sort_key?: string;
  event?: PlannerEvent | null;
  session?: SessionSummary | null;
  items?: InboxItem[];
}

export interface PlannerDay {
  date: string;
  events: PlannerEvent[];
  sessions: SessionSummary[];
  blocks: PlannerBlock[];
  other_items: InboxItem[];
  suggestions: PlannerSuggestion[];
}

export interface PlannerEventCreate {
  title: string;
  start?: string;
  end?: string;
  source?: string;
  source_url?: string;
  attendees?: string[];
  topics?: string[];
  session_id?: string;
  body?: string;
  id?: string;
}

export class ServerUnreachable extends Error {}

export class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export class PawnClient {
  constructor(private getSettings: () => PawnSettings) {}

  private get base(): string {
    return this.getSettings().serverUrl.replace(/\/+$/, "");
  }

  private headers(json = true): Record<string, string> {
    const h: Record<string, string> = {};
    if (json) h["Content-Type"] = "application/json";
    const token = this.getSettings().apiToken.trim();
    if (token) h.Authorization = `Bearer ${token}`;
    return h;
  }

  private async request(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<RequestUrlResponse> {
    let resp: RequestUrlResponse;
    try {
      resp = await requestUrl({
        url: `${this.base}${path}`,
        method,
        headers: this.headers(body !== undefined),
        body: body !== undefined ? JSON.stringify(body) : undefined,
        throw: false,
      });
    } catch (e) {
      throw new ServerUnreachable(e instanceof Error ? e.message : String(e));
    }
    if (resp.status >= 400) {
      throw new HttpError(resp.status, errorDetail(resp));
    }
    return resp;
  }

  async listModels(): Promise<ModelCatalog> {
    const resp = await this.request("GET", "/v1/pawn/models");
    return resp.json as ModelCatalog;
  }

  async health(): Promise<boolean> {
    try {
      const resp = await requestUrl({ url: `${this.base}/health`, method: "GET", throw: false });
      return resp.status >= 200 && resp.status < 300;
    } catch {
      return false;
    }
  }

  // ── chat ────────────────────────────────────────────────────────────────

  /** Stream a chat turn. Desktop streams over fetch; mobile gets the full SSE body at once. */
  async chat(req: ChatRequest, handlers: ChatHandlers, signal?: AbortSignal): Promise<void> {
    const dispatch = (ev: SseEvent) => dispatchChatEvent(ev, handlers);
    if (!Platform.isMobile && typeof fetch === "function") {
      try {
        await this.streamFetch("/v1/pawn/chat", req, dispatch, signal);
        return;
      } catch (e) {
        if (signal?.aborted) return;
        if (!(e instanceof TypeError)) throw e;
        // TypeError = network/CORS failure; retry through requestUrl below.
      }
    }
    const resp = await this.request("POST", "/v1/pawn/chat", req);
    if (signal?.aborted) return;
    for (const ev of parseSseText(resp.text)) dispatch(ev);
  }

  private async streamFetch(
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
    if (!resp.ok) {
      throw new HttpError(resp.status, (await resp.text()).slice(0, 500));
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

  // ── jobs ────────────────────────────────────────────────────────────────

  async createAskJob(req: AskJobRequest): Promise<Job> {
    const resp = await this.request("POST", "/v1/jobs", { kind: "ask", ...req });
    return resp.json as Job;
  }

  async listSessions(params: ListSessionsParams = {}): Promise<SessionsPage> {
    const qs = new URLSearchParams();
    if (params.q) qs.set("q", params.q);
    if (params.limit != null) qs.set("limit", String(params.limit));
    if (params.needs_speakers) qs.set("needs_speakers", "true");
    if (params.speaker) qs.set("speaker", params.speaker);
    const query = qs.toString() ? `?${qs}` : "";
    const resp = await this.request("GET", `/v1/sessions${query}`);
    const body = resp.json as {
      data?: SessionSummary[];
      facets?: { speakers?: SessionSpeakerFacet[]; needs_speakers?: number };
    };
    return {
      data: (body.data ?? []) as SessionSummary[],
      facets: {
        speakers: body.facets?.speakers ?? [],
        needs_speakers: Number(body.facets?.needs_speakers ?? 0),
      },
    };
  }

  async getPlannerDay(date: string): Promise<PlannerDay> {
    const qs = date ? `?date=${encodeURIComponent(date)}` : "";
    const resp = await this.request("GET", `/v1/planner/day${qs}`);
    const body = resp.json as Partial<PlannerDay>;
    return {
      date: body.date || date,
      events: body.events ?? [],
      sessions: body.sessions ?? [],
      blocks: body.blocks ?? [],
      other_items: body.other_items ?? [],
      suggestions: body.suggestions ?? [],
    };
  }

  async createPlannerEvent(body: PlannerEventCreate): Promise<PlannerEvent> {
    const resp = await this.request("POST", "/v1/planner/events", body);
    return resp.json as PlannerEvent;
  }

  async linkPlannerEvent(
    eventId: string,
    sessionId: string | null,
  ): Promise<PlannerEvent> {
    const resp = await this.request(
      "POST",
      `/v1/planner/events/${encodeURIComponent(eventId)}/link`,
      { session_id: sessionId },
    );
    return resp.json as PlannerEvent;
  }

  async deletePlannerEvent(
    eventId: string,
  ): Promise<{ id: string; path: string; deleted: boolean; title?: string }> {
    const resp = await this.request(
      "DELETE",
      `/v1/planner/events/${encodeURIComponent(eventId)}`,
    );
    return resp.json as { id: string; path: string; deleted: boolean; title?: string };
  }

  async prepPlannerEvent(eventId: string): Promise<Job> {
    const resp = await this.request(
      "POST",
      `/v1/planner/events/${encodeURIComponent(eventId)}/prep`,
      {},
    );
    return resp.json as Job;
  }

  async getSession(sessionId: string): Promise<SessionDetail> {
    const resp = await this.request("GET", `/v1/sessions/${encodeURIComponent(sessionId)}`);
    return resp.json as SessionDetail;
  }

  async createSpeaker(displayName: string): Promise<GallerySpeaker & { created?: boolean }> {
    const resp = await this.request("POST", "/v1/speakers", { display_name: displayName });
    return resp.json as GallerySpeaker & { created?: boolean };
  }

  async enrollSessionSpeaker(
    sessionId: string,
    body: { speaker: string; from_label: string; force?: boolean },
  ): Promise<{
    enrollment_id: string;
    speaker_id: string;
    display_name: string;
    duration: number;
    quality_score?: number | null;
  }> {
    const resp = await this.request(
      "POST",
      `/v1/sessions/${encodeURIComponent(sessionId)}/enroll`,
      body,
    );
    return resp.json as {
      enrollment_id: string;
      speaker_id: string;
      display_name: string;
      duration: number;
      quality_score?: number | null;
    };
  }

  async reidentifySession(sessionId: string): Promise<{
    session_id: string;
    segments_updated: number;
    matched: Record<string, string>;
    unmatched: string[];
    summary: string;
  }> {
    const resp = await this.request(
      "POST",
      `/v1/sessions/${encodeURIComponent(sessionId)}/reidentify`,
      {},
    );
    return resp.json as {
      session_id: string;
      segments_updated: number;
      matched: Record<string, string>;
      unmatched: string[];
      summary: string;
    };
  }

  async relabelSession(
    sessionId: string,
    body: { from: string; to: string },
  ): Promise<{ session_id: string; from: string; to: string; summary: string }> {
    const resp = await this.request(
      "POST",
      `/v1/sessions/${encodeURIComponent(sessionId)}/relabel`,
      body,
    );
    return resp.json as { session_id: string; from: string; to: string; summary: string };
  }

  async reprocessSession(
    sessionId: string,
    body: {
      confirm: boolean;
      mode?: "rediarize" | "retranscribe";
      backend?: "nemo" | "whisper";
      language?: string | null;
      threshold?: number;
      cross_file_threshold?: number;
      margin?: number | null;
    },
  ): Promise<{
    job_id: string;
    session_id: string;
    mode: string;
    queued: boolean;
    command: string;
    result?: string;
  }> {
    const resp = await this.request(
      "POST",
      `/v1/sessions/${encodeURIComponent(sessionId)}/reprocess`,
      body,
    );
    return resp.json as {
      job_id: string;
      session_id: string;
      mode: string;
      queued: boolean;
      command: string;
      result?: string;
    };
  }

  async ignoreSessionSpeaker(
    sessionId: string,
    body: { from: string; ignore?: boolean },
  ): Promise<{
    session_id: string;
    from_label: string;
    ignored: boolean;
    needs_speakers: boolean;
    speakers_trained: boolean;
  }> {
    const resp = await this.request(
      "POST",
      `/v1/sessions/${encodeURIComponent(sessionId)}/ignore`,
      body,
    );
    return resp.json as {
      session_id: string;
      from_label: string;
      ignored: boolean;
      needs_speakers: boolean;
      speakers_trained: boolean;
    };
  }

  async setSessionSpeakersTrained(
    sessionId: string,
    body: { trained?: boolean } = {},
  ): Promise<{
    session_id: string;
    speakers_trained: boolean;
    needs_speakers: boolean;
  }> {
    const resp = await this.request(
      "POST",
      `/v1/sessions/${encodeURIComponent(sessionId)}/speakers-trained`,
      { trained: body.trained ?? true },
    );
    return resp.json as {
      session_id: string;
      speakers_trained: boolean;
      needs_speakers: boolean;
    };
  }

  async listItems(params: ListItemsParams | string = {}): Promise<ListItemsResult> {
    const opts: ListItemsParams =
      typeof params === "string" ? { status: params } : params ?? {};
    const qs = new URLSearchParams();
    if (opts.status) qs.set("status", opts.status);
    if (opts.statuses) qs.set("statuses", opts.statuses);
    if (opts.kind) qs.set("kind", opts.kind);
    if (opts.q) qs.set("q", opts.q);
    if (opts.source_kind) qs.set("source_kind", opts.source_kind);
    if (opts.source_ref) qs.set("source_ref", opts.source_ref);
    if (opts.limit != null) qs.set("limit", String(opts.limit));
    if (opts.offset != null) qs.set("offset", String(opts.offset));
    const query = qs.toString() ? `?${qs}` : "";
    const resp = await this.request("GET", `/v1/items${query}`);
    const body = resp.json as { items?: InboxItem[]; total?: number };
    const items = body.items ?? [];
    return { items, total: body.total ?? items.length };
  }

  async extractSession(
    sessionId: string,
    body: { items?: boolean; people?: boolean } = {},
  ): Promise<SessionExtractResult> {
    const resp = await this.request(
      "POST",
      `/v1/sessions/${encodeURIComponent(sessionId)}/extract`,
      {
        items: body.items !== false,
        people: Boolean(body.people),
      },
    );
    return resp.json as SessionExtractResult;
  }

  async itemAction(id: string, action: string, arg?: string): Promise<string> {
    const resp = await this.request("POST", `/v1/items/${encodeURIComponent(id)}/action`, {
      action,
      arg,
    });
    return String((resp.json as { receipt?: string }).receipt ?? "ok");
  }

  async fileCapture(body: {
    path: string;
    collection?: string;
    entity?: string;
    tags?: string[];
    ignore?: boolean;
  }): Promise<{ path: string; status: string; entity_path?: string | null }> {
    const resp = await this.request("POST", "/v1/captures/file", body);
    return resp.json as { path: string; status: string; entity_path?: string | null };
  }

  async deleteCaptureSnippet(
    snippetId: string,
    path: string,
  ): Promise<{ path: string; deleted: boolean; removed_assets?: string[] }> {
    const q = `path=${encodeURIComponent(path)}`;
    const resp = await this.request(
      "DELETE",
      `/v1/captures/snippets/${encodeURIComponent(snippetId)}?${q}`,
    );
    return resp.json as {
      path: string;
      deleted: boolean;
      removed_assets?: string[];
    };
  }

  async deleteItems(body: {
    ids?: string[];
    all_open?: boolean;
    statuses?: string;
    kind?: string;
    q?: string;
  }): Promise<{ deleted: number }> {
    const resp = await this.request("POST", "/v1/items/delete", body);
    return { deleted: Number((resp.json as { deleted?: number }).deleted ?? 0) };
  }

  async pushNote(path: string, content: string, mode: "replace" | "append"): Promise<Job> {
    const resp = await this.request("POST", "/v1/jobs", {
      kind: "push_note",
      path,
      content,
      mode,
    });
    return resp.json as Job;
  }

  async upload(params: {
    filename: string;
    data: ArrayBuffer;
    contentType?: string;
    conversation?: string;
    notePath?: string;
    index?: boolean;
  }): Promise<Job> {
    const fields: Record<string, string> = {};
    if (params.conversation) fields.conversation = params.conversation;
    if (params.notePath) fields.note_path = params.notePath;
    if (params.index) fields.index = "true";
    const { body, contentType } = buildMultipart(fields, {
      name: "file",
      filename: params.filename,
      contentType: params.contentType || "application/octet-stream",
      data: params.data,
    });
    let resp: RequestUrlResponse;
    try {
      resp = await requestUrl({
        url: `${this.base}/v1/jobs/upload`,
        method: "POST",
        headers: this.headers(false),
        contentType,
        body,
        throw: false,
      });
    } catch (e) {
      throw new ServerUnreachable(e instanceof Error ? e.message : String(e));
    }
    if (resp.status >= 400) throw new HttpError(resp.status, errorDetail(resp));
    return resp.json as Job;
  }

  async listJobs(params: { conversation?: string; limit?: number } = {}): Promise<Job[]> {
    const q = new URLSearchParams();
    if (params.conversation) q.set("conversation", params.conversation);
    q.set("limit", String(params.limit ?? 50));
    const resp = await this.request("GET", `/v1/jobs?${q.toString()}`);
    return ((resp.json as { data?: Job[] }).data ?? []) as Job[];
  }

  async getJob(id: string): Promise<Job> {
    const resp = await this.request("GET", `/v1/jobs/${encodeURIComponent(id)}`);
    return resp.json as Job;
  }

  async approveJob(id: string, result?: string): Promise<Job> {
    const resp = await this.request(
      "POST",
      `/v1/jobs/${encodeURIComponent(id)}/approve`,
      result != null ? { result } : {},
    );
    return resp.json as Job;
  }

  async cancelJob(id: string): Promise<Job> {
    const resp = await this.request("POST", `/v1/jobs/${encodeURIComponent(id)}/cancel`, {});
    return resp.json as Job;
  }

  async dismissJob(id: string): Promise<Job> {
    const resp = await this.request("POST", `/v1/jobs/${encodeURIComponent(id)}/dismiss`, {});
    return resp.json as Job;
  }

  async deleteJob(id: string): Promise<{ id: string; deleted: boolean }> {
    const resp = await this.request("DELETE", `/v1/jobs/${encodeURIComponent(id)}`);
    return resp.json as { id: string; deleted: boolean };
  }

  async deleteJobs(body: {
    ids?: string[];
    flush_terminal?: boolean;
  }): Promise<{ deleted: number; ids: string[]; skipped: string[] }> {
    const resp = await this.request("POST", "/v1/jobs/delete", body);
    return resp.json as { deleted: number; ids: string[]; skipped: string[] };
  }

  /** Long-poll vault writes. Empty ``events`` means the timeout elapsed. */
  async waitForVaultEvents(since: number, timeout = 25): Promise<VaultEventsPage> {
    const q = new URLSearchParams({ since: String(since), timeout: String(timeout) });
    const resp = await this.request("GET", `/v1/vault/events?${q.toString()}`);
    const body = resp.json as Partial<VaultEventsPage>;
    return {
      seq: Number(body.seq ?? since),
      resync: Boolean(body.resync),
      events: Array.isArray(body.events) ? body.events : [],
    };
  }

  /** Desktop only: follow /v1/jobs/events until aborted. Resolves when the stream ends. */
  async followJobEvents(onJob: (job: Job) => void, signal: AbortSignal): Promise<void> {
    await this.streamFetch(
      "/v1/jobs/events",
      undefined,
      (ev) => {
        if (ev.event !== "job") return;
        try {
          onJob(JSON.parse(ev.data) as Job);
        } catch {
          /* ignore malformed event */
        }
      },
      signal,
    );
  }

  get canStreamEvents(): boolean {
    return !Platform.isMobile && typeof fetch === "function";
  }
}

function dispatchChatEvent(ev: SseEvent, h: ChatHandlers): void {
  let data: Record<string, unknown> = {};
  try {
    data = JSON.parse(ev.data) as Record<string, unknown>;
  } catch {
    return;
  }
  switch (ev.event) {
    case "progress":
      h.onProgress?.(String(data.text ?? ""));
      break;
    case "answer":
      h.onAnswer?.(String(data.content ?? ""));
      break;
    case "job":
      h.onJob?.(data as unknown as Job);
      break;
    case "error":
      h.onError?.(String(data.message ?? "Unknown error"));
      break;
  }
}

function errorDetail(resp: RequestUrlResponse): string {
  try {
    const detail = (resp.json as { detail?: unknown })?.detail;
    if (typeof detail === "string") return `HTTP ${resp.status}: ${detail}`;
  } catch {
    /* not json */
  }
  const text = typeof resp.text === "string" ? resp.text : "";
  return `HTTP ${resp.status}: ${text.slice(0, 300)}`;
}

function buildMultipart(
  fields: Record<string, string>,
  file: { name: string; filename: string; contentType: string; data: ArrayBuffer },
): { body: ArrayBuffer; contentType: string } {
  const boundary = `----pawn${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`;
  const enc = new TextEncoder();
  const parts: Uint8Array[] = [];
  for (const [k, v] of Object.entries(fields)) {
    parts.push(
      enc.encode(`--${boundary}\r\nContent-Disposition: form-data; name="${k}"\r\n\r\n${v}\r\n`),
    );
  }
  const safeName = file.filename.replace(/"/g, "_");
  parts.push(
    enc.encode(
      `--${boundary}\r\nContent-Disposition: form-data; name="${file.name}"; ` +
        `filename="${safeName}"\r\nContent-Type: ${file.contentType}\r\n\r\n`,
    ),
  );
  parts.push(new Uint8Array(file.data));
  parts.push(enc.encode(`\r\n--${boundary}--\r\n`));
  const total = parts.reduce((n, p) => n + p.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.byteLength;
  }
  return { body: out.buffer, contentType: `multipart/form-data; boundary=${boundary}` };
}
