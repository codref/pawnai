import { Platform, requestUrl, RequestUrlResponse } from "obsidian";
import type { PawnSettings } from "./settings";
import { parseSseText, SseEvent, SseParser } from "./sse";

export type JobKind = "ask" | "push_note" | "upload";
export type JobStatus =
  | "todo"
  | "queued"
  | "claimed"
  | "running"
  | "review"
  | "done"
  | "blocked";

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
}

export interface ChatRequest {
  conversation: string;
  message: string;
  active_note?: NoteContext;
  selection?: string;
  context?: NoteContext[];
  background?: boolean;
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
  note_key?: string | null;
  created_at?: string | null;
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

  async listItems(status?: string): Promise<InboxItem[]> {
    const query = status ? `?status=${encodeURIComponent(status)}` : "";
    const resp = await this.request("GET", `/v1/items${query}`);
    const body = resp.json as { items?: InboxItem[] };
    return body.items ?? [];
  }

  async itemAction(id: string, action: string, arg?: string): Promise<string> {
    const resp = await this.request("POST", `/v1/items/${encodeURIComponent(id)}/action`, {
      action,
      arg,
    });
    return String((resp.json as { receipt?: string }).receipt ?? "ok");
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
