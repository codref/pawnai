import { Notice, TFile } from "obsidian";
import { AskJobRequest, Job, PawnClient, ServerUnreachable } from "../api";
import type PawnPlugin from "../main";
import { jobFromTaskNote, offlineJobs, queueOffline } from "../offline";
import { parseTaskNote, setApproved, setDismissed, tasksFolder } from "../tasks";

const TERMINAL = new Set(["review", "done", "blocked"]);
const DELETABLE = new Set(["review", "done", "blocked"]);
const FLUSHABLE = new Set(["done", "blocked"]);
const ACTIVE_POLL_MS = 5000;
const IDLE_POLL_MS = 30000;
const MAX_BACKOFF_MS = 60000;

export type JobsStatusScope = "all" | "active" | "review" | "done" | "blocked";

export function isActive(job: Job): boolean {
  return !TERMINAL.has(job.status);
}

export function isDeletable(job: Job): boolean {
  return DELETABLE.has(job.status);
}

export function isFlushable(job: Job): boolean {
  return FLUSHABLE.has(job.status);
}

/** Client-side job registry: server list + live events, merged with local task notes. */
export class JobStore {
  private jobs = new Map<string, Job>();
  private listeners = new Set<() => void>();
  private abort: AbortController | null = null;
  private timer: number | null = null;
  private stopped = true;
  private flushing = false;
  online = false;
  scope: JobsStatusScope = "all";
  kind = "";
  q = "";
  conversationOnly = false;

  constructor(
    private plugin: PawnPlugin,
    private client: PawnClient,
  ) {}

  start(): void {
    this.stopped = false;
    void this.refresh();
    if (this.client.canStreamEvents) void this.followLoop();
    else this.schedulePoll();

    const folderPrefix = () => `${tasksFolder(this.plugin.settings.agentRoot)}/`;
    this.plugin.registerEvent(
      this.plugin.app.vault.on("modify", (file) => {
        if (file instanceof TFile && file.path.startsWith(folderPrefix())) {
          void this.mergeTaskFile(file);
        }
      }),
    );
    this.plugin.registerEvent(
      this.plugin.app.vault.on("create", (file) => {
        if (file instanceof TFile && file.path.startsWith(folderPrefix())) {
          void this.mergeTaskFile(file);
        }
      }),
    );
  }

  stop(): void {
    this.stopped = true;
    this.abort?.abort();
    if (this.timer != null) window.clearTimeout(this.timer);
  }

  onChange(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(): void {
    for (const fn of this.listeners) fn();
  }

  setQuery(partial: {
    scope?: JobsStatusScope;
    kind?: string;
    q?: string;
    conversationOnly?: boolean;
  }): void {
    if (partial.scope !== undefined) this.scope = partial.scope;
    if (partial.kind !== undefined) this.kind = partial.kind;
    if (partial.q !== undefined) this.q = partial.q;
    if (partial.conversationOnly !== undefined) this.conversationOnly = partial.conversationOnly;
  }

  all(): Job[] {
    return [...this.jobs.values()].sort((a, b) =>
      (b.updated_at ?? b.created_at ?? b.id).localeCompare(a.updated_at ?? a.created_at ?? a.id),
    );
  }

  filtered(conversation: string | null): Job[] {
    const needle = this.q.trim().toLowerCase();
    return this.all().filter((job) => {
      if (this.conversationOnly && conversation && job.conversation !== conversation) return false;
      if (this.kind && job.kind !== this.kind) return false;
      if (this.scope === "active" && !isActive(job)) return false;
      if (this.scope === "review" && !(job.status === "review" && !job.approved)) return false;
      if (this.scope === "done" && job.status !== "done") return false;
      if (this.scope === "blocked" && job.status !== "blocked") return false;
      if (!needle) return true;
      const hay = [job.title, job.instruction, job.result, job.conversation, job.kind, job.status]
        .join("\n")
        .toLowerCase();
      return hay.includes(needle);
    });
  }

  flushableCount(): number {
    let n = 0;
    for (const j of this.jobs.values()) if (isFlushable(j)) n++;
    return n;
  }

  isFlushing(): boolean {
    return this.flushing;
  }

  get(id: string): Job | undefined {
    return this.jobs.get(id);
  }

  counts(): { active: number; review: number } {
    let active = 0;
    let review = 0;
    for (const j of this.jobs.values()) {
      if (isActive(j)) active++;
      else if (j.status === "review" && !j.approved) review++;
    }
    return { active, review };
  }

  remove(id: string): void {
    if (!this.jobs.has(id)) return;
    this.jobs.delete(id);
    this.emit();
  }

  upsert(job: Job, opts: { quiet?: boolean } = {}): void {
    if (job.status === "deleted") {
      this.remove(job.id);
      return;
    }
    const prev = this.jobs.get(job.id);
    if (prev && !prev.offline && job.offline) return; // server state wins
    // Server payloads omit `offline`; clear a sticky flag from an earlier offline card.
    const merged: Job = { ...prev, ...job };
    if (!job.offline) merged.offline = false;
    this.jobs.set(job.id, merged);
    if (!opts.quiet && prev && isActive(prev) && !isActive(job)) this.notifyDone(job);
    this.emit();
  }

  private notifyDone(job: Job): void {
    if (!this.plugin.settings.notifyOnJobDone) return;
    const verb =
      job.status === "blocked" ? "failed" : job.status === "review" ? "ready for review" : "done";
    const n = new Notice(`Pawn job ${verb}: ${job.title}`, 8000);
    n.noticeEl.addClass("pawn-notice");
    n.noticeEl.onclick = () => void this.plugin.openChat({ tab: "jobs", focusJob: job.id });
  }

  async refresh(): Promise<void> {
    try {
      const jobs = await this.client.listJobs({ limit: 100 });
      this.online = true;
      const seen = new Set(jobs.map((j) => j.id));
      for (const job of jobs) this.upsert(job, { quiet: !this.jobs.has(job.id) });
      // Drop server-backed jobs that disappeared (deleted elsewhere).
      for (const id of [...this.jobs.keys()]) {
        const cur = this.jobs.get(id);
        if (cur && !cur.offline && !seen.has(id) && isDeletable(cur)) this.jobs.delete(id);
      }
    } catch {
      this.online = false;
    }
    try {
      for (const job of await offlineJobs(this.plugin.app, this.plugin.settings.agentRoot)) {
        if (!this.jobs.has(job.id)) this.jobs.set(job.id, job);
      }
    } catch {
      /* vault scan is best effort */
    }
    this.emit();
    this.plugin.updateStatusBar();
  }

  private async mergeTaskFile(file: TFile): Promise<void> {
    const parsed = parseTaskNote(await this.plugin.app.vault.cachedRead(file), file.path);
    if (!parsed) return;
    const existing = this.jobs.get(parsed.id);
    if (existing && !existing.offline) {
      // Server-backed job: fetch fresh state instead of trusting the synced note.
      if (this.online) {
        try {
          this.upsert(await this.client.getJob(parsed.id));
        } catch {
          /* keep current */
        }
      }
      return;
    }
    this.upsert(jobFromTaskNote(parsed));
  }

  private async followLoop(): Promise<void> {
    let backoff = 2000;
    while (!this.stopped) {
      this.abort = new AbortController();
      const started = Date.now();
      try {
        await this.client.followJobEvents((job) => {
          this.online = true;
          this.upsert(job);
          this.plugin.updateStatusBar();
        }, this.abort.signal);
      } catch {
        this.online = false;
      }
      if (this.stopped) return;
      if (Date.now() - started > 30000) backoff = 2000;
      await sleep(backoff);
      backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
      await this.refresh();
    }
  }

  private schedulePoll(): void {
    if (this.stopped) return;
    const delay = this.counts().active > 0 ? ACTIVE_POLL_MS : IDLE_POLL_MS;
    this.timer = window.setTimeout(async () => {
      await this.refresh();
      this.schedulePoll();
    }, delay);
  }

  /** Submit an ask job; falls back to a task note when the server is unreachable. */
  async submitAsk(req: AskJobRequest): Promise<Job> {
    let job: Job;
    try {
      job = await this.client.createAskJob(req);
      this.online = true;
    } catch (e) {
      if (!(e instanceof ServerUnreachable)) throw e;
      this.online = false;
      job = await queueOffline(this.plugin.app, this.plugin.settings.agentRoot, req);
      new Notice("Pawn server unreachable: job saved as a task note and will run after sync.");
    }
    this.upsert(job, { quiet: true });
    this.plugin.updateStatusBar();
    if (!this.client.canStreamEvents) this.poke();
    return job;
  }

  track(job: Job): void {
    this.upsert(job, { quiet: true });
    this.plugin.updateStatusBar();
    if (!this.client.canStreamEvents) this.poke();
  }

  private poke(): void {
    if (this.timer != null) window.clearTimeout(this.timer);
    this.schedulePoll();
  }

  async approve(job: Job): Promise<void> {
    try {
      if (job.offline) throw new ServerUnreachable("offline job");
      this.upsert(await this.client.approveJob(job.id, job.result ?? undefined));
      new Notice("Approved: indexed into Pawn memory.");
    } catch (e) {
      if (!(e instanceof ServerUnreachable) || !job.task_key) {
        new Notice(`Approve failed: ${e instanceof Error ? e.message : e}`);
        return;
      }
      await setApproved(this.plugin.app, job.task_key);
      this.upsert({ ...job, approved: true, status: "done" });
      new Notice("Marked approved in the task note; Pawn indexes it after sync.");
    }
    this.plugin.updateStatusBar();
  }

  async dismiss(job: Job): Promise<void> {
    try {
      if (job.offline) throw new ServerUnreachable("offline job");
      this.upsert(await this.client.dismissJob(job.id));
      new Notice("Dismissed: closed without indexing.");
    } catch (e) {
      if (!(e instanceof ServerUnreachable) || !job.task_key) {
        new Notice(`Dismiss failed: ${e instanceof Error ? e.message : e}`);
        return;
      }
      await setDismissed(this.plugin.app, job.task_key);
      this.upsert({ ...job, approved: false, status: "done" });
      new Notice("Marked done in the task note; Pawn closes it after sync.");
    }
    this.plugin.updateStatusBar();
  }

  async cancel(job: Job): Promise<void> {
    try {
      this.upsert(await this.client.cancelJob(job.id));
    } catch (e) {
      new Notice(`Cancel failed: ${e instanceof Error ? e.message : e}`);
    }
    this.plugin.updateStatusBar();
  }

  private async trashTaskNote(job: Job): Promise<void> {
    const path = job.task_key;
    if (!path) return;
    const file = this.plugin.app.vault.getAbstractFileByPath(path);
    if (file instanceof TFile) {
      try {
        await this.plugin.app.vault.trash(file, true);
      } catch {
        /* best effort */
      }
    }
  }

  async deleteIds(ids: string[]): Promise<number> {
    let deleted = 0;
    const offlineIds: string[] = [];
    const onlineIds: string[] = [];
    for (const id of ids) {
      const job = this.jobs.get(id);
      if (!job || !isDeletable(job)) continue;
      if (job.offline) offlineIds.push(id);
      else onlineIds.push(id);
    }
    for (const id of offlineIds) {
      const job = this.jobs.get(id);
      if (job) await this.trashTaskNote(job);
      this.jobs.delete(id);
      deleted += 1;
    }
    if (onlineIds.length) {
      try {
        const result = await this.client.deleteJobs({ ids: onlineIds });
        this.online = true;
        for (const id of result.ids) {
          const job = this.jobs.get(id);
          if (job) await this.trashTaskNote(job);
          this.jobs.delete(id);
        }
        deleted += result.deleted;
      } catch (e) {
        if (!(e instanceof ServerUnreachable)) throw e;
        this.online = false;
        for (const id of onlineIds) {
          const job = this.jobs.get(id);
          if (job) await this.trashTaskNote(job);
          this.jobs.delete(id);
          deleted += 1;
        }
      }
    }
    this.emit();
    this.plugin.updateStatusBar();
    return deleted;
  }

  async flushTerminal(): Promise<number> {
    if (this.flushing) throw new Error("Flush already in progress.");
    this.flushing = true;
    this.emit();
    try {
      const localFlush = this.all().filter(isFlushable);
      if (this.online) {
        try {
          const result = await this.client.deleteJobs({ flush_terminal: true });
          for (const id of result.ids) {
            const job = this.jobs.get(id);
            if (job) await this.trashTaskNote(job);
            this.jobs.delete(id);
          }
          // Also clear any leftover offline done/blocked cards.
          for (const job of localFlush) {
            if (job.offline && this.jobs.has(job.id)) {
              await this.trashTaskNote(job);
              this.jobs.delete(job.id);
            }
          }
          this.emit();
          this.plugin.updateStatusBar();
          return result.deleted;
        } catch (e) {
          if (!(e instanceof ServerUnreachable)) throw e;
          this.online = false;
        }
      }
      return this.deleteIds(localFlush.map((j) => j.id));
    } finally {
      this.flushing = false;
      this.emit();
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => window.setTimeout(r, ms));
}
