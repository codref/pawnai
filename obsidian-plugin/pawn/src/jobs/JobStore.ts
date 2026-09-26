import { Notice, TFile } from "obsidian";
import { AskJobRequest, Job, PawnClient, ServerUnreachable } from "../api";
import type PawnPlugin from "../main";
import { jobFromTaskNote, offlineJobs, queueOffline } from "../offline";
import { parseTaskNote, setApproved, tasksFolder } from "../tasks";

const TERMINAL = new Set(["review", "done", "blocked"]);
const ACTIVE_POLL_MS = 5000;
const IDLE_POLL_MS = 30000;
const MAX_BACKOFF_MS = 60000;

export function isActive(job: Job): boolean {
  return !TERMINAL.has(job.status);
}

/** Client-side job registry: server list + live events, merged with local task notes. */
export class JobStore {
  private jobs = new Map<string, Job>();
  private listeners = new Set<() => void>();
  private abort: AbortController | null = null;
  private timer: number | null = null;
  private stopped = true;
  online = false;

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

  all(): Job[] {
    return [...this.jobs.values()].sort((a, b) =>
      (b.created_at ?? b.id).localeCompare(a.created_at ?? a.id),
    );
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

  upsert(job: Job, opts: { quiet?: boolean } = {}): void {
    const prev = this.jobs.get(job.id);
    if (prev && !prev.offline && job.offline) return; // server state wins
    this.jobs.set(job.id, { ...prev, ...job });
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
      for (const job of jobs) this.upsert(job, { quiet: !this.jobs.has(job.id) });
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

  async cancel(job: Job): Promise<void> {
    try {
      this.upsert(await this.client.cancelJob(job.id));
    } catch (e) {
      new Notice(`Cancel failed: ${e instanceof Error ? e.message : e}`);
    }
    this.plugin.updateStatusBar();
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => window.setTimeout(r, ms));
}
