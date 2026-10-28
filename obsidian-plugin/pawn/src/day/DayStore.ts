import type { PlannerDay, PlannerEvent, SessionSummary } from "../api";
import type PawnPlugin from "../main";
import { HttpError, ServerUnreachable } from "../api";
import { parseFrontmatterLocal } from "./offline";

/** Fetches one calendar day for the Day (planner) tab. */
export class DayStore {
  private day: PlannerDay | null = null;
  private dateKey = "";
  private offline = false;
  private loading = false;
  private error: string | null = null;
  private listeners = new Set<() => void>();
  /** Monotonic token so overlapping refreshes don't clobber newer results. */
  private gen = 0;

  constructor(private plugin: PawnPlugin) {
    this.dateKey = window.moment().format("YYYY-MM-DD");
  }

  getDate(): string {
    return this.dateKey;
  }

  setDate(date: string): void {
    const key = (date || "").trim() || window.moment().format("YYYY-MM-DD");
    if (key === this.dateKey) return;
    this.dateKey = key;
    // Drop stale day so we don't flash the wrong date while loading.
    this.day = null;
    this.error = null;
    void this.refresh();
  }

  shiftDays(delta: number): void {
    const next = window.moment(this.dateKey, "YYYY-MM-DD").add(delta, "days");
    this.setDate(next.format("YYYY-MM-DD"));
  }

  goToday(): void {
    this.setDate(window.moment().format("YYYY-MM-DD"));
  }

  snapshot(): PlannerDay | null {
    return this.day;
  }

  isOffline(): boolean {
    return this.offline;
  }

  isLoading(): boolean {
    return this.loading;
  }

  lastError(): string | null {
    return this.error;
  }

  onChange(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(): void {
    for (const fn of this.listeners) fn();
  }

  async refresh(): Promise<void> {
    const token = ++this.gen;
    const asked = this.dateKey;
    this.loading = true;
    this.error = null;
    this.emit();
    try {
      const page = await this.plugin.client.getPlannerDay(asked);
      if (token !== this.gen || asked !== this.dateKey) return;
      this.day = page;
      this.offline = false;
      this.error = null;
    } catch (err) {
      if (token !== this.gen || asked !== this.dateKey) return;
      if (err instanceof ServerUnreachable) {
        this.offline = true;
        this.day = await this.scanVault(asked);
        this.error = null;
      } else {
        console.warn("day refresh failed", err);
        const msg =
          err instanceof HttpError
            ? err.message
            : err instanceof Error
              ? err.message
              : String(err);
        this.error = msg || "Day load failed";
        // Unstick the UI: empty day shell instead of endless Loading.
        if (!this.day || this.day.date !== asked) {
          this.day = {
            date: asked,
            events: [],
            sessions: [],
            blocks: [],
            other_items: [],
            suggestions: [],
          };
        }
      }
    } finally {
      if (token === this.gen) {
        this.loading = false;
        this.emit();
      }
    }
  }

  /** Offline: events + transcripts for the day from the vault. */
  private async scanVault(day: string): Promise<PlannerDay> {
    const events: PlannerEvent[] = [];
    const sessions: SessionSummary[] = [];
    const files = this.plugin.app.vault.getMarkdownFiles();
    for (const file of files) {
      if (!file.path.startsWith("Pawn/Events/") && !file.path.startsWith("Pawn/Transcripts/")) {
        continue;
      }
      const cache = this.plugin.app.metadataCache.getFileCache(file);
      const fm = cache?.frontmatter ?? {};
      if (file.path.startsWith("Pawn/Events/") && fm.pawn === "event") {
        const start = String(fm.start || "");
        const eventDay = start.slice(0, 10);
        if (eventDay !== day && !file.name.startsWith(day)) continue;
        const text = await this.plugin.app.vault.cachedRead(file);
        const parsed = parseFrontmatterLocal(text);
        events.push({
          id: String(fm.id || file.basename),
          title: String(fm.title || file.basename),
          start: String(fm.start || ""),
          end: String(fm.end || ""),
          source: String(fm.source || "manual"),
          source_url: String(fm.source_url || ""),
          attendees: Array.isArray(fm.attendees) ? fm.attendees.map(String) : [],
          topics: Array.isArray(fm.topics) ? fm.topics.map(String) : [],
          session_id: String(fm.session_id || ""),
          path: file.path,
          prep: parsed.prep,
          body: parsed.body,
          day,
        });
      }
      if (file.path.startsWith("Pawn/Transcripts/") && fm.pawn === "transcript") {
        const tday = String(fm.date || "").slice(0, 10);
        if (tday !== day) continue;
        sessions.push({
          session_id: String(fm.session_id || file.basename),
          title: file.basename,
          created_at: tday,
          transcript_path: file.path,
        });
      }
    }
    const blocks = [
      ...events.map((event) => ({
        kind: "event" as const,
        sort_key: event.start || day,
        event,
        session: sessions.find((s) => s.session_id === event.session_id) ?? null,
        items: [],
      })),
      ...sessions
        .filter((s) => !events.some((e) => e.session_id === s.session_id))
        .map((session) => ({
          kind: "session" as const,
          sort_key: session.created_at || day,
          event: null,
          session,
          items: [],
        })),
    ].sort((a, b) => String(a.sort_key).localeCompare(String(b.sort_key)));
    return {
      date: day,
      events,
      sessions,
      blocks,
      other_items: [],
      suggestions: [],
    };
  }
}
