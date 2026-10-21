import { Notice, TFile } from "obsidian";
import type { SessionSpeakerFacet, SessionSummary } from "../api";
import { HttpError, ServerUnreachable } from "../api";
import type PawnPlugin from "../main";

export type SessionsScope = "all" | "needs";

/** Fetches diarization sessions for the Sessions navigation tab. */
export class SessionsStore {
  private items: SessionSummary[] = [];
  private speakerFacets: SessionSpeakerFacet[] = [];
  private needsCount = 0;
  private listeners = new Set<() => void>();
  private stopped = true;
  private generation = 0;
  private offline = false;

  /** Primary scope: All (default) or Needs speakers. */
  scope: SessionsScope = "all";
  /** Optional named-speaker chip filter. */
  speaker = "";
  q = "";

  constructor(private plugin: PawnPlugin) {}

  start(): void {
    this.stopped = false;
    void this.refresh();
  }

  stop(): void {
    this.stopped = true;
  }

  onChange(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(): void {
    for (const fn of this.listeners) fn();
  }

  all(): SessionSummary[] {
    return this.items;
  }

  facets(): SessionSpeakerFacet[] {
    return this.speakerFacets;
  }

  /** Badge: sessions still needing speaker training (from last facets). */
  attention(): number {
    if (this.offline) {
      return this.items.filter((s) => s.needs_speakers).length;
    }
    return this.needsCount;
  }

  isOffline(): boolean {
    return this.offline;
  }

  setQuery(opts: { scope?: SessionsScope; speaker?: string; q?: string }): void {
    if (opts.scope != null) this.scope = opts.scope;
    if (opts.speaker != null) this.speaker = opts.speaker;
    if (opts.q != null) this.q = opts.q;
  }

  async refresh(): Promise<void> {
    if (this.stopped) return;
    const generation = ++this.generation;
    try {
      const page = await this.plugin.client.listSessions({
        q: this.q.trim() || undefined,
        limit: 80,
        needs_speakers: this.scope === "needs",
        speaker: this.speaker.trim() || undefined,
      });
      if (generation !== this.generation || this.stopped) return;
      this.items = page.data;
      this.speakerFacets = page.facets.speakers ?? [];
      this.needsCount = page.facets.needs_speakers ?? 0;
      this.offline = false;
      this.emit();
    } catch (e) {
      if (generation !== this.generation || this.stopped) return;
      if (e instanceof ServerUnreachable || e instanceof HttpError) {
        this.offline = true;
        this.items = await this.scanVault();
        this.speakerFacets = this.facetsFromItems(this.items);
        this.needsCount = this.items.filter((s) => s.needs_speakers).length;
        this.emit();
        return;
      }
      new Notice(e instanceof Error ? e.message : String(e), 5000);
    }
  }

  private facetsFromItems(items: SessionSummary[]): SessionSpeakerFacet[] {
    const counts = new Map<string, number>();
    for (const s of items) {
      for (const name of s.speakers ?? []) {
        if (/^SPEAKER_\d+$/.test(name)) continue;
        counts.set(name, (counts.get(name) ?? 0) + 1);
      }
    }
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 12)
      .map(([name, count]) => ({ name, count }));
  }

  /** Offline browse: vault notes with ``pawn: transcript`` frontmatter. */
  private async scanVault(): Promise<SessionSummary[]> {
    const out: SessionSummary[] = [];
    const files = this.plugin.app.vault.getMarkdownFiles();
    const speakerFilter = this.speaker.trim().toLowerCase();
    for (const file of files) {
      const cache = this.plugin.app.metadataCache.getFileCache(file);
      const fm = cache?.frontmatter;
      if (!fm || fm.pawn !== "transcript") continue;
      const sessionId = String(fm.session_id || file.basename).trim();
      if (!sessionId) continue;
      const speakers = Array.isArray(fm.speakers)
        ? fm.speakers.map(String)
        : typeof fm.speakers === "string"
          ? [fm.speakers]
          : [];
      const anon = speakers.filter((s) => /^SPEAKER_\d+$/.test(s));
      const q = this.q.trim().toLowerCase();
      if (
        q &&
        !sessionId.toLowerCase().includes(q) &&
        !file.basename.toLowerCase().includes(q) &&
        !speakers.some((s) => s.toLowerCase().includes(q))
      ) {
        continue;
      }
      const needs = anon.length > 0;
      if (this.scope === "needs" && !needs) continue;
      if (speakerFilter && !speakers.some((s) => s.toLowerCase() === speakerFilter)) {
        continue;
      }
      out.push({
        session_id: sessionId,
        title: file.basename,
        transcript_path: file.path,
        speakers,
        needs_speakers: needs,
        unmapped_count: anon.length,
        updated_at: window.moment(file.stat.mtime).toISOString(),
      });
    }
    out.sort((a, b) => String(b.updated_at).localeCompare(String(a.updated_at)));
    return out.slice(0, 80);
  }

  resolveTranscript(session: SessionSummary): TFile | null {
    const path = (session.transcript_path || "").trim();
    if (!path) return null;
    const file = this.plugin.app.vault.getAbstractFileByPath(path);
    return file instanceof TFile ? file : null;
  }
}
