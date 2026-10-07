import type { InboxItem } from "../api";
import { ServerUnreachable } from "../api";
import type PawnPlugin from "../main";
import { OPEN_STATUSES, PAGE_SIZE, listOpenItemNotes } from "./items";

export type ItemsSection = "items" | "ideas";

export interface ItemsQuery {
  kind: string;
  q: string;
}

/** Coworker Items backed by the API online, vault notes offline. */
export class ItemsStore {
  private items: InboxItem[] = [];
  private total = 0;
  private listeners = new Set<() => void>();
  private stopped = true;
  private generation = 0;
  private offline = false;
  kind = "";
  q = "";

  constructor(private plugin: PawnPlugin) {}

  start(): void {
    this.stopped = false;
    const refresh = () => void this.refresh();
    this.plugin.registerEvent(this.plugin.app.vault.on("create", refresh));
    this.plugin.registerEvent(this.plugin.app.vault.on("modify", refresh));
    this.plugin.registerEvent(this.plugin.app.vault.on("delete", refresh));
    this.plugin.registerEvent(this.plugin.app.vault.on("rename", refresh));
    void this.refresh();
  }

  stop(): void {
    this.stopped = true;
  }

  onChange(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  all(): InboxItem[] {
    return this.items;
  }

  count(): number {
    return this.total || this.items.length;
  }

  attention(): number {
    return this.count();
  }

  isOffline(): boolean {
    return this.offline;
  }

  hasMore(): boolean {
    return !this.offline && this.items.length < this.total;
  }

  setQuery(partial: Partial<ItemsQuery>): void {
    if (partial.kind !== undefined) this.kind = partial.kind;
    if (partial.q !== undefined) this.q = partial.q;
  }

  private notify(): void {
    for (const fn of this.listeners) fn();
    this.plugin.updateStatusBar();
  }

  async refresh(): Promise<void> {
    if (this.stopped) return;
    const generation = ++this.generation;
    try {
      const result = await this.plugin.client.listItems({
        statuses: OPEN_STATUSES.join(","),
        kind: this.kind || undefined,
        q: this.q.trim() || undefined,
        limit: PAGE_SIZE,
        offset: 0,
      });
      if (this.stopped || generation !== this.generation) return;
      this.items = result.items;
      this.total = result.total;
      this.offline = false;
    } catch (e) {
      if (!(e instanceof ServerUnreachable)) {
        // Auth/server errors: keep last good list rather than pretending offline.
        if (this.plugin.jobs?.online) return;
      }
      const itemsDir = `${this.plugin.settings.agentRoot.replace(/\/+$/, "")}/Items`;
      const items = await listOpenItemNotes(this.plugin.app, itemsDir);
      if (this.stopped || generation !== this.generation) return;
      let filtered = items;
      if (this.kind) filtered = filtered.filter((i) => i.kind === this.kind);
      if (this.q.trim()) {
        const needle = this.q.trim().toLowerCase();
        filtered = filtered.filter(
          (i) =>
            (i.text || "").toLowerCase().includes(needle) ||
            (i.thread || "").toLowerCase().includes(needle),
        );
      }
      this.items = filtered;
      this.total = filtered.length;
      this.offline = true;
    }
    this.notify();
  }

  async loadMore(): Promise<void> {
    if (this.stopped || this.offline || !this.hasMore()) return;
    const generation = this.generation;
    try {
      const result = await this.plugin.client.listItems({
        statuses: OPEN_STATUSES.join(","),
        kind: this.kind || undefined,
        q: this.q.trim() || undefined,
        limit: PAGE_SIZE,
        offset: this.items.length,
      });
      if (this.stopped || generation !== this.generation) return;
      const seen = new Set(this.items.map((i) => i.id));
      for (const item of result.items) {
        if (!seen.has(item.id)) this.items.push(item);
      }
      this.total = result.total;
      this.notify();
    } catch {
      /* keep current page */
    }
  }

  async runAction(id: string, action: string): Promise<string> {
    const item = this.items.find((i) => i.id === id || i.short_id === id);
    if (this.offline || !this.plugin.jobs?.online) {
      const noteKey = item?.note_key;
      if (!noteKey) throw new Error("Offline and no local note for this item.");
      const { setItemNoteAction } = await import("./items");
      await setItemNoteAction(this.plugin.app, noteKey, action);
      this.items = this.items.filter((i) => i.id !== item?.id);
      this.total = Math.max(0, this.total - 1);
      this.notify();
      return `Queued ${action} on note (sync will apply).`;
    }
    const receipt = await this.plugin.client.itemAction(id, action);
    this.items = this.items.filter((i) => i.id !== id && i.short_id !== id);
    this.total = Math.max(0, this.total - 1);
    this.notify();
    return receipt;
  }

  async deleteIds(ids: string[]): Promise<number> {
    if (!ids.length) return 0;
    if (this.offline || !this.plugin.jobs?.online) {
      let n = 0;
      for (const id of ids) {
        await this.runAction(id, "delete");
        n += 1;
      }
      return n;
    }
    const result = await this.plugin.client.deleteItems({ ids });
    const drop = new Set(ids);
    this.items = this.items.filter((i) => !drop.has(i.id) && !drop.has(i.short_id));
    this.total = Math.max(0, this.total - result.deleted);
    this.notify();
    return result.deleted;
  }

  async deleteAllOpen(): Promise<number> {
    if (this.offline || !this.plugin.jobs?.online) {
      const ids = this.items.map((i) => i.id);
      return this.deleteIds(ids);
    }
    const result = await this.plugin.client.deleteItems({
      all_open: true,
      kind: this.kind || undefined,
      q: this.q.trim() || undefined,
    });
    this.items = [];
    this.total = 0;
    this.notify();
    return result.deleted;
  }
}
