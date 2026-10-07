import { TAbstractFile, TFile } from "obsidian";
import type { InboxItem } from "../api";
import { ServerUnreachable } from "../api";
import type PawnPlugin from "../main";
import {
  ItemsScope,
  PAGE_SIZE,
  CLOSED_STATUSES,
  countVaultItemNotes,
  flushClosedVaultNotes,
  isItemNotePath,
  listItemNotes,
  statusesForScope,
} from "./items";

export type ItemsSection = "items" | "ideas";

export interface ItemsQuery {
  kind: string;
  q: string;
  scope: ItemsScope;
}

const DEBOUNCE_MS = 400;

/** Coworker Items backed by the API online, vault notes offline. */
export class ItemsStore {
  private items: InboxItem[] = [];
  private total = 0;
  private openTotal = 0;
  private vaultNotes = 0;
  private listeners = new Set<() => void>();
  private stopped = true;
  private generation = 0;
  private offline = false;
  private debounceTimer = 0;
  private refreshing = false;
  kind = "";
  q = "";
  scope: ItemsScope = "open";

  constructor(private plugin: PawnPlugin) {}

  private itemsDir(): string {
    return `${this.plugin.settings.agentRoot.replace(/\/+$/, "")}/Items`;
  }

  start(): void {
    this.stopped = false;
    const onVault = (file: TAbstractFile) => {
      if (!(file instanceof TFile)) return;
      if (!isItemNotePath(file.path, this.itemsDir())) return;
      this.scheduleRefresh();
    };
    this.plugin.registerEvent(this.plugin.app.vault.on("create", onVault));
    this.plugin.registerEvent(this.plugin.app.vault.on("modify", onVault));
    this.plugin.registerEvent(this.plugin.app.vault.on("delete", onVault));
    this.plugin.registerEvent(
      this.plugin.app.vault.on("rename", (file, oldPath) => {
        const dir = this.itemsDir();
        if (
          (file instanceof TFile && isItemNotePath(file.path, dir)) ||
          isItemNotePath(oldPath, dir)
        ) {
          this.scheduleRefresh();
        }
      }),
    );
    void this.refresh();
  }

  stop(): void {
    this.stopped = true;
    if (this.debounceTimer) window.clearTimeout(this.debounceTimer);
    this.debounceTimer = 0;
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

  /** Badge / status bar: open items only. */
  attention(): number {
    return this.openTotal;
  }

  vaultNoteCount(): number {
    return this.vaultNotes;
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
    if (partial.scope !== undefined) this.scope = partial.scope;
  }

  scheduleRefresh(): void {
    if (this.stopped) return;
    if (this.debounceTimer) window.clearTimeout(this.debounceTimer);
    this.debounceTimer = window.setTimeout(() => {
      this.debounceTimer = 0;
      void this.refresh();
    }, DEBOUNCE_MS);
  }

  private fingerprint(): string {
    return [
      this.offline,
      this.scope,
      this.total,
      this.openTotal,
      this.vaultNotes,
      this.items.map((i) => `${i.id}:${i.status}`).join(","),
    ].join("|");
  }

  private notify(): void {
    for (const fn of this.listeners) fn();
    this.plugin.updateStatusBar();
  }

  private listParams(offset = 0) {
    return {
      statuses: statusesForScope(this.scope),
      kind: this.kind || undefined,
      q: this.q.trim() || undefined,
      limit: PAGE_SIZE,
      offset,
    };
  }

  async refresh(): Promise<void> {
    if (this.stopped || this.refreshing) {
      if (this.refreshing) this.scheduleRefresh();
      return;
    }
    this.refreshing = true;
    const generation = ++this.generation;
    const before = this.fingerprint();
    this.vaultNotes = countVaultItemNotes(this.plugin.app, this.itemsDir());
    try {
      try {
        const result = await this.plugin.client.listItems(this.listParams(0));
        if (this.stopped || generation !== this.generation) return;
        this.items = result.items;
        this.total = result.total;
        this.offline = false;
        if (this.scope === "open") {
          this.openTotal = result.total;
        } else {
          const open = await this.plugin.client.listItems({
            statuses: statusesForScope("open"),
            limit: 1,
            offset: 0,
          });
          if (this.stopped || generation !== this.generation) return;
          this.openTotal = open.total;
        }
      } catch (e) {
        if (!(e instanceof ServerUnreachable) && this.plugin.jobs?.online) return;
        const items = await listItemNotes(this.plugin.app, this.itemsDir(), this.scope);
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
        if (this.scope === "open") {
          this.openTotal = filtered.length;
        } else {
          const openOnly = await listItemNotes(this.plugin.app, this.itemsDir(), "open");
          this.openTotal = openOnly.length;
        }
      }
      if (before !== this.fingerprint()) this.notify();
      else this.plugin.updateStatusBar();
    } finally {
      this.refreshing = false;
    }
  }

  async loadMore(): Promise<void> {
    if (this.stopped || this.offline || !this.hasMore()) return;
    const generation = this.generation;
    try {
      const result = await this.plugin.client.listItems(this.listParams(this.items.length));
      if (this.stopped || generation !== this.generation) return;
      const seen = new Set(this.items.map((i) => i.id));
      let added = false;
      for (const item of result.items) {
        if (!seen.has(item.id)) {
          this.items.push(item);
          added = true;
        }
      }
      this.total = result.total;
      if (added) this.notify();
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
      if (this.scope === "open") this.openTotal = Math.max(0, this.openTotal - 1);
      this.notify();
      return `Queued ${action} on note (sync will apply).`;
    }
    const receipt = await this.plugin.client.itemAction(id, action);
    this.items = this.items.filter((i) => i.id !== id && i.short_id !== id);
    this.total = Math.max(0, this.total - 1);
    if (this.scope === "open" || ["todo", "file", "task", "delete", "ignore"].includes(action)) {
      this.openTotal = Math.max(0, this.openTotal - 1);
    }
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
    this.openTotal = Math.max(0, this.openTotal - result.deleted);
    this.notify();
    return result.deleted;
  }

  async deleteAllMatching(): Promise<number> {
    if (this.offline || !this.plugin.jobs?.online) {
      const ids = this.items.map((i) => i.id);
      return this.deleteIds(ids);
    }
    const result = await this.plugin.client.deleteItems({
      all_open: true,
      statuses: statusesForScope(this.scope),
      kind: this.kind || undefined,
      q: this.q.trim() || undefined,
    });
    this.items = [];
    this.total = 0;
    if (this.scope === "open" || this.scope === "all") this.openTotal = 0;
    this.notify();
    return result.deleted;
  }

  /** @deprecated use deleteAllMatching */
  async deleteAllOpen(): Promise<number> {
    return this.deleteAllMatching();
  }

  /**
   * Remove closed items from the server and trash leftover closed notes in Items/.
   * Open items are left alone.
   */
  async flushClosed(): Promise<{ deleted: number; notes: number }> {
    let deleted = 0;
    if (!this.offline && this.plugin.jobs?.online) {
      const result = await this.plugin.client.deleteItems({
        all_open: true,
        statuses: CLOSED_STATUSES.join(","),
      });
      deleted = result.deleted;
    }
    const notes = await flushClosedVaultNotes(this.plugin.app, this.itemsDir());
    await this.refresh();
    return { deleted, notes };
  }

  flushableCount(): number {
    return Math.max(0, this.vaultNotes - this.openTotal);
  }
}
