import { TAbstractFile, TFile } from "obsidian";
import type { InboxItem } from "../api";
import { ServerUnreachable } from "../api";
import type PawnPlugin from "../main";
import {
  ItemsScope,
  PAGE_SIZE,
  CLOSED_STATUSES,
  countVaultItemNotes,
  flushFlushableVaultNotes,
  isItemNotePath,
  listItemNotes,
  listVaultItemRefs,
  readOpenedAt,
  setItemNoteOpenedAt,
  statusesForScope,
} from "./items";

export type ItemsSection = "items" | "ideas" | "captures";

export interface ItemsQuery {
  kind: string;
  q: string;
  scope: ItemsScope;
}

const DEBOUNCE_MS = 400;
const SUPPRESS_NOTE_MS = 1200;

/** Coworker Items backed by the API online, vault notes offline. */
export class ItemsStore {
  private items: InboxItem[] = [];
  private total = 0;
  private openTotal = 0;
  private vaultNotes = 0;
  private flushable = 0;
  private listeners = new Set<() => void>();
  private stopped = true;
  private generation = 0;
  private offline = false;
  private debounceTimer = 0;
  private refreshing = false;
  private flushing = false;
  private flushLabel = "";
  /** note_key → expiry (ms) — skip vault-driven refresh after local triage. */
  private suppressUntil = new Map<string, number>();
  /** Instant unread overlay until vault metadata catches up. */
  private openedAtById = new Map<string, string>();
  kind = "";
  q = "";
  scope: ItemsScope = "open";

  constructor(private plugin: PawnPlugin) {}

  private itemsDir(): string {
    return `${this.plugin.settings.agentRoot.replace(/\/+$/, "")}/Items`;
  }

  private suppressNoteKey(noteKey: string | null | undefined): void {
    if (!noteKey) return;
    this.suppressUntil.set(noteKey, Date.now() + SUPPRESS_NOTE_MS);
  }

  private isSuppressed(path: string): boolean {
    const until = this.suppressUntil.get(path);
    if (until == null) return false;
    if (Date.now() >= until) {
      this.suppressUntil.delete(path);
      return false;
    }
    return true;
  }

  start(): void {
    this.stopped = false;
    const onVault = (file: TAbstractFile) => {
      if (!(file instanceof TFile)) return;
      if (!isItemNotePath(file.path, this.itemsDir())) return;
      if (this.isSuppressed(file.path)) return;
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
          if (
            (file instanceof TFile && this.isSuppressed(file.path)) ||
            this.isSuppressed(oldPath)
          ) {
            return;
          }
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
      this.flushable,
      this.items.map((i) => `${i.id}:${i.status}:${i.opened_at || ""}`).join(","),
    ].join("|");
  }

  private mergeOpenedAt(items: InboxItem[]): InboxItem[] {
    return items.map((item) => {
      const overlay = this.openedAtById.get(item.id) || this.openedAtById.get(item.short_id);
      const fromVault = readOpenedAt(this.plugin.app, item.note_key);
      const opened = overlay || item.opened_at || fromVault || null;
      if (opened && opened !== item.opened_at) {
        return { ...item, opened_at: opened };
      }
      if (opened) this.openedAtById.set(item.id, opened);
      return item.opened_at ? item : { ...item, opened_at: opened };
    });
  }

  /** Mark an item opened: overlay first, then write ``opened_at`` on the note. */
  markOpened(item: InboxItem): void {
    if (item.opened_at || this.openedAtById.has(item.id)) return;
    const iso = window.moment().toISOString();
    this.openedAtById.set(item.id, iso);
    if (item.short_id) this.openedAtById.set(item.short_id, iso);
    this.items = this.items.map((row) =>
      row.id === item.id || row.short_id === item.id ? { ...row, opened_at: iso } : row,
    );
    this.notify();
    const noteKey = item.note_key;
    if (!noteKey) return;
    this.suppressNoteKey(noteKey);
    void setItemNoteOpenedAt(this.plugin.app, noteKey, iso).catch(() => {
      /* keep overlay even if write fails */
    });
  }

  /** Live open ids / short_ids / note paths that must never be flushed. */
  private async collectOpenKeepSet(): Promise<Set<string>> {
    const keep = new Set<string>();
    if (this.offline || !this.plugin.jobs?.online) {
      const open = await listItemNotes(this.plugin.app, this.itemsDir(), "open");
      for (const item of open) {
        keep.add(item.id);
        if (item.short_id) keep.add(item.short_id);
        if (item.note_key) keep.add(item.note_key);
      }
      return keep;
    }
    let offset = 0;
    for (;;) {
      const page = await this.plugin.client.listItems({
        statuses: statusesForScope("open"),
        limit: PAGE_SIZE,
        offset,
      });
      for (const item of page.items) {
        keep.add(item.id);
        if (item.short_id) keep.add(item.short_id);
        if (item.note_key) keep.add(item.note_key);
      }
      offset += page.items.length;
      if (offset >= page.total || page.items.length === 0) break;
    }
    return keep;
  }

  private async recountFlushable(keep?: Set<string>): Promise<void> {
    this.vaultNotes = countVaultItemNotes(this.plugin.app, this.itemsDir());
    const live = keep ?? (await this.collectOpenKeepSet());
    const refs = await listVaultItemRefs(this.plugin.app, this.itemsDir());
    let n = 0;
    for (const ref of refs) {
      const kept =
        (ref.id && live.has(ref.id)) ||
        (ref.short_id && live.has(ref.short_id)) ||
        live.has(ref.path);
      if (!kept) n += 1;
    }
    this.flushable = n;
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

  /**
   * Reload the list. Pass ``{ flushable: false }`` after triage to skip the
   * expensive open-id walk; flush button / full refresh still recount.
   */
  async refresh(opts?: { flushable?: boolean }): Promise<void> {
    if (this.stopped || this.refreshing) {
      if (this.refreshing) this.scheduleRefresh();
      return;
    }
    const wantFlushable = opts?.flushable !== false;
    this.refreshing = true;
    const generation = ++this.generation;
    const before = this.fingerprint();
    try {
      try {
        const result = await this.plugin.client.listItems(this.listParams(0));
        if (this.stopped || generation !== this.generation) return;
        this.items = this.mergeOpenedAt(result.items);
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
        if (wantFlushable) {
          const keep = await this.collectOpenKeepSet();
          if (this.stopped || generation !== this.generation) return;
          await this.recountFlushable(keep);
        } else {
          this.vaultNotes = countVaultItemNotes(this.plugin.app, this.itemsDir());
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
              (i.thread || "").toLowerCase().includes(needle) ||
              (i.context || "").toLowerCase().includes(needle),
          );
        }
        this.items = this.mergeOpenedAt(filtered);
        this.total = filtered.length;
        this.offline = true;
        if (this.scope === "open") {
          this.openTotal = filtered.length;
        } else {
          const openOnly = await listItemNotes(this.plugin.app, this.itemsDir(), "open");
          this.openTotal = openOnly.length;
        }
        if (wantFlushable) {
          const keep = await this.collectOpenKeepSet();
          await this.recountFlushable(keep);
        } else {
          this.vaultNotes = countVaultItemNotes(this.plugin.app, this.itemsDir());
        }
      }
      if (before !== this.fingerprint()) this.notify();
      else this.plugin.updateStatusBar();
    } finally {
      this.refreshing = false;
    }
  }

  private isOpenStatus(item: InboxItem): boolean {
    return ["new", "notified", "snoozed"].includes((item.status || "").toLowerCase());
  }

  /** Drop one item from the local list and adjust counters (optimistic UI). */
  private dropLocal(id: string): { item: InboxItem; index: number } | null {
    const index = this.items.findIndex((i) => i.id === id || i.short_id === id);
    if (index < 0) return null;
    const item = this.items[index];
    this.items = this.items.filter((_, i) => i !== index);
    this.total = Math.max(0, this.total - 1);
    if (this.isOpenStatus(item)) this.openTotal = Math.max(0, this.openTotal - 1);
    this.suppressNoteKey(item.note_key);
    return { item, index };
  }

  private restoreLocal(item: InboxItem, index: number): void {
    const exists = this.items.some((i) => i.id === item.id);
    if (exists) return;
    const at = Math.min(Math.max(0, index), this.items.length);
    this.items = [...this.items.slice(0, at), item, ...this.items.slice(at)];
    this.total += 1;
    if (this.isOpenStatus(item)) this.openTotal += 1;
  }

  async loadMore(): Promise<void> {
    if (this.stopped || this.offline || !this.hasMore()) return;
    const generation = this.generation;
    try {
      const result = await this.plugin.client.listItems(this.listParams(this.items.length));
      if (this.stopped || generation !== this.generation) return;
      const seen = new Set(this.items.map((i) => i.id));
      let added = false;
      for (const item of this.mergeOpenedAt(result.items)) {
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

  /**
   * Trash the local item note after the server deleted it from S3.
   * Without this, Sync Engine can re-upload the still-open vault file.
   */
  private async trashNoteLocally(noteKey: string | null | undefined): Promise<void> {
    if (!noteKey) return;
    const file = this.plugin.app.vault.getAbstractFileByPath(noteKey);
    if (!(file instanceof TFile)) return;
    this.suppressNoteKey(noteKey);
    try {
      await this.plugin.app.fileManager.trashFile(file);
    } catch {
      /* already gone */
    }
  }

  async runAction(id: string, action: string, arg?: string): Promise<string> {
    const removed = this.dropLocal(id);
    if (removed) this.notify();
    const item = removed?.item;
    try {
      if (this.offline || !this.plugin.jobs?.online) {
        const noteKey = item?.note_key;
        if (!noteKey) throw new Error("Offline and no local note for this item.");
        const { setItemNoteAction } = await import("./items");
        await setItemNoteAction(this.plugin.app, noteKey, action, arg);
        return `Queued ${action} on note (sync will apply).`;
      }
      const receipt = await this.plugin.client.itemAction(id, action, arg);
      if (action === "delete") await this.trashNoteLocally(item?.note_key);
      return receipt;
    } catch (err) {
      if (removed) {
        this.restoreLocal(removed.item, removed.index);
        this.notify();
      }
      throw err;
    }
  }

  async deleteIds(ids: string[]): Promise<number> {
    if (!ids.length) return 0;
    const snapshot = ids.map((id) => this.dropLocal(id)).filter(Boolean) as {
      item: InboxItem;
      index: number;
    }[];
    if (snapshot.length) this.notify();
    try {
      if (this.offline || !this.plugin.jobs?.online) {
        let n = 0;
        for (const { item } of snapshot) {
          const noteKey = item.note_key;
          if (!noteKey) continue;
          const { setItemNoteAction } = await import("./items");
          await setItemNoteAction(this.plugin.app, noteKey, "delete");
          n += 1;
        }
        return n;
      }
      const result = await this.plugin.client.deleteItems({ ids });
      for (const { item } of snapshot) {
        await this.trashNoteLocally(item.note_key);
      }
      return result.deleted;
    } catch (err) {
      for (const entry of snapshot.reverse()) {
        this.restoreLocal(entry.item, entry.index);
      }
      if (snapshot.length) this.notify();
      throw err;
    }
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

  isFlushing(): boolean {
    return this.flushing;
  }

  flushProgress(): string {
    return this.flushLabel;
  }

  /**
   * Trash orphan/stale Items/ notes (anything not in the live open set) and
   * delete closed server records. Open items are never touched.
   */
  async flushClosed(): Promise<{ deleted: number; notes: number }> {
    if (this.flushing) throw new Error("Flush already in progress.");
    this.flushing = true;
    this.flushLabel = "Finding open items to keep…";
    this.notify();
    let deleted = 0;
    try {
      const keep = await this.collectOpenKeepSet();
      this.flushLabel = `Keeping ${this.openTotal} open… collecting closed records`;
      this.notify();

      if (!this.offline && this.plugin.jobs?.online) {
        const ids: string[] = [];
        let offset = 0;
        for (;;) {
          const page = await this.plugin.client.listItems({
            statuses: CLOSED_STATUSES.join(","),
            limit: PAGE_SIZE,
            offset,
          });
          for (const item of page.items) {
            const status = (item.status || "").toLowerCase();
            if (!(CLOSED_STATUSES as readonly string[]).includes(status)) continue;
            ids.push(item.id);
          }
          offset += page.items.length;
          this.flushLabel = `Collecting closed records… ${ids.length}`;
          this.notify();
          if (offset >= page.total || page.items.length === 0) break;
        }
        const batch = 40;
        for (let i = 0; i < ids.length; i += batch) {
          const chunk = ids.slice(i, i + batch);
          const result = await this.plugin.client.deleteItems({ ids: chunk });
          deleted += result.deleted;
          this.flushLabel = `Deleting closed… ${Math.min(i + batch, ids.length)}/${ids.length}`;
          this.notify();
        }
      }

      this.flushLabel = "Trashing orphan notes…";
      this.notify();
      const notes = await flushFlushableVaultNotes(
        this.plugin.app,
        this.itemsDir(),
        keep,
        (done, total) => {
          this.flushLabel = `Trashing orphan notes… ${done}/${total}`;
          this.notify();
        },
      );
      this.flushLabel = "Refreshing…";
      this.notify();
      await this.refresh();
      return { deleted, notes };
    } finally {
      this.flushing = false;
      this.flushLabel = "";
      this.notify();
    }
  }

  flushableCount(): number {
    return this.flushable;
  }
}
