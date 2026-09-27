import { Notice, TFile } from "obsidian";
import { InboxItem, PawnClient } from "../api";
import type PawnPlugin from "../main";

const POLL_MS = 15000;

/** Polls GET /v1/items for the Inbox tab and the status bar. */
export class InboxStore {
  private items: InboxItem[] = [];
  private listeners = new Set<() => void>();
  private timer: number | null = null;
  private stopped = true;

  constructor(
    private plugin: PawnPlugin,
    private client: PawnClient,
  ) {}

  start(): void {
    this.stopped = false;
    void this.refresh();
    this.timer = window.setInterval(() => void this.refresh(), POLL_MS);
  }

  stop(): void {
    this.stopped = true;
    if (this.timer != null) window.clearInterval(this.timer);
  }

  onChange(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  all(): InboxItem[] {
    return this.items;
  }

  attention(): number {
    return this.items.filter((item) => item.interrupt && (item.status === "new" || item.status === "notified")).length;
  }

  async refresh(): Promise<void> {
    if (this.stopped) return;
    try {
      const open = await this.client.listItems();
      this.items = open.filter((item) => item.status === "new" || item.status === "notified" || item.status === "snoozed");
    } catch {
      /* server offline: keep the last list */
    }
    for (const fn of this.listeners) fn();
    this.plugin.updateStatusBar();
  }

  async act(item: InboxItem, action: string): Promise<void> {
    try {
      const receipt = await this.client.itemAction(item.short_id || item.id, action);
      new Notice(receipt, 4000);
    } catch (e) {
      new Notice(e instanceof Error ? e.message : String(e), 6000);
    }
    await this.refresh();
  }
}

export function renderInbox(parent: HTMLElement, plugin: PawnPlugin): void {
  const items = plugin.inbox.all();
  if (!items.length) {
    parent.createDiv({ cls: "pawn-empty", text: "Nothing waiting." });
    return;
  }
  for (const item of items) {
    const card = parent.createDiv({ cls: "pawn-job-card" });
    card.createDiv({ cls: "pawn-job-title", text: item.text });
    const meta = [item.kind, item.thread, item.status].filter(Boolean).join(" · ");
    card.createDiv({ cls: "pawn-job-meta", text: meta });
    const actions = card.createDiv({ cls: "pawn-job-actions" });
    const add = (label: string, action: string) => {
      const button = actions.createEl("button", { text: label });
      button.onclick = () => void plugin.inbox.act(item, action);
    };
    if (item.kind === "schedule_proposal" || item.kind === "proposal") {
      add("Approve", "approve");
      add("Reject", "reject");
    } else {
      add("File", "file");
      add("Task", "task");
      add("Later", "later");
      add("Ignore", "ignore");
    }
    if (item.note_key && plugin.app.vault.getAbstractFileByPath(item.note_key) instanceof TFile) {
      const open = actions.createEl("button", { text: "Open" });
      open.onclick = () => {
        const file = plugin.app.vault.getAbstractFileByPath(item.note_key as string);
        if (file instanceof TFile) void plugin.app.workspace.getLeaf(false).openFile(file);
      };
    }
  }
}
