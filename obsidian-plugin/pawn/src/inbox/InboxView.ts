import { Notice, TFile } from "obsidian";
import {
  IdeaCard,
  createTaskFromIdea,
  listInboxIdeas,
  noticeError,
  parkIdea,
  setIdeaStatus,
} from "./ideas";
import type PawnPlugin from "../main";

/** Lists Ideas/ notes with status inbox for the Inbox tab and the status bar. */
export class InboxStore {
  private items: IdeaCard[] = [];
  private listeners = new Set<() => void>();
  private stopped = true;
  private generation = 0;

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

  all(): IdeaCard[] {
    return this.items;
  }

  attention(): number {
    return this.items.length;
  }

  async refresh(): Promise<void> {
    if (this.stopped) return;
    const generation = ++this.generation;
    let items: IdeaCard[] = [];
    try {
      items = await listInboxIdeas(this.plugin.app);
    } catch {
      return;
    }
    if (this.stopped || generation !== this.generation) return;
    this.items = items;
    for (const fn of this.listeners) fn();
    this.plugin.updateStatusBar();
  }
}

export function renderInbox(parent: HTMLElement, plugin: PawnPlugin): void {
  const items = plugin.inbox.all();
  if (!items.length) {
    parent.createDiv({
      cls: "pawn-empty",
      text: "No ideas waiting. Use /idea or Quick capture.",
    });
    return;
  }
  for (const item of items) {
    const card = parent.createDiv({ cls: "pawn-job-card pawn-inbox-card" });
    const title = card.createDiv({ cls: "pawn-inbox-title", text: item.title });
    title.onclick = () => {
      if (item.file instanceof TFile) void plugin.app.workspace.getLeaf(false).openFile(item.file);
    };
    if (item.line && item.line !== item.title) {
      card.createDiv({ cls: "pawn-inbox-line", text: item.line });
    }
    const actions = card.createDiv({ cls: "pawn-job-actions" });
    const add = (label: string, run: () => Promise<void>) => {
      const button = actions.createEl("button", { text: label });
      button.onclick = () => void run().catch(noticeError);
    };
    add("Keep", async () => {
      await setIdeaStatus(plugin.app, item.file, "later");
      new Notice("Kept for later.", 4000);
    });
    add("Goal", async () => {
      const receipt = await parkIdea(plugin.app, item.path, item.title, item.line || item.title);
      await setIdeaStatus(plugin.app, item.file, "goal");
      new Notice(receipt, 4000);
    });
    add("Task", async () => {
      const path = await createTaskFromIdea(
        plugin.app,
        plugin.settings.agentRoot,
        item.path,
        item.title,
        item.line || item.title,
      );
      await setIdeaStatus(plugin.app, item.file, "task");
      new Notice(`Task ${path}`, 4000);
    });
    add("Drop", async () => {
      await setIdeaStatus(plugin.app, item.file, "dropped");
      new Notice("Dropped.", 4000);
    });
  }
}
