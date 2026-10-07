import { Modal, Notice, TFile } from "obsidian";
import type { InboxItem } from "../api";
import type PawnPlugin from "../main";
import {
  IdeaCard,
  createTaskFromIdea,
  listInboxIdeas,
  noticeError,
  parkIdea,
  setIdeaStatus,
} from "./ideas";
import { groupByDay, kindLabel } from "./items";
import type { ItemsSection } from "./ItemsStore";

/** Lists Ideas/ notes with status inbox for the Ideas chip and status bar. */
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

class ConfirmModal extends Modal {
  constructor(
    app: PawnPlugin["app"],
    private titleText: string,
    private body: string,
    private onConfirm: () => void,
  ) {
    super(app);
  }

  onOpen(): void {
    const { contentEl } = this;
    contentEl.createEl("h3", { text: this.titleText });
    contentEl.createEl("p", { text: this.body });
    const row = contentEl.createDiv({ cls: "pawn-job-actions" });
    const cancel = row.createEl("button", { text: "Cancel" });
    cancel.onclick = () => this.close();
    const ok = row.createEl("button", { text: "Delete", cls: "mod-warning" });
    ok.onclick = () => {
      this.onConfirm();
      this.close();
    };
  }
}

const KINDS = [
  "",
  "decision",
  "commitment",
  "open_question",
  "block",
  "contradiction",
  "proposal",
  "people_update",
  "schedule_proposal",
];

interface InboxUiState {
  section: ItemsSection;
  selectMode: boolean;
  selected: Set<string>;
  search: string;
  kind: string;
  expanded: Set<string>;
}

const uiState: InboxUiState = {
  section: "items",
  selectMode: false,
  selected: new Set(),
  search: "",
  kind: "",
  expanded: new Set(),
};

/** Shared action buttons for an item (sidebar row or note bar). */
export function mountItemActions(
  parent: HTMLElement,
  plugin: PawnPlugin,
  item: InboxItem,
  onDone?: () => void,
): void {
  const actions = parent.createDiv({ cls: "pawn-job-actions pawn-item-actions" });
  const run = (label: string, action: string, cls?: string) => {
    const button = actions.createEl("button", { text: label });
    if (cls) button.addClass(cls);
    button.onclick = () => {
      void plugin.items
        .runAction(item.id, action)
        .then((receipt) => {
          new Notice(receipt, 4000);
          onDone?.();
        })
        .catch(noticeError);
    };
  };

  const kind = item.kind || "";
  if (kind === "people_update" || kind === "schedule_proposal" || kind === "proposal") {
    run("Approve", "approve", "mod-cta");
    run("Reject", "reject", "mod-warning");
    run("Delete", "delete", "mod-warning");
    return;
  }

  run("Add TODO", "todo", "mod-cta");
  if (item.thread) run("File to thread", "file");
  run("Ask Pawn", "task");
  run("Delete", "delete", "mod-warning");
}

export function renderInbox(parent: HTMLElement, plugin: PawnPlugin): void {
  parent.empty();
  const state = uiState;

  const toolbar = parent.createDiv({ cls: "pawn-inbox-toolbar" });
  const sections = toolbar.createDiv({ cls: "pawn-inbox-sections" });
  const itemCount = plugin.items.attention();
  const ideaCount = plugin.inbox.attention();
  for (const [id, label, count] of [
    ["items", "Items", itemCount],
    ["ideas", "Ideas", ideaCount],
  ] as const) {
    const b = sections.createEl("button", {
      text: count ? `${label} (${count})` : label,
    });
    b.toggleClass("is-active", state.section === id);
    b.onclick = () => {
      state.section = id;
      renderInbox(parent, plugin);
    };
  }

  if (state.section === "ideas") {
    renderIdeas(parent, plugin);
    return;
  }

  const filters = parent.createDiv({ cls: "pawn-inbox-filters" });
  const search = filters.createEl("input", {
    type: "search",
    placeholder: "Search items…",
    cls: "pawn-inbox-search",
  });
  search.value = state.search;
  search.oninput = () => {
    state.search = search.value;
  };
  const applySearch = () => {
    plugin.items.setQuery({ q: state.search });
    void plugin.items.refresh().then(() => renderInbox(parent, plugin));
  };
  search.onchange = applySearch;
  search.onkeydown = (ev) => {
    if (ev.key === "Enter") applySearch();
  };

  const kinds = filters.createDiv({ cls: "pawn-inbox-kinds" });
  for (const kind of KINDS) {
    const label = kind ? kindLabel(kind) : "All kinds";
    const chip = kinds.createEl("button", { text: label, cls: "pawn-chip" });
    chip.toggleClass("is-active", state.kind === kind);
    chip.onclick = () => {
      state.kind = kind;
      plugin.items.setQuery({ kind });
      void plugin.items.refresh().then(() => renderInbox(parent, plugin));
    };
  }

  const bulk = parent.createDiv({ cls: "pawn-inbox-bulk" });
  const selectBtn = bulk.createEl("button", {
    text: state.selectMode ? "Cancel select" : "Select",
  });
  selectBtn.onclick = () => {
    state.selectMode = !state.selectMode;
    state.selected.clear();
    renderInbox(parent, plugin);
  };
  if (state.selectMode) {
    const delSel = bulk.createEl("button", {
      text: `Delete selected (${state.selected.size})`,
      cls: "mod-warning",
    });
    delSel.disabled = state.selected.size === 0;
    delSel.onclick = () => {
      const ids = Array.from(state.selected);
      new ConfirmModal(
        plugin.app,
        "Delete selected items?",
        `Delete ${ids.length} item(s). They will not come back.`,
        () => {
          void plugin.items
            .deleteIds(ids)
            .then((n) => {
              state.selected.clear();
              state.selectMode = false;
              new Notice(`Deleted ${n}.`, 4000);
              renderInbox(parent, plugin);
            })
            .catch(noticeError);
        },
      ).open();
    };
  }
  const delAll = bulk.createEl("button", { text: "Delete all open", cls: "mod-warning" });
  delAll.disabled = itemCount === 0;
  delAll.onclick = () => {
    new ConfirmModal(
      plugin.app,
      "Delete all open items?",
      `This removes ${itemCount} open item(s) matching the current filters.`,
      () => {
        void plugin.items
          .deleteAllOpen()
          .then((n) => {
            state.selected.clear();
            state.selectMode = false;
            new Notice(`Deleted ${n}.`, 4000);
            renderInbox(parent, plugin);
          })
          .catch(noticeError);
      },
    ).open();
  };

  if (plugin.items.isOffline()) {
    parent.createDiv({
      cls: "pawn-inbox-offline",
      text: "Server offline — showing item notes from the vault.",
    });
  }

  const items = plugin.items.all();
  if (!items.length) {
    parent.createDiv({
      cls: "pawn-empty",
      text: "No open items. Meeting extracts land here for triage.",
    });
    return;
  }

  const list = parent.createDiv({ cls: "pawn-inbox-list" });
  for (const group of groupByDay(items)) {
    list.createDiv({ cls: "pawn-inbox-day", text: group.label });
    for (const item of group.items) {
      renderItemRow(list, plugin, item, state, () => renderInbox(parent, plugin));
    }
  }

  if (plugin.items.hasMore()) {
    const more = parent.createEl("button", { text: "Load more", cls: "pawn-inbox-more" });
    more.onclick = () => {
      void plugin.items.loadMore().then(() => renderInbox(parent, plugin));
    };
  }
}

function renderItemRow(
  parent: HTMLElement,
  plugin: PawnPlugin,
  item: InboxItem,
  state: InboxUiState,
  rerender: () => void,
): void {
  const card = parent.createDiv({ cls: "pawn-job-card pawn-inbox-card pawn-item-card" });
  const head = card.createDiv({ cls: "pawn-item-head" });
  if (state.selectMode) {
    const box = head.createEl("input", { type: "checkbox", cls: "pawn-item-check" });
    box.checked = state.selected.has(item.id);
    box.onchange = () => {
      if (box.checked) state.selected.add(item.id);
      else state.selected.delete(item.id);
      rerender();
    };
  }
  head.createSpan({ cls: "pawn-item-kind", text: kindLabel(item.kind) });
  const title = head.createDiv({ cls: "pawn-inbox-title" });
  title.setText((item.text || "").replace(/\s+/g, " ").trim() || "(empty)");
  title.onclick = () => {
    const path = item.note_key;
    const file = path ? plugin.app.vault.getAbstractFileByPath(path) : null;
    if (file instanceof TFile) {
      void plugin.app.workspace.getLeaf(false).openFile(file);
      return;
    }
    if (state.expanded.has(item.id)) state.expanded.delete(item.id);
    else state.expanded.add(item.id);
    rerender();
  };

  const meta = card.createDiv({ cls: "pawn-job-meta" });
  const bits = [
    item.created_at ? window.moment(item.created_at).fromNow() : "",
    item.thread || "",
    item.status,
  ].filter(Boolean);
  meta.setText(bits.join(" · "));

  const expandBtn = card.createEl("button", {
    cls: "pawn-item-expand",
    text: state.expanded.has(item.id) ? "Hide details" : "Details",
  });
  expandBtn.onclick = () => {
    if (state.expanded.has(item.id)) state.expanded.delete(item.id);
    else state.expanded.add(item.id);
    rerender();
  };

  if (state.expanded.has(item.id)) {
    const detail = card.createDiv({ cls: "pawn-item-detail" });
    if (item.quote) detail.createEl("blockquote", { text: item.quote });
    if (item.reason) detail.createDiv({ text: item.reason, cls: "pawn-item-reason" });
    if (item.note_key) {
      const link = detail.createEl("a", { text: item.note_key, cls: "internal-link" });
      link.onclick = (ev) => {
        ev.preventDefault();
        const file = plugin.app.vault.getAbstractFileByPath(item.note_key!);
        if (file instanceof TFile) void plugin.app.workspace.getLeaf(false).openFile(file);
      };
    }
  }

  mountItemActions(card, plugin, item, rerender);
}

function renderIdeas(parent: HTMLElement, plugin: PawnPlugin): void {
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
