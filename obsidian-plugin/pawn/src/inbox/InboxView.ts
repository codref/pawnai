import { Modal, Notice, TAbstractFile, TFile, setIcon } from "obsidian";
import type { InboxItem } from "../api";
import type PawnPlugin from "../main";
import {
  CAPTURES_INBOX_FOLDER,
  CaptureCard,
  captureCreatedAt,
  captureTypeClass,
  captureTypeLabel,
  groupCapturesByDay,
  listInboxCaptures,
} from "./captures";
import {
  IDEAS_FOLDER,
  IdeaCard,
  createTaskFromIdea,
  listInboxIdeas,
  noticeError,
  parkIdea,
  setIdeaStatus,
} from "./ideas";
import { ItemsScope, groupByDay, kindClass, kindLabel } from "./items";
import type { ItemsSection } from "./ItemsStore";

const IDEAS_DEBOUNCE_MS = 400;

/** Lists Ideas/ notes with status inbox for the Ideas chip and status bar. */
export class InboxStore {
  private items: IdeaCard[] = [];
  private listeners = new Set<() => void>();
  private stopped = true;
  private generation = 0;
  private debounceTimer = 0;
  private signature = "";

  constructor(private plugin: PawnPlugin) {}

  start(): void {
    this.stopped = false;
    const onVault = (file: TAbstractFile) => {
      if (!(file instanceof TFile)) return;
      if (!file.path.startsWith(`${IDEAS_FOLDER}/`) && file.path !== IDEAS_FOLDER) return;
      this.scheduleRefresh();
    };
    this.plugin.registerEvent(this.plugin.app.vault.on("create", onVault));
    this.plugin.registerEvent(this.plugin.app.vault.on("modify", onVault));
    this.plugin.registerEvent(this.plugin.app.vault.on("delete", onVault));
    this.plugin.registerEvent(
      this.plugin.app.vault.on("rename", (file, oldPath) => {
        if (
          (file instanceof TFile && file.path.startsWith(`${IDEAS_FOLDER}/`)) ||
          oldPath.startsWith(`${IDEAS_FOLDER}/`)
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

  all(): IdeaCard[] {
    return this.items;
  }

  attention(): number {
    return this.items.length;
  }

  scheduleRefresh(): void {
    if (this.stopped) return;
    if (this.debounceTimer) window.clearTimeout(this.debounceTimer);
    this.debounceTimer = window.setTimeout(() => {
      this.debounceTimer = 0;
      void this.refresh();
    }, IDEAS_DEBOUNCE_MS);
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
    const next = items.map((i) => `${i.path}:${i.title}`).join("|");
    this.items = items;
    if (next === this.signature) {
      this.plugin.updateStatusBar();
      return;
    }
    this.signature = next;
    for (const fn of this.listeners) fn();
    this.plugin.updateStatusBar();
  }
}

/** Research capture inbox atoms (status inbox|proposed). */
export class CapturesStore {
  private items: CaptureCard[] = [];
  private listeners = new Set<() => void>();
  private stopped = true;
  private generation = 0;
  private debounceTimer = 0;
  private signature = "";

  constructor(private plugin: PawnPlugin) {}

  start(): void {
    this.stopped = false;
    const folder = CAPTURES_INBOX_FOLDER;
    const onVault = (file: TAbstractFile) => {
      if (!(file instanceof TFile)) return;
      if (!file.path.startsWith(`${folder}/`) && file.path !== folder) return;
      this.scheduleRefresh();
    };
    this.plugin.registerEvent(this.plugin.app.vault.on("create", onVault));
    this.plugin.registerEvent(this.plugin.app.vault.on("modify", onVault));
    this.plugin.registerEvent(this.plugin.app.vault.on("delete", onVault));
    this.plugin.registerEvent(
      this.plugin.app.vault.on("rename", (file, oldPath) => {
        if (
          (file instanceof TFile && file.path.startsWith(`${folder}/`)) ||
          oldPath.startsWith(`${folder}/`)
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

  all(): CaptureCard[] {
    return this.items;
  }

  attention(): number {
    return this.items.length;
  }

  scheduleRefresh(): void {
    if (this.stopped) return;
    if (this.debounceTimer) window.clearTimeout(this.debounceTimer);
    this.debounceTimer = window.setTimeout(() => {
      this.debounceTimer = 0;
      void this.refresh();
    }, IDEAS_DEBOUNCE_MS);
  }

  async refresh(): Promise<void> {
    if (this.stopped) return;
    const generation = ++this.generation;
    let items: CaptureCard[] = [];
    try {
      items = await listInboxCaptures(this.plugin.app);
    } catch {
      return;
    }
    if (this.stopped || generation !== this.generation) return;
    const next = items.map((i) => `${i.path}:${i.status}:${i.title}`).join("|");
    this.items = items;
    if (next === this.signature) {
      this.plugin.updateStatusBar();
      return;
    }
    this.signature = next;
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
  scope: ItemsScope;
  expanded: Set<string>;
}

const uiState: InboxUiState = {
  section: "items",
  selectMode: false,
  selected: new Set(),
  search: "",
  kind: "",
  scope: "open",
  expanded: new Set(),
};

/** Quiet icon actions for an item (sidebar row or note bar). */
export function mountItemActions(
  parent: HTMLElement,
  plugin: PawnPlugin,
  item: InboxItem,
  onDone?: () => void,
): void {
  const actions = parent.createDiv({ cls: "pawn-item-actions" });
  const iconBtn = (icon: string, label: string, action: string) => {
    const button = actions.createEl("button", {
      cls: "clickable-icon pawn-item-action",
      attr: { "aria-label": label, title: label },
    });
    setIcon(button, icon);
    button.onclick = (ev) => {
      ev.stopPropagation();
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
    iconBtn("check", "Approve", "approve");
    iconBtn("x", "Reject", "reject");
    iconBtn("circle-check", "Done", "ignore");
    iconBtn("trash-2", "Delete", "delete");
    return;
  }

  iconBtn("circle-check", "Done", "ignore");
  iconBtn("list-plus", "Add TODO", "todo");
  if (item.thread) iconBtn("folder-input", "File to thread", "file");
  iconBtn("bot", "Ask Pawn", "task");
  iconBtn("trash-2", "Delete", "delete");
}

export function renderInbox(parent: HTMLElement, plugin: PawnPlugin): void {
  parent.empty();
  const state = uiState;
  state.scope = plugin.items.scope || state.scope;
  state.kind = plugin.items.kind || state.kind;
  state.search = plugin.items.q || state.search;

  const toolbar = parent.createDiv({ cls: "pawn-inbox-toolbar" });
  const sections = toolbar.createDiv({ cls: "pawn-inbox-sections" });
  const openCount = plugin.items.attention();
  const ideaCount = plugin.inbox.attention();
  const captureCount = plugin.captures.attention();
  const vaultCount = plugin.items.vaultNoteCount();
  for (const [id, label, count] of [
    ["items", "Items", openCount],
    ["ideas", "Ideas", ideaCount],
    ["captures", "Captures", captureCount],
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
  if (state.section === "captures") {
    renderCaptures(parent, plugin);
    return;
  }

  const filters = parent.createDiv({ cls: "pawn-inbox-filters" });
  const scopes = filters.createDiv({ cls: "pawn-inbox-scopes" });
  for (const [scope, label] of [
    ["open", "Open"],
    ["closed", "Closed"],
    ["all", "All"],
  ] as const) {
    const chip = scopes.createEl("button", { text: label, cls: "pawn-chip" });
    chip.toggleClass("is-active", state.scope === scope);
    chip.onclick = () => {
      state.scope = scope;
      plugin.items.setQuery({ scope });
      void plugin.items.refresh().then(() => renderInbox(parent, plugin));
    };
  }

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
    const chip = kinds.createEl("button", {
      text: label,
      cls: `pawn-chip${kind ? ` ${kindClass(kind)}` : ""}`,
    });
    chip.toggleClass("is-active", state.kind === kind);
    chip.onclick = () => {
      state.kind = kind;
      plugin.items.setQuery({ kind });
      void plugin.items.refresh().then(() => renderInbox(parent, plugin));
    };
  }

  const flushable = plugin.items.flushableCount();
  const flushing = plugin.items.isFlushing();
  if (flushable > 0 || flushing) {
    const flushRow = parent.createDiv({ cls: "pawn-inbox-flush" });
    flushRow.createSpan({
      cls: "pawn-inbox-hint",
      text: flushing
        ? plugin.items.flushProgress() || "Flushing closed items…"
        : `${vaultCount} notes in Items/ · ${openCount} open · ${flushable} stale can be flushed.`,
    });
    const flushBtn = flushRow.createEl("button", {
      cls: "clickable-icon pawn-item-action",
      attr: {
        "aria-label": flushing ? "Flushing…" : `Flush ${flushable} stale notes`,
        title: flushing ? "Flushing…" : `Flush ${flushable} stale notes`,
      },
    });
    setIcon(flushBtn, flushing ? "loader" : "archive");
    flushBtn.toggleClass("is-busy", flushing);
    flushBtn.disabled = flushing;
    if (!flushing) {
      flushBtn.onclick = () => {
        new ConfirmModal(
          plugin.app,
          "Flush stale item notes?",
          `Trash ${flushable} note(s) under Items/ that are not in the open queue, and remove closed server records. The ${openCount} open item(s) stay.`,
          () => {
            void plugin.items
              .flushClosed()
              .then(({ deleted, notes }) => {
                new Notice(`Removed ${deleted} closed records, trashed ${notes} notes.`, 5000);
                renderInbox(parent, plugin);
              })
              .catch(noticeError);
          },
        ).open();
      };
    }
  }

  const listedIds = plugin.items.all().map((i) => i.id);
  const allSelected =
    listedIds.length > 0 && listedIds.every((id) => state.selected.has(id));
  const bulk = parent.createDiv({ cls: "pawn-inbox-bulk" });
  const selectBtn = bulk.createEl("button", {
    cls: "clickable-icon",
    attr: {
      "aria-label": state.selectMode ? "Cancel select" : "Select",
      title: state.selectMode ? "Cancel select" : "Select",
    },
  });
  setIcon(selectBtn, state.selectMode ? "x" : "check-square");
  selectBtn.onclick = () => {
    state.selectMode = !state.selectMode;
    state.selected.clear();
    renderInbox(parent, plugin);
  };
  const selectAllLabel = allSelected ? "Clear selection" : "Select all";
  const selectAllBtn = bulk.createEl("button", {
    cls: "clickable-icon",
    attr: { "aria-label": selectAllLabel, title: selectAllLabel },
  });
  setIcon(selectAllBtn, "list-checks");
  selectAllBtn.disabled = listedIds.length === 0 || flushing;
  selectAllBtn.onclick = () => {
    if (allSelected) {
      state.selected.clear();
    } else {
      state.selectMode = true;
      state.selected = new Set(listedIds);
    }
    renderInbox(parent, plugin);
  };
  const delSel = bulk.createEl("button", {
    cls: "clickable-icon",
    attr: {
      "aria-label":
        state.selected.size > 0
          ? `Delete selected (${state.selected.size})`
          : "Delete selected",
      title:
        state.selected.size > 0
          ? `Delete selected (${state.selected.size})`
          : "Delete selected",
    },
  });
  setIcon(delSel, "trash-2");
  delSel.disabled = state.selected.size === 0 || flushing;
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
      text:
        state.scope === "open"
          ? "No open items. Meeting extracts land here for triage."
          : "No items match this filter.",
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
  const noteFile = item.note_key
    ? plugin.app.vault.getAbstractFileByPath(item.note_key)
    : null;
  const hasNote = noteFile instanceof TFile;
  const orphaned = !hasNote;

  const top = card.createDiv({ cls: "pawn-item-top" });
  if (state.selectMode) {
    const box = top.createEl("input", { type: "checkbox", cls: "pawn-item-check" });
    box.checked = state.selected.has(item.id);
    box.onchange = () => {
      if (box.checked) state.selected.add(item.id);
      else state.selected.delete(item.id);
      rerender();
    };
  }
  const body = top.createDiv({ cls: "pawn-item-body" });
  body.createDiv({
    cls: `pawn-item-kind ${kindClass(item.kind)}`,
    text: kindLabel(item.kind),
  });
  const title = body.createDiv({
    cls: orphaned ? "pawn-inbox-title is-orphan" : "pawn-inbox-title",
  });
  title.setText((item.text || "").replace(/\s+/g, " ").trim() || "(empty)");
  if (orphaned) {
    title.setAttr("title", item.note_key ? "Item note missing" : "No item note");
  }
  title.onclick = () => {
    if (hasNote) {
      void plugin.app.workspace.getLeaf(false).openFile(noteFile);
      return;
    }
    if (item.note_key) new Notice("Item note missing.", 3000);
    if (state.expanded.has(item.id)) state.expanded.delete(item.id);
    else state.expanded.add(item.id);
    rerender();
  };

  const metaRow = card.createDiv({ cls: "pawn-item-meta-row" });
  const meta = metaRow.createDiv({ cls: "pawn-job-meta" });
  const bits = [
    item.created_at ? window.moment(item.created_at).fromNow() : "",
    item.thread || "",
    item.status,
  ].filter(Boolean);
  meta.setText(bits.join(" · "));
  mountItemActions(metaRow, plugin, item, rerender);

  const expandBtn = card.createEl("button", {
    cls: "pawn-item-expand clickable-icon",
    attr: {
      "aria-label": state.expanded.has(item.id) ? "Hide details" : "Details",
      title: state.expanded.has(item.id) ? "Hide details" : "Details",
    },
  });
  setIcon(expandBtn, state.expanded.has(item.id) ? "chevron-up" : "chevron-down");
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
      if (hasNote) {
        const link = detail.createEl("a", { text: item.note_key, cls: "internal-link" });
        link.onclick = (ev) => {
          ev.preventDefault();
          void plugin.app.workspace.getLeaf(false).openFile(noteFile);
        };
      } else {
        const missing = detail.createDiv({
          cls: "pawn-item-note-missing",
          text: item.note_key,
        });
        missing.setAttr("title", "Item note missing");
      }
    }
  }
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

function mountCaptureActions(
  parent: HTMLElement,
  plugin: PawnPlugin,
  item: CaptureCard,
  onDone?: () => void,
): void {
  const actions = parent.createDiv({ cls: "pawn-item-actions" });
  const iconBtn = (icon: string, label: string, run: () => Promise<void>) => {
    const button = actions.createEl("button", {
      cls: "clickable-icon pawn-item-action",
      attr: { "aria-label": label, title: label },
    });
    setIcon(button, icon);
    button.onclick = (ev) => {
      ev.stopPropagation();
      void run().catch(noticeError);
    };
  };

  iconBtn("folder-input", "File", async () => {
    if (!item.collection || !item.entity) {
      new Notice("Open the note and set collection + entity, then File again.", 5000);
      if (item.file instanceof TFile) {
        void plugin.app.workspace.getLeaf(false).openFile(item.file);
      }
      return;
    }
    const result = await plugin.client.fileCapture({
      path: item.path,
      collection: item.collection,
      entity: item.entity,
    });
    new Notice(
      result.entity_path ? `Filed → ${result.entity_path}` : `Filed ${result.path}`,
      4000,
    );
    await plugin.captures.refresh();
    onDone?.();
  });
  iconBtn("eye-off", "Ignore", async () => {
    await plugin.client.fileCapture({ path: item.path, ignore: true });
    new Notice("Ignored.", 4000);
    await plugin.captures.refresh();
    onDone?.();
  });
  iconBtn("file", "Open", async () => {
    if (item.file instanceof TFile) {
      await plugin.app.workspace.getLeaf(false).openFile(item.file);
    }
  });
}

function renderCaptureRow(
  parent: HTMLElement,
  plugin: PawnPlugin,
  item: CaptureCard,
  state: InboxUiState,
  rerender: () => void,
): void {
  const card = parent.createDiv({
    cls: "pawn-job-card pawn-inbox-card pawn-item-card pawn-capture-card",
  });
  const needsFiling = !item.collection || !item.entity;

  const top = card.createDiv({ cls: "pawn-item-top" });
  const body = top.createDiv({ cls: "pawn-item-body" });
  body.createDiv({
    cls: `pawn-item-kind ${captureTypeClass(item.type, item.snippet_kind)}`,
    text: captureTypeLabel(item.type, item.snippet_kind),
  });
  const title = body.createDiv({
    cls: needsFiling ? "pawn-inbox-title is-orphan" : "pawn-inbox-title",
  });
  title.setText(item.title || "(empty)");
  if (needsFiling) {
    title.setAttr("title", "Set collection + entity before filing");
  }
  title.onclick = () => {
    if (item.file instanceof TFile) {
      void plugin.app.workspace.getLeaf(false).openFile(item.file);
    }
  };

  const metaRow = card.createDiv({ cls: "pawn-item-meta-row" });
  const meta = metaRow.createDiv({ cls: "pawn-job-meta" });
  const created = captureCreatedAt(item);
  const place =
    item.collection && item.entity
      ? `${item.collection} › ${item.entity}`
      : item.collection || item.entity || "";
  const bits = [
    created ? window.moment(created).fromNow() : "",
    place,
    item.status,
  ].filter(Boolean);
  meta.setText(bits.join(" · "));
  mountCaptureActions(metaRow, plugin, item, () => {
    void plugin.captures.refresh().then(rerender);
  });

  const expandBtn = card.createEl("button", {
    cls: "pawn-item-expand clickable-icon",
    attr: {
      "aria-label": state.expanded.has(item.path) ? "Hide details" : "Details",
      title: state.expanded.has(item.path) ? "Hide details" : "Details",
    },
  });
  setIcon(expandBtn, state.expanded.has(item.path) ? "chevron-up" : "chevron-down");
  expandBtn.onclick = () => {
    if (state.expanded.has(item.path)) state.expanded.delete(item.path);
    else state.expanded.add(item.path);
    rerender();
  };

  if (state.expanded.has(item.path)) {
    const detail = card.createDiv({ cls: "pawn-item-detail" });
    if (item.caption) detail.createEl("blockquote", { text: item.caption });
    if (item.hint) detail.createDiv({ text: `Hint: ${item.hint}`, cls: "pawn-item-reason" });
    if (item.source_url) {
      const link = detail.createEl("a", {
        text: item.source_url,
        cls: "external-link",
        attr: { href: item.source_url },
      });
      link.onclick = (ev) => {
        ev.preventDefault();
        window.open(item.source_url, "_blank");
      };
    }
    const pathLink = detail.createEl("a", { text: item.path, cls: "internal-link" });
    pathLink.onclick = (ev) => {
      ev.preventDefault();
      if (item.file instanceof TFile) {
        void plugin.app.workspace.getLeaf(false).openFile(item.file);
      }
    };
  }
}

function renderCaptures(parent: HTMLElement, plugin: PawnPlugin): void {
  const items = plugin.captures.all();
  if (!items.length) {
    parent.createDiv({
      cls: "pawn-empty",
      text: "No research captures waiting. Use the browser extension Research target.",
    });
    return;
  }
  const list = parent.createDiv({ cls: "pawn-inbox-list" });
  const state = uiState;
  const refresh = () => renderInbox(parent, plugin);

  for (const group of groupCapturesByDay(items)) {
    list.createDiv({ cls: "pawn-inbox-day", text: group.label });
    for (const item of group.items) {
      renderCaptureRow(list, plugin, item, state, refresh);
    }
  }
}
