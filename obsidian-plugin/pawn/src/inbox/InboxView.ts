import { Modal, Notice, TAbstractFile, TFile, setIcon } from "obsidian";
import type { InboxItem } from "../api";
import type PawnPlugin from "../main";
import {
  CAPTURES_INBOX_FOLDER,
  CaptureCard,
  captureCreatedAt,
  captureKindKey,
  captureKindOptions,
  captureMatchesQuery,
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
import {
  ItemsGroupBy,
  ItemsScope,
  groupByDay,
  groupBySession,
  isItemUnread,
  kindClass,
  kindLabel,
} from "./items";
import type { ItemsSection } from "./ItemsStore";
import { pickTodoProject } from "./projects";

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

export type CapturesStatusScope = "inbox" | "proposed" | "all";

/** Research capture inbox atoms (status inbox|proposed). */
export class CapturesStore {
  private items: CaptureCard[] = [];
  private listeners = new Set<() => void>();
  private stopped = true;
  private generation = 0;
  private debounceTimer = 0;
  private signature = "";
  q = "";
  kind = "";
  scope: CapturesStatusScope = "all";

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

  /** Inbox atoms after status / kind / search filters. */
  filtered(): CaptureCard[] {
    return this.items.filter((card) => {
      if (this.scope !== "all" && card.status !== this.scope) return false;
      if (this.kind && captureKindKey(card) !== this.kind) return false;
      return captureMatchesQuery(card, this.q);
    });
  }

  setQuery(partial: {
    q?: string;
    kind?: string;
    scope?: CapturesStatusScope;
  }): void {
    if (partial.q !== undefined) this.q = partial.q;
    if (partial.kind !== undefined) this.kind = partial.kind;
    if (partial.scope !== undefined) this.scope = partial.scope;
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

  async deletePaths(paths: string[]): Promise<number> {
    let n = 0;
    for (const path of paths) {
      const file = this.plugin.app.vault.getAbstractFileByPath(path);
      if (!(file instanceof TFile)) continue;
      try {
        await this.plugin.app.vault.trash(file, true);
        n += 1;
      } catch {
        // skip failures; refresh will reconcile
      }
    }
    this.items = this.items.filter((i) => !paths.includes(i.path));
    this.signature = "";
    await this.refresh();
    return n;
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
  reviewMode: boolean;
  selected: Set<string>;
  search: string;
  kind: string;
  scope: ItemsScope;
  expanded: Set<string>;
  focusedId: string | null;
}

const uiState: InboxUiState = {
  section: "items",
  selectMode: false,
  reviewMode: false,
  selected: new Set(),
  search: "",
  kind: "",
  scope: "open",
  expanded: new Set(),
  focusedId: null,
};

const CONTEXT_PREVIEW = 220;

/** Preserved across ChatView body rebuilds (store notify). */
let pendingJobsScroll: number | null = null;

/** Call before recreating the inbox root so scroll position survives notify. */
export function preserveInboxScroll(top: number): void {
  pendingJobsScroll = top;
}

function contextPreview(item: InboxItem): string {
  const raw = (item.context || "").trim() || (item.quote || "").trim();
  if (!raw) return "";
  const oneLine = raw.replace(/\s+/g, " ");
  if (oneLine.length <= CONTEXT_PREVIEW) return oneLine;
  return `${oneLine.slice(0, CONTEXT_PREVIEW).replace(/\s+\S*$/, "")}…`;
}

function anchorAfterRemove(ids: string[], removedId: string): string | null {
  const idx = ids.indexOf(removedId);
  if (idx < 0) return ids[0] ?? null;
  return ids[idx + 1] ?? ids[idx - 1] ?? null;
}

function restoreScroll(el: HTMLElement, top: number): void {
  el.scrollTop = top;
  requestAnimationFrame(() => {
    if (el.isConnected) el.scrollTop = top;
  });
}

function focusCard(listRoot: HTMLElement, id: string | null): void {
  if (!id) return;
  const card = listRoot.querySelector(
    `.pawn-item-card[data-item-id="${CSS.escape(id)}"]`,
  ) as HTMLElement | null;
  if (!card) return;
  card.addClass("is-focused");
  card.scrollIntoView({ block: "nearest" });
}

export interface ItemActionsOpts {
  /** Include Ask Pawn (task). Default true for note bar; Inbox rows pass false. */
  ask?: boolean;
  /** Called before the async action (for busy UI / focus anchor). */
  onStart?: (action: string) => void;
}

/** Pick a project then run todo. Returns null if the user cancelled. */
export async function runTodoWithProject(
  plugin: PawnPlugin,
  itemId: string,
  opts?: { onConfirm?: () => void },
): Promise<string | null> {
  const choice = await pickTodoProject(plugin.app, plugin.settings.agentRoot);
  if (choice.type === "cancel") return null;
  opts?.onConfirm?.();
  const arg = choice.type === "project" ? choice.name : undefined;
  return plugin.items.runAction(itemId, "todo", arg);
}

/** Quiet icon actions for an item (sidebar row or note bar). */
export function mountItemActions(
  parent: HTMLElement,
  plugin: PawnPlugin,
  item: InboxItem,
  onDone?: () => void,
  opts?: ItemActionsOpts,
): void {
  const ask = opts?.ask !== false;
  const actions = parent.createDiv({ cls: "pawn-item-actions" });
  const run = (action: string, arg?: string) => {
    opts?.onStart?.(action);
    return plugin.items.runAction(item.id, action, arg);
  };
  const iconBtn = (icon: string, label: string, action: string) => {
    const button = actions.createEl("button", {
      cls: "clickable-icon pawn-item-action",
      attr: { "aria-label": label, title: label },
    });
    setIcon(button, icon);
    button.onclick = (ev) => {
      ev.stopPropagation();
      button.addClass("is-busy");
      button.setAttr("disabled", "true");
      void run(action)
        .then((receipt) => {
          new Notice(receipt, 4000);
          onDone?.();
        })
        .catch((err) => {
          button.removeClass("is-busy");
          button.removeAttribute("disabled");
          noticeError(err);
        });
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

  const todoBtn = actions.createEl("button", {
    cls: "clickable-icon pawn-item-action",
    attr: { "aria-label": "Add TODO", title: "Add TODO" },
  });
  setIcon(todoBtn, "list-plus");
  todoBtn.onclick = (ev) => {
    ev.stopPropagation();
    todoBtn.addClass("is-busy");
    todoBtn.setAttr("disabled", "true");
    void runTodoWithProject(plugin, item.id, {
      onConfirm: () => opts?.onStart?.("todo"),
    })
      .then((receipt) => {
        if (receipt == null) {
          todoBtn.removeClass("is-busy");
          todoBtn.removeAttribute("disabled");
          return;
        }
        new Notice(receipt, 4000);
        onDone?.();
      })
      .catch((err) => {
        todoBtn.removeClass("is-busy");
        todoBtn.removeAttribute("disabled");
        noticeError(err);
      });
  };

  if (item.thread) iconBtn("folder-input", "File to thread", "file");
  if (ask) iconBtn("bot", "Ask Pawn", "task");
  iconBtn("trash-2", "Delete", "delete");
}

async function openItemSessionTranscript(
  plugin: PawnPlugin,
  sessionId: string,
): Promise<void> {
  const sessions = plugin.sessions?.all() ?? [];
  const match = sessions.find((s) => s.session_id === sessionId);
  if (match) {
    const file = plugin.sessions?.resolveTranscript(match);
    if (file) {
      await plugin.app.workspace.getLeaf(false).openFile(file);
      return;
    }
  }
  // Offline / not in list: scan vault for pawn: transcript with this session_id.
  for (const file of plugin.app.vault.getMarkdownFiles()) {
    const fm = plugin.app.metadataCache.getFileCache(file)?.frontmatter;
    if (!fm || fm.pawn !== "transcript") continue;
    if (String(fm.session_id || file.basename).trim() === sessionId) {
      await plugin.app.workspace.getLeaf(false).openFile(file);
      return;
    }
  }
  new Notice("Transcript not in vault yet.", 4000);
}

/**
 * Switch to Inbox Items, group by session, and filter to one session origin.
 */
export function focusInboxOnSession(plugin: PawnPlugin, sessionId: string): void {
  uiState.section = "items";
  plugin.items.setQuery({
    groupBy: "session",
    sourceRef: sessionId,
    scope: "open",
  });
  void plugin.openChat({ tab: "inbox" }).then(() => {
    void plugin.items.refresh();
  });
}

export function renderInbox(parent: HTMLElement, plugin: PawnPlugin): void {
  const scrollTop = pendingJobsScroll ?? parent.scrollTop;
  pendingJobsScroll = null;
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
      if (state.section !== id) {
        state.selectMode = false;
        state.reviewMode = false;
        state.selected.clear();
        state.focusedId = null;
      }
      state.section = id;
      renderInbox(parent, plugin);
    };
  }

  if (state.section === "ideas") {
    renderIdeas(parent, plugin);
    restoreScroll(parent, scrollTop);
    return;
  }
  if (state.section === "captures") {
    renderCaptures(parent, plugin);
    restoreScroll(parent, scrollTop);
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

  const groupBy = plugin.items.groupBy;
  const groupWrap = scopes.createDiv({ cls: "pawn-inbox-groupby" });
  groupWrap.createSpan({ cls: "pawn-inbox-groupby-label", text: "Group" });
  const groupSelect = groupWrap.createEl("select", {
    cls: "dropdown pawn-inbox-groupby-select",
    attr: { "aria-label": "Group items by" },
  });
  for (const [id, label] of [
    ["day", "Date"],
    ["session", "Session"],
  ] as const) {
    const opt = groupSelect.createEl("option", { text: label, value: id });
    if (groupBy === id) opt.selected = true;
  }
  groupSelect.onchange = () => {
    plugin.items.setQuery({ groupBy: groupSelect.value as ItemsGroupBy });
    renderInbox(parent, plugin);
  };

  if (plugin.items.sourceRef.trim()) {
    const filterRow = parent.createDiv({ cls: "pawn-inbox-flush" });
    filterRow.createSpan({
      cls: "pawn-inbox-hint",
      text: `Filtered: ${plugin.items.sourceRef}`,
    });
    const clear = filterRow.createEl("button", {
      text: "Clear",
      cls: "pawn-chip",
    });
    clear.onclick = () => {
      plugin.items.setQuery({ sourceRef: "" });
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
  if (state.focusedId && !listedIds.includes(state.focusedId)) {
    state.focusedId = listedIds[0] ?? null;
  }
  if (state.reviewMode && !state.focusedId && listedIds.length) {
    state.focusedId = listedIds[0];
  }
  const allSelected =
    listedIds.length > 0 && listedIds.every((id) => state.selected.has(id));
  const bulk = parent.createDiv({ cls: "pawn-inbox-bulk" });
  const reviewBtn = bulk.createEl("button", {
    cls: "clickable-icon",
    attr: {
      "aria-label": state.reviewMode ? "Exit review" : "Review",
      title: state.reviewMode
        ? "Exit review (j/k · d done · x delete · t todo)"
        : "Review with keyboard",
    },
  });
  setIcon(reviewBtn, state.reviewMode ? "keyboard" : "eye");
  reviewBtn.toggleClass("is-active", state.reviewMode);
  reviewBtn.onclick = () => {
    state.reviewMode = !state.reviewMode;
    if (state.reviewMode) {
      state.selectMode = false;
      state.selected.clear();
      state.focusedId = listedIds[0] ?? null;
    } else {
      state.focusedId = null;
    }
    renderInbox(parent, plugin);
  };
  const selectBtn = bulk.createEl("button", {
    cls: "clickable-icon",
    attr: {
      "aria-label": state.selectMode ? "Cancel select" : "Select",
      title: state.selectMode ? "Cancel select" : "Select",
    },
  });
  setIcon(selectBtn, state.selectMode ? "x" : "check-square");
  selectBtn.toggleClass("is-active", state.selectMode);
  selectBtn.onclick = () => {
    state.selectMode = !state.selectMode;
    state.selected.clear();
    if (state.selectMode) {
      state.reviewMode = false;
      state.focusedId = null;
    }
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
      state.reviewMode = false;
      state.focusedId = null;
      state.selected = new Set(listedIds);
    }
    renderInbox(parent, plugin);
  };
  const delSel = bulk.createEl("button", {
    cls: "clickable-icon pawn-inbox-del-sel",
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
        state.selected.clear();
        state.selectMode = false;
        void plugin.items
          .deleteIds(ids)
          .then((n) => new Notice(`Deleted ${n}.`, 4000))
          .catch(noticeError);
      },
    ).open();
  };
  if (state.reviewMode) {
    parent.createDiv({
      cls: "pawn-inbox-hint",
      text: "Review · j/k move · d done · x delete · t todo · f file · e expand · Enter open",
    });
  }

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
    restoreScroll(parent, scrollTop);
    return;
  }

  const list = parent.createDiv({ cls: "pawn-inbox-list" });
  const rerender = () => renderInbox(parent, plugin);
  const runTriaged = (itemId: string, action: string, arg?: string) => {
    state.focusedId = anchorAfterRemove(listedIds, itemId);
    state.selected.delete(itemId);
    // Store notify rebuilds the inbox; avoid rendering into a detached parent.
    void plugin.items
      .runAction(itemId, action, arg)
      .then((receipt) => {
        if (!state.reviewMode) new Notice(receipt, 3000);
      })
      .catch(noticeError);
  };

  const runTodoTriaged = (itemId: string) => {
    void runTodoWithProject(plugin, itemId, {
      onConfirm: () => {
        state.focusedId = anchorAfterRemove(listedIds, itemId);
        state.selected.delete(itemId);
      },
    })
      .then((receipt) => {
        if (receipt == null) return;
        if (!state.reviewMode) new Notice(receipt, 3000);
      })
      .catch(noticeError);
  };

  const rowHandlers = {
    rerender,
    onSelectToggle: () => updateBulkDeleteLabel(parent, state),
    onActionStart: (id: string) => {
      state.focusedId = anchorAfterRemove(listedIds, id);
    },
  };
  if (plugin.items.groupBy === "session") {
    const titles = new Map<string, string>();
    for (const s of plugin.sessions?.all() ?? []) {
      titles.set(s.session_id, s.title || s.session_id);
    }
    for (const day of groupBySession(items, titles)) {
      list.createDiv({ cls: "pawn-inbox-day", text: day.label });
      for (const session of day.sessions) {
        const sessionHead = list.createDiv({ cls: "pawn-inbox-session" });
        sessionHead.createSpan({
          cls: "pawn-inbox-session-title",
          text: session.label,
        });
        sessionHead.createSpan({
          cls: "pawn-inbox-session-count",
          text: String(session.items.length),
        });
        for (const item of session.items) {
          renderItemRow(list, plugin, item, state, rowHandlers);
        }
      }
    }
  } else {
    for (const group of groupByDay(items)) {
      list.createDiv({ cls: "pawn-inbox-day", text: group.label });
      for (const item of group.items) {
        renderItemRow(list, plugin, item, state, rowHandlers);
      }
    }
  }

  if (plugin.items.hasMore()) {
    const more = parent.createEl("button", { text: "Load more", cls: "pawn-inbox-more" });
    more.onclick = () => {
      void plugin.items.loadMore().then(() => renderInbox(parent, plugin));
    };
  }

  restoreScroll(parent, scrollTop);
  if (state.reviewMode) {
    focusCard(parent, state.focusedId);
    parent.tabIndex = 0;
    parent.onkeydown = (ev) => {
      if (!state.reviewMode) return;
      const target = ev.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }
      const ids = plugin.items.all().map((i) => i.id);
      if (!ids.length) return;
      let focus = state.focusedId && ids.includes(state.focusedId) ? state.focusedId : ids[0];
      const idx = ids.indexOf(focus);
      const key = ev.key;
      if (key === "j" || key === "ArrowDown") {
        ev.preventDefault();
        state.focusedId = ids[Math.min(ids.length - 1, idx + 1)];
        rerender();
        return;
      }
      if (key === "k" || key === "ArrowUp") {
        ev.preventDefault();
        state.focusedId = ids[Math.max(0, idx - 1)];
        rerender();
        return;
      }
      if (key === "e") {
        ev.preventDefault();
        if (state.expanded.has(focus)) state.expanded.delete(focus);
        else state.expanded.add(focus);
        rerender();
        return;
      }
      if (key === "Enter") {
        ev.preventDefault();
        const item = plugin.items.all().find((i) => i.id === focus);
        const noteFile = item?.note_key
          ? plugin.app.vault.getAbstractFileByPath(item.note_key)
          : null;
        if (noteFile instanceof TFile && item) {
          plugin.items.markOpened(item);
          void plugin.app.workspace.getLeaf(false).openFile(noteFile);
        }
        return;
      }
      if (key === "d") {
        ev.preventDefault();
        runTriaged(focus, "ignore");
        return;
      }
      if (key === "x" || key === "Backspace") {
        ev.preventDefault();
        runTriaged(focus, "delete");
        return;
      }
      if (key === "t") {
        ev.preventDefault();
        runTodoTriaged(focus);
        return;
      }
      if (key === "f") {
        const item = plugin.items.all().find((i) => i.id === focus);
        if (!item?.thread) return;
        ev.preventDefault();
        runTriaged(focus, "file");
      }
    };
    parent.focus({ preventScroll: true });
  }
}

function updateBulkDeleteLabel(parent: HTMLElement, state: InboxUiState): void {
  const delSel = parent.querySelector(".pawn-inbox-del-sel") as HTMLButtonElement | null;
  if (!delSel) return;
  const label =
    state.selected.size > 0
      ? `Delete selected (${state.selected.size})`
      : "Delete selected";
  delSel.setAttr("aria-label", label);
  delSel.setAttr("title", label);
  delSel.disabled = state.selected.size === 0;
}

interface ItemRowHandlers {
  rerender: () => void;
  onSelectToggle: () => void;
  onActionStart: (id: string) => void;
}

function renderItemRow(
  parent: HTMLElement,
  plugin: PawnPlugin,
  item: InboxItem,
  state: InboxUiState,
  handlers: ItemRowHandlers,
): void {
  const { rerender, onSelectToggle, onActionStart } = handlers;
  const card = parent.createDiv({ cls: "pawn-job-card pawn-inbox-card pawn-item-card" });
  card.dataset.itemId = item.id;
  if (state.reviewMode && state.focusedId === item.id) card.addClass("is-focused");
  const unread = isItemUnread(item);
  if (unread) card.addClass("is-unread");
  const noteFile = item.note_key
    ? plugin.app.vault.getAbstractFileByPath(item.note_key)
    : null;
  const hasNote = noteFile instanceof TFile;
  const orphaned = !hasNote;

  if (unread) {
    card.createDiv({
      cls: "pawn-item-unread-dot",
      attr: { "aria-label": "Unread", title: "Unread" },
    });
  }

  const top = card.createDiv({ cls: "pawn-item-top" });
  if (state.selectMode) {
    const box = top.createEl("input", { type: "checkbox", cls: "pawn-item-check" });
    box.checked = state.selected.has(item.id);
    box.onclick = (ev) => ev.stopPropagation();
    box.onchange = () => {
      if (box.checked) state.selected.add(item.id);
      else state.selected.delete(item.id);
      onSelectToggle();
    };
  }
  const body = top.createDiv({ cls: "pawn-item-body" });
  body.createDiv({
    cls: `pawn-item-kind ${kindClass(item.kind)}`,
    text: kindLabel(item.kind),
  });
  const title = body.createDiv({
    cls: orphaned
      ? "pawn-inbox-title is-orphan"
      : unread
        ? "pawn-inbox-title is-unread-title"
        : "pawn-inbox-title",
  });
  title.setText((item.text || "").replace(/\s+/g, " ").trim() || "(empty)");
  if (orphaned) {
    title.setAttr("title", item.note_key ? "Item note missing" : "No item note");
  }
  const preview = contextPreview(item);
  if (preview) {
    body.createDiv({ cls: "pawn-inbox-context", text: preview });
  }
  title.onclick = () => {
    if (state.reviewMode) {
      state.focusedId = item.id;
      rerender();
      return;
    }
    if (hasNote) {
      plugin.items.markOpened(item);
      void plugin.app.workspace.getLeaf(false).openFile(noteFile);
      return;
    }
    if (item.note_key) new Notice("Item note missing.", 3000);
    if (state.expanded.has(item.id)) state.expanded.delete(item.id);
    else state.expanded.add(item.id);
    rerender();
  };
  if (state.reviewMode) {
    card.onclick = () => {
      if (state.focusedId !== item.id) {
        state.focusedId = item.id;
        rerender();
      }
    };
  }

  const metaRow = card.createDiv({ cls: "pawn-item-meta-row" });
  const meta = metaRow.createDiv({ cls: "pawn-job-meta" });
  const bits = [
    item.created_at ? window.moment(item.created_at).fromNow() : "",
    item.thread || "",
    item.status,
  ].filter(Boolean);
  meta.setText(bits.join(" · "));
  mountItemActions(
    metaRow,
    plugin,
    item,
    undefined, // ItemsStore.notify rebuilds the list
    {
      ask: false,
      onStart: () => onActionStart(item.id),
    },
  );
  if (item.source_kind === "session" && item.source_ref) {
    const actions = metaRow.querySelector(".pawn-item-actions");
    if (actions instanceof HTMLElement) {
      const openTx = actions.createEl("button", {
        cls: "clickable-icon pawn-item-action",
        attr: {
          "aria-label": "Open transcript",
          title: `Session ${item.source_ref}`,
        },
      });
      setIcon(openTx, "file-text");
      actions.insertBefore(openTx, actions.firstChild);
      openTx.onclick = (ev) => {
        ev.stopPropagation();
        void openItemSessionTranscript(plugin, item.source_ref!);
      };
    }
  }

  const expandBtn = card.createEl("button", {
    cls: "pawn-item-expand clickable-icon",
    attr: {
      "aria-label": state.expanded.has(item.id) ? "Hide details" : "Details",
      title: state.expanded.has(item.id) ? "Hide details" : "Details",
    },
  });
  setIcon(expandBtn, state.expanded.has(item.id) ? "chevron-up" : "chevron-down");
  expandBtn.onclick = (ev) => {
    ev.stopPropagation();
    if (state.expanded.has(item.id)) state.expanded.delete(item.id);
    else state.expanded.add(item.id);
    rerender();
  };

  if (state.expanded.has(item.id)) {
    const detail = card.createDiv({ cls: "pawn-item-detail" });
    if (item.context) {
      detail.createDiv({ cls: "pawn-item-context-full", text: item.context });
    }
    if (item.quote) detail.createEl("blockquote", { text: item.quote });
    if (item.reason) detail.createDiv({ text: item.reason, cls: "pawn-item-reason" });
    const askRow = detail.createDiv({ cls: "pawn-item-detail-actions" });
    const askBtn = askRow.createEl("button", {
      cls: "clickable-icon pawn-item-action",
      attr: { "aria-label": "Ask Pawn", title: "Ask Pawn" },
    });
    setIcon(askBtn, "bot");
    askBtn.onclick = (ev) => {
      ev.stopPropagation();
      onActionStart(item.id);
      void plugin.items
        .runAction(item.id, "task")
        .then((receipt) => new Notice(receipt, 4000))
        .catch(noticeError);
    };
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
  iconBtn("trash-2", "Delete", async () => {
    const n = await plugin.captures.deletePaths([item.path]);
    new Notice(n ? "Deleted." : "Already gone.", 4000);
    onDone?.();
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
  if (state.selectMode) {
    const check = top.createEl("input", {
      type: "checkbox",
      cls: "pawn-item-check",
    });
    check.checked = state.selected.has(item.path);
    check.onclick = (ev) => ev.stopPropagation();
    check.onchange = () => {
      if (check.checked) state.selected.add(item.path);
      else state.selected.delete(item.path);
      rerender();
    };
  }
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
    if (state.selectMode) {
      if (state.selected.has(item.path)) state.selected.delete(item.path);
      else state.selected.add(item.path);
      rerender();
      return;
    }
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
  const state = uiState;
  const refresh = () => renderInbox(parent, plugin);

  const filters = parent.createDiv({ cls: "pawn-inbox-filters" });
  const scopes = filters.createDiv({ cls: "pawn-inbox-scopes" });
  for (const [scope, label] of [
    ["inbox", "Inbox"],
    ["proposed", "Proposed"],
    ["all", "All"],
  ] as const) {
    const chip = scopes.createEl("button", { text: label, cls: "pawn-chip" });
    chip.toggleClass("is-active", plugin.captures.scope === scope);
    chip.onclick = () => {
      plugin.captures.setQuery({ scope });
      refresh();
    };
  }

  const search = filters.createEl("input", {
    type: "search",
    placeholder: "Search captures…",
    cls: "pawn-inbox-search",
  });
  search.value = plugin.captures.q;
  search.oninput = () => {
    plugin.captures.setQuery({ q: search.value });
  };
  const applySearch = () => {
    plugin.captures.setQuery({ q: search.value });
    refresh();
  };
  search.onchange = applySearch;
  search.onkeydown = (ev) => {
    if (ev.key === "Enter") applySearch();
  };

  const kinds = filters.createDiv({ cls: "pawn-inbox-kinds" });
  for (const kind of captureKindOptions(plugin.captures.all())) {
    const label = kind ? captureTypeLabel(kind, kind) : "All kinds";
    const chip = kinds.createEl("button", {
      text: label,
      cls: `pawn-chip${kind ? ` ${captureTypeClass(kind, kind)}` : ""}`,
    });
    chip.toggleClass("is-active", plugin.captures.kind === kind);
    chip.onclick = () => {
      plugin.captures.setQuery({ kind });
      refresh();
    };
  }

  const listed = plugin.captures.filtered();
  const listedPaths = listed.map((i) => i.path);
  const allSelected =
    listedPaths.length > 0 && listedPaths.every((p) => state.selected.has(p));
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
    refresh();
  };
  const selectAllLabel = allSelected ? "Clear selection" : "Select all";
  const selectAllBtn = bulk.createEl("button", {
    cls: "clickable-icon",
    attr: { "aria-label": selectAllLabel, title: selectAllLabel },
  });
  setIcon(selectAllBtn, "list-checks");
  selectAllBtn.disabled = listedPaths.length === 0;
  selectAllBtn.onclick = () => {
    if (allSelected) {
      state.selected.clear();
    } else {
      state.selectMode = true;
      state.selected = new Set(listedPaths);
    }
    refresh();
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
  delSel.disabled = state.selected.size === 0;
  delSel.onclick = () => {
    const paths = Array.from(state.selected);
    new ConfirmModal(
      plugin.app,
      "Delete selected captures?",
      `Trash ${paths.length} capture note(s). They will not come back from the inbox.`,
      () => {
        void plugin.captures
          .deletePaths(paths)
          .then((n) => {
            state.selected.clear();
            state.selectMode = false;
            new Notice(`Deleted ${n}.`, 4000);
            refresh();
          })
          .catch(noticeError);
      },
    ).open();
  };

  if (!plugin.captures.all().length) {
    parent.createDiv({
      cls: "pawn-empty",
      text: "No research captures waiting. Use the browser extension Research target.",
    });
    return;
  }
  if (!listed.length) {
    parent.createDiv({
      cls: "pawn-empty",
      text: "No captures match this filter.",
    });
    return;
  }

  const list = parent.createDiv({ cls: "pawn-inbox-list" });
  for (const group of groupCapturesByDay(listed)) {
    list.createDiv({ cls: "pawn-inbox-day", text: group.label });
    for (const item of group.items) {
      renderCaptureRow(list, plugin, item, state, refresh);
    }
  }
}
