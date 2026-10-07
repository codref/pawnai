import {
  Component,
  ItemView,
  MarkdownRenderer,
  Notice,
  Platform,
  TFile,
  WorkspaceLeaf,
  setIcon,
} from "obsidian";
import { resolveActiveMarkdownFile } from "../active";
import { ChatImage, ModelChoice, NoteContext, ServerUnreachable } from "../api";
import { PromptCommand, renderPrompt } from "../commands/PromptCommands";
import { renderInbox } from "../inbox/InboxView";
import { captureIdea, noticeError } from "../inbox/ideas";
import { JobFilter, renderJobCard, renderJobsList } from "../jobs/JobsView";
import type PawnPlugin from "../main";
import { ContextBar, ContextSnapshot, resolveDroppedNote } from "./ContextBar";
import { exportExcalidrawImage, fileIsExcalidraw } from "./excalidraw";
import { bytesToBase64, isImageName, mimeForName, noteExcalidrawTargets, noteImageTargets } from "./noteImages";
import { renderMessageActions, renderUserMessageActions } from "./MessageActions";
import {
  ChatMessage,
  conversationLabel,
  newId,
  noteConversationId,
  SelectionRef,
} from "./conversations";

export const PAWN_CHAT_VIEW = "pawn-chat";

type TabId = "chat" | "jobs" | "inbox";

interface Pending {
  abort: AbortController;
  progress: string[];
  el: HTMLElement | null;
}

interface SlashItem {
  slug: string;
  label: string;
  hint: string;
  run: () => void;
}

export interface OpenChatOptions {
  tab?: TabId;
  focusJob?: string;
  conversation?: string;
  prefill?: string;
}

export class PawnChatView extends ItemView {
  private tab: TabId = "chat";
  private conversationId = "";
  private pinned = false;
  private background = false;
  private pending: Pending | null = null;
  private jobsFilter: JobFilter = "all";
  private focusJob: string | null = null;
  private context: ContextBar;
  private threadEl: HTMLElement | null = null;
  private bodyEl: HTMLElement | null = null;
  private composer: HTMLTextAreaElement | null = null;
  private slashEl: HTMLElement | null = null;
  private slashItems: SlashItem[] = [];
  private slashIndex = 0;
  private unsubscribeJobs: (() => void) | null = null;
  private unsubscribeInbox: (() => void) | null = null;
  private unsubscribeItems: (() => void) | null = null;
  private fileInput: HTMLInputElement | null = null;
  private rereadButton: HTMLButtonElement | null = null;
  private keyboardFrame = 0;
  private keyboardTimer = 0;
  private keyboardInsetApplied = false;
  /** Keep the thread pinned to the bottom while the reply streams / paints. */
  private stickToBottom = true;
  /** Offer to switch when a draft blocks auto-follow of the open note. */
  private pendingSwitch: { conversationId: string; label: string } | null = null;
  /** Composer text restored across full re-renders. */
  private draftText = "";
  private switchBannerEl: HTMLElement | null = null;
  private modelChoices: ModelChoice[] = [];
  private backgroundModel = "";
  private modelButton: HTMLButtonElement | null = null;
  private modelButtonLabel: HTMLElement | null = null;
  private modelPop: HTMLElement | null = null;
  private modelPopCloser: (() => void) | null = null;
  /** Pictures dropped on the composer; sent as question images. */
  private pendingImages: { filename: string; mediaType: string; data: ArrayBuffer }[] = [];
  /** One-shot: recaption images this conversation has already seen. */
  private rereadImages = false;

  constructor(
    leaf: WorkspaceLeaf,
    private plugin: PawnPlugin,
  ) {
    super(leaf);
    this.context = new ContextBar(this.app, plugin.settings.autoIncludeActiveNote);
  }

  getViewType(): string {
    return PAWN_CHAT_VIEW;
  }

  getDisplayText(): string {
    return "Pawn";
  }

  getIcon(): string {
    return "chess-king";
  }

  async onOpen(): Promise<void> {
    try {
      this.setup();
    } catch (e) {
      this.showFatal(e);
    }
  }

  private setup(): void {
    this.containerEl.addClass("pawn-view");
    this.conversationId = this.defaultConversation();
    this.unsubscribeJobs = this.plugin.jobs.onChange(() => this.onJobsChanged());
    this.unsubscribeInbox = this.plugin.inbox?.onChange(() => this.onInboxChanged());
    this.unsubscribeItems = this.plugin.items?.onChange(() => this.onInboxChanged());
    this.registerEvent(
      this.app.workspace.on("active-leaf-change", () => this.onActiveNoteChanged()),
    );
    let timer: number | null = null;
    this.registerDomEvent(document, "selectionchange", () => {
      if (timer != null) window.clearTimeout(timer);
      timer = window.setTimeout(() => this.context.refresh(), 300);
    });
    this.bindMobileKeyboard();
    this.render();
    void this.loadModels();
  }

  private async loadModels(): Promise<void> {
    try {
      const page = await this.plugin.client.listModels();
      this.modelChoices = page.models ?? [];
      this.backgroundModel = page.background || page.default || "";
    } catch {
      return;
    }
    this.refreshModelButton();
  }

  private selectedModel(): string {
    const stored = this.plugin.conversations.get(this.conversationId).model;
    return stored || this.backgroundModel || this.modelChoices[0]?.id || "";
  }

  private refreshModelButton(): void {
    const label = this.modelButtonLabel;
    const button = this.modelButton;
    if (!label || !button) return;
    const current = this.selectedModel();
    const ids = this.modelChoices.map((c) => c.id);
    const name = current ? modelChipLabel(current, ids) : "Model";
    const choice = this.currentChoice();
    const effort = choice?.reasoning?.length ? REASONING_LABEL[this.effectiveReasoning()] : "";
    const text = [name, effort].filter(Boolean).join(" ");
    label.setText(text);
    const route = choice?.routes?.length ? ROUTE_LABEL[this.effectiveRoute()] : "";
    const vision = choice?.vision ? "Vision" : "";
    button.title = [current, effort, route, vision].filter(Boolean).join(" · ") || "Model";
    button.setAttr("aria-label", button.title);
    this.paintRereadButton();
  }

  private currentChoice(): ModelChoice | undefined {
    const id = this.selectedModel();
    return this.modelChoices.find((choice) => choice.id === id);
  }

  private effectiveReasoning(): string {
    const stored = this.plugin.conversations.get(this.conversationId).reasoning;
    return stored || this.currentChoice()?.reasoning_default || "low";
  }

  private effectiveRoute(): string {
    const stored = this.plugin.conversations.get(this.conversationId).route;
    return stored || this.currentChoice()?.route_default || "balanced";
  }

  private tuningFields(): { reasoning?: string; route?: string } {
    const choice = this.currentChoice();
    return {
      reasoning: choice?.reasoning?.length ? this.effectiveReasoning() : undefined,
      route: choice?.routes?.length ? this.effectiveRoute() : undefined,
    };
  }

  private closeModelPop(): void {
    this.modelPopCloser?.();
    this.modelPopCloser = null;
    this.modelPop?.remove();
    this.modelPop = null;
  }

  private openModelPop(): void {
    const anchor = this.modelButton;
    if (!anchor) return;
    if (this.modelPop) {
      this.closeModelPop();
      return;
    }
    const pop = document.body.createDiv({ cls: "pawn-model-pop" });
    this.modelPop = pop;
    const choice = this.currentChoice();
    const ids = this.modelChoices.map((c) => c.id);
    const current = this.selectedModel();
    this.addPopRow(pop, "Model", current ? modelChipLabel(current, ids) : "None", () => {
      this.openSubmenu(
        pop,
        ids.length
          ? ids.map((id) => ({
              value: id,
              label: modelMenuLabel(id, this.modelChoices.some((c) => c.id === id && c.vision)),
            }))
          : [{ value: "", label: "No models configured", disabled: true }],
        current,
        (id) => {
          if (!id) return;
          this.plugin.conversations.setModel(this.conversationId, id);
          this.refreshModelButton();
          this.closeModelPop();
        },
      );
    });
    if (choice?.reasoning?.length) {
      const effort = this.effectiveReasoning();
      this.addPopRow(pop, "Effort", REASONING_LABEL[effort] ?? effort, () => {
        this.openSubmenu(
          pop,
          (choice.reasoning ?? []).map((value) => ({
            value,
            label: REASONING_LABEL[value] ?? value,
          })),
          effort,
          (value) => {
            this.plugin.conversations.setTuning(this.conversationId, { reasoning: value });
            this.refreshModelButton();
            this.closeModelPop();
          },
        );
      });
    }
    if (choice?.routes?.length) {
      const route = this.effectiveRoute();
      this.addPopRow(pop, "Route", ROUTE_LABEL[route] ?? route, () => {
        this.openSubmenu(
          pop,
          (choice.routes ?? []).map((value) => ({
            value,
            label: ROUTE_LABEL[value] ?? value,
          })),
          route,
          (value) => {
            this.plugin.conversations.setTuning(this.conversationId, { route: value });
            this.refreshModelButton();
            this.closeModelPop();
          },
        );
      });
    }

    const rect = anchor.getBoundingClientRect();
    pop.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - 248))}px`;
    pop.style.top = "0px";
    window.requestAnimationFrame(() => {
      if (this.modelPop !== pop) return;
      const top = Math.max(8, rect.top - pop.offsetHeight - 6);
      pop.style.top = `${top}px`;
    });

    const onDown = (ev: MouseEvent) => {
      const target = ev.target;
      if (!(target instanceof Node)) return;
      if (pop.contains(target) || anchor.contains(target)) return;
      this.closeModelPop();
    };
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") this.closeModelPop();
    };
    window.setTimeout(() => {
      document.addEventListener("mousedown", onDown, true);
      document.addEventListener("keydown", onKey);
    }, 0);
    this.modelPopCloser = () => {
      document.removeEventListener("mousedown", onDown, true);
      document.removeEventListener("keydown", onKey);
    };
  }

  private addPopRow(pop: HTMLElement, key: string, value: string, onOpen: () => void): void {
    const row = pop.createEl("button", {
      cls: "pawn-model-row",
      attr: { type: "button" },
    });
    row.createSpan({ cls: "pawn-model-row-key", text: key });
    row.createSpan({ cls: "pawn-model-row-value", text: value });
    const chevron = row.createSpan({ cls: "pawn-model-row-chevron" });
    setIcon(chevron, "chevron-right");
    row.onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      pop.querySelectorAll(".pawn-model-row.is-open").forEach((el) => el.removeClass("is-open"));
      row.addClass("is-open");
      onOpen();
    };
  }

  private openSubmenu(
    pop: HTMLElement,
    options: { value: string; label: string; disabled?: boolean }[],
    current: string,
    apply: (value: string) => void,
  ): void {
    pop.querySelector(".pawn-model-sub")?.remove();
    const sub = pop.createDiv({ cls: "pawn-model-sub" });
    for (const option of options) {
      const item = sub.createEl("button", {
        cls: "pawn-model-sub-item" + (option.value === current ? " is-active" : ""),
        attr: { type: "button" },
      });
      if (option.disabled) item.disabled = true;
      item.createSpan({ text: option.label });
      if (option.value === current) {
        const mark = item.createSpan({ cls: "pawn-model-sub-check" });
        setIcon(mark, "check");
      }
      item.onclick = (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (option.disabled) return;
        apply(option.value);
      };
    }
    const row = pop.querySelector(".pawn-model-row.is-open");
    if (row instanceof HTMLElement) sub.style.top = `${row.offsetTop}px`;
    const overflow = pop.getBoundingClientRect().right + sub.offsetWidth + 12 > window.innerWidth;
    sub.toggleClass("is-left", overflow);
  }

  async onClose(): Promise<void> {
    this.unsubscribeInbox?.();
    this.unsubscribeItems?.();
    this.unsubscribeJobs?.();
    this.closeModelPop();
    this.pending?.abort.abort();
    this.clearKeyboardInset();
    this.containerEl.empty();
  }

  /**
   * Keep the composer above the soft keyboard.
   *
   * On Android the webview stays full height while the keyboard is up, and
   * side drawers do too. Obsidian writes `--keyboard-height` on the document
   * element (and sometimes shrinks the main app container). This shortens the
   * chat view only when its box still extends past the visible bottom.
   */
  private bindMobileKeyboard(): void {
    if (!Platform.isMobile) return;
    const win = this.containerEl.win;
    const doc = this.containerEl.doc;
    const vv = win.visualViewport;
    const onInset = () => this.scheduleKeyboardInset();
    const observer = new MutationObserver(onInset);
    observer.observe(doc.documentElement, { attributes: true, attributeFilter: ["style"] });
    if (doc.body) observer.observe(doc.body, { attributes: true, attributeFilter: ["style"] });
    vv?.addEventListener("resize", onInset);
    vv?.addEventListener("scroll", onInset);
    this.registerEvent(this.app.workspace.on("layout-change", onInset));
    this.registerDomEvent(win, "resize", onInset);
    this.register(() => {
      observer.disconnect();
      vv?.removeEventListener("resize", onInset);
      vv?.removeEventListener("scroll", onInset);
      if (this.keyboardFrame) win.cancelAnimationFrame(this.keyboardFrame);
      if (this.keyboardTimer) win.clearTimeout(this.keyboardTimer);
      this.keyboardFrame = 0;
      this.keyboardTimer = 0;
      this.clearKeyboardInset();
    });
  }

  private scheduleKeyboardInset(): void {
    const win = this.contentEl.win;
    this.applyKeyboardInset();
    if (this.keyboardFrame) win.cancelAnimationFrame(this.keyboardFrame);
    this.keyboardFrame = win.requestAnimationFrame(() => {
      this.keyboardFrame = 0;
      this.applyKeyboardInset();
    });
    if (this.keyboardTimer) win.clearTimeout(this.keyboardTimer);
    this.keyboardTimer = win.setTimeout(() => {
      this.keyboardTimer = 0;
      this.applyKeyboardInset();
    }, 300);
  }

  private applyKeyboardInset(): void {
    const root = this.contentEl;
    const win = root.win;
    const kb = readKeyboardHeight(root.doc);
    const vv = win.visualViewport;
    const visualInset = vv ? Math.max(0, win.innerHeight - vv.height - vv.offsetTop) : 0;
    if (kb < 50 && visualInset < 50) {
      this.clearKeyboardInset();
      return;
    }
    const vvBottom = vv ? vv.offsetTop + vv.height : win.innerHeight;
    const visibleBottom = Math.min(vvBottom, win.innerHeight - kb);
    const rect = root.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) {
      this.clearKeyboardInset();
      return;
    }
    const parentBottom = root.parentElement?.getBoundingClientRect().bottom ?? rect.bottom;
    const natural = parentBottom - rect.top;
    const wanted = visibleBottom - rect.top;
    if (natural - wanted < 8) {
      this.clearKeyboardInset();
      return;
    }
    const height = Math.max(0, Math.floor(wanted));
    root.style.flexGrow = "0";
    root.style.flexShrink = "0";
    root.style.minHeight = "0";
    root.style.height = `${height}px`;
    root.style.maxHeight = `${height}px`;
    this.keyboardInsetApplied = true;
    this.containerEl.addClass("is-keyboard-open");
    this.scrollThreadToBottom();
  }

  private clearKeyboardInset(): void {
    if (!this.keyboardInsetApplied) return;
    const root = this.contentEl;
    root.style.height = "";
    root.style.maxHeight = "";
    root.style.minHeight = "";
    root.style.flexGrow = "";
    root.style.flexShrink = "";
    this.containerEl.removeClass("is-keyboard-open");
    this.keyboardInsetApplied = false;
  }

  // ── public API used by the plugin ─────────────────────────────────────────

  /** Not named `open`: that is Obsidian's internal View lifecycle method. */
  applyOptions(opts: OpenChatOptions): void {
    if (opts.conversation) this.switchConversation(opts.conversation, true);
    if (opts.tab) this.tab = opts.tab;
    if (opts.focusJob) {
      this.focusJob = opts.focusJob;
      this.jobsFilter = "all";
    }
    if (opts.prefill != null) this.draftText = opts.prefill;
    this.render();
    if (this.tab === "chat") this.composer?.focus();
  }

  addContextFile(file: TFile): void {
    this.tab = "chat";
    this.render();
    this.context.addFile(file);
  }

  async runPrompt(cmd: PromptCommand): Promise<void> {
    this.tab = "chat";
    this.render();
    const snap = this.context.snapshot(false);
    const message = renderPrompt(cmd.prompt, {
      selection: snap.selection?.text,
      noteTitle: snap.activeNote?.basename,
    });
    await this.send(message, { background: cmd.background });
  }

  // ── conversation handling ────────────────────────────────────────────────

  private defaultConversation(): string {
    if (this.plugin.settings.conversationMode === "note") {
      const file = resolveActiveMarkdownFile(this.app);
      if (file) return noteConversationId(file.path);
    }
    return this.plugin.data.lastGlobalConversation || this.newGlobalConversation();
  }

  private newGlobalConversation(): string {
    const id = `chat:${newId()}`;
    this.plugin.data.lastGlobalConversation = id;
    this.plugin.persistSoon();
    return id;
  }

  private switchConversation(id: string, pin: boolean): void {
    this.conversationId = id;
    this.pinned = pin && this.plugin.settings.conversationMode === "note";
    this.pendingSwitch = null;
    if (!id.startsWith("note:")) {
      this.plugin.data.lastGlobalConversation = id;
      this.plugin.persistSoon();
    }
  }

  private composerDraft(): string {
    return (this.composer?.value ?? this.draftText).trim();
  }

  private captureDraft(): void {
    if (this.composer) this.draftText = this.composer.value;
  }

  private restoreDraft(): void {
    if (!this.composer || !this.draftText) return;
    this.composer.value = this.draftText;
    this.autosize();
  }

  private onActiveNoteChanged(): void {
    if (this.plugin.settings.conversationMode === "note" && !this.pinned && !this.pending) {
      const file = resolveActiveMarkdownFile(this.app);
      const next = file ? noteConversationId(file.path) : this.conversationId;
      if (next !== this.conversationId) {
        if (this.composerDraft()) {
          this.pendingSwitch = {
            conversationId: next,
            label: file?.basename ?? "note",
          };
          this.renderComposerBanner();
          this.context.refresh();
          return;
        }
        this.pendingSwitch = null;
        this.conversationId = next;
        this.render();
        return;
      }
    }
    if (this.pendingSwitch) {
      const file = resolveActiveMarkdownFile(this.app);
      const current = file ? noteConversationId(file.path) : null;
      if (!current || current === this.conversationId) {
        this.pendingSwitch = null;
        this.renderComposerBanner();
      } else if (file && this.pendingSwitch.conversationId !== current) {
        this.pendingSwitch = { conversationId: current, label: file.basename };
        this.renderComposerBanner();
      }
    }
    this.context.refresh();
  }

  private acceptPendingSwitch(): void {
    if (!this.pendingSwitch) return;
    this.captureDraft();
    this.switchConversation(this.pendingSwitch.conversationId, true);
    this.render();
  }

  private dismissPendingSwitch(): void {
    this.pendingSwitch = null;
    this.renderComposerBanner();
  }

  private onJobsChanged(): void {
    if (this.tab === "jobs") this.renderBody();
    else if (this.tab === "chat") this.renderThread();
    this.renderTabs();
  }

  private onInboxChanged(): void {
    if (this.tab === "inbox") this.renderBody();
    this.renderTabs();
  }

  // ── rendering ────────────────────────────────────────────────────────────

  private tabsEl: HTMLElement | null = null;
  private renderScope: Component | null = null;

  /** Fresh owner for rendered Markdown so re-renders don't accumulate children. */
  private freshScope(): Component {
    if (this.renderScope) this.removeChild(this.renderScope);
    this.renderScope = this.addChild(new Component());
    return this.renderScope;
  }

  private render(): void {
    this.captureDraft();
    const root = this.contentEl;
    root.empty();
    root.addClass("pawn-root");
    try {
      this.renderHeader(root.createDiv({ cls: "pawn-header" }));
      this.bodyEl = root.createDiv({ cls: "pawn-body" });
      this.renderBody();
    } catch (e) {
      this.showFatal(e);
    }
  }

  private showFatal(e: unknown): void {
    console.error("Pawn chat view failed to render", e);
    const root = this.contentEl;
    root.empty();
    const box = root.createDiv({ cls: "pawn-msg is-error" });
    box.createEl("p", {
      text: `Pawn ${this.plugin.manifest.version}: chat view failed to render.`,
    });
    box.createEl("pre", {
      text: e instanceof Error ? `${e.message}\n\n${e.stack ?? ""}` : String(e),
    });
  }

  private renderHeader(header: HTMLElement): void {
    const row = header.createDiv({ cls: "pawn-header-row" });
    const select = row.createEl("select", { cls: "dropdown pawn-conv-select" });
    const options: Array<[string, string]> = [];
    const active = resolveActiveMarkdownFile(this.app);
    if (active) options.push([noteConversationId(active.path), `Note: ${active.basename}`]);
    for (const conv of this.plugin.conversations.recent()) {
      if (!options.some(([id]) => id === conv.id)) {
        const prefix = conv.id.startsWith("note:") ? "Note: " : "";
        options.push([conv.id, prefix + conversationLabel(conv.id, conv.title)]);
      }
    }
    if (!options.some(([id]) => id === this.conversationId)) {
      const conv = this.plugin.conversations.get(this.conversationId);
      options.unshift([this.conversationId, conversationLabel(this.conversationId, conv.title)]);
    }
    for (const [id, label] of options) select.createEl("option", { value: id, text: label });
    select.value = this.conversationId;
    select.onchange = () => {
      this.switchConversation(select.value, true);
      this.render();
    };

    const iconBtn = (icon: string, label: string, onClick: () => void) => {
      const b = row.createEl("button", { cls: "clickable-icon", attr: { "aria-label": label } });
      setIcon(b, icon);
      b.onclick = onClick;
      return b;
    };
    iconBtn("plus", "New chat", () => {
      this.switchConversation(this.newGlobalConversation(), true);
      this.tab = "chat";
      this.render();
    });
    iconBtn("rotate-ccw", "Reset conversation (clears Pawn's memory of it)", () => {
      void this.send("/reset");
    });
    if (this.pinned) {
      iconBtn("pin-off", "Follow the active note again", () => {
        this.pinned = false;
        this.conversationId = this.defaultConversation();
        this.render();
      });
    }

    this.tabsEl = header.createDiv({ cls: "pawn-tabs" });
    this.renderTabs();
  }

  private renderTabs(): void {
    const tabs = this.tabsEl;
    if (!tabs) return;
    tabs.empty();
    const { active, review } = this.plugin.jobs.counts();
    const mk = (id: TabId, label: string) => {
      const b = tabs.createEl("button", { text: label });
      b.toggleClass("is-active", this.tab === id);
      b.onclick = () => {
        this.tab = id;
        if (id === "chat") this.focusJob = null;
        this.render();
      };
    };
    mk("chat", "Chat");
    const badge = [active ? `${active} running` : "", review ? `${review} review` : ""]
      .filter(Boolean)
      .join(", ");
    mk("jobs", badge ? `Jobs (${badge})` : "Jobs");
    const waiting =
      (this.plugin.items?.attention() ?? 0) + (this.plugin.inbox?.attention() ?? 0);
    mk("inbox", waiting ? `Inbox (${waiting})` : "Inbox");
    const dot = tabs.createSpan({
      cls: this.plugin.jobs.online ? "pawn-dot is-online" : "pawn-dot is-offline",
      attr: { "aria-label": this.plugin.jobs.online ? "Server reachable" : "Server offline" },
    });
    dot.setText("●");
  }

  private renderBody(): void {
    this.captureDraft();
    const body = this.bodyEl;
    if (!body) return;
    body.empty();
    if (this.tab === "inbox") {
      void this.plugin.items?.refresh();
      renderInbox(body.createDiv({ cls: "pawn-jobs" }), this.plugin);
      return;
    }
    if (this.tab === "jobs") {
      const jobs = body.createDiv({ cls: "pawn-jobs" });
      renderJobsList(jobs, this.plugin, this.freshScope(), {
        filter: this.jobsFilter,
        conversation: this.conversationId,
        focusJob: this.focusJob,
        onFilter: (f) => {
          this.jobsFilter = f;
          this.renderBody();
        },
        onOpenConversation: (conv) => {
          this.switchConversation(conv, true);
          this.tab = "chat";
          this.render();
        },
      });
      return;
    }
    this.threadEl = body.createDiv({ cls: "pawn-thread" });
    this.registerDomEvent(this.threadEl, "scroll", () => {
      const t = this.threadEl;
      if (!t) return;
      this.stickToBottom = t.scrollHeight - t.scrollTop - t.clientHeight < 40;
    });
    this.renderThread();
    this.renderComposer(body.createDiv({ cls: "pawn-composer" }));
    this.registerDrop(body);
  }

  private scrollThreadToBottom(force = false): void {
    const thread = this.threadEl;
    if (!thread) return;
    if (!force && !this.stickToBottom && !this.pending) return;
    thread.scrollTop = thread.scrollHeight;
    const win = thread.win;
    win.requestAnimationFrame(() => {
      if (!this.threadEl) return;
      if (!force && !this.stickToBottom && !this.pending) return;
      this.threadEl.scrollTop = this.threadEl.scrollHeight;
    });
  }

  private renderThread(): void {
    const thread = this.threadEl;
    if (!thread || this.tab !== "chat") return;
    if (this.pending) this.stickToBottom = true;
    thread.empty();
    const scope = this.freshScope();
    const conv = this.plugin.conversations.get(this.conversationId);
    if (!conv.messages.length && !this.pending) {
      const empty = thread.createDiv({ cls: "pawn-empty" });
      empty.createEl("p", {
        text: this.conversationId.startsWith("note:")
          ? "Ask Pawn about this note. Unpin it with × on the file chip, and pin it again to include it."
          : "Ask Pawn anything. Pin the open note from the chip, or add others with @ or +.",
      });
      empty.createEl("p", {
        cls: "pawn-hint",
        text: "Type / for prompt commands. Toggle Background for long jobs that report back here.",
      });
    }
    for (const msg of conv.messages) this.renderMessage(thread, msg, scope);
    if (this.pending) {
      const p = thread.createDiv({ cls: "pawn-msg is-assistant is-pending" });
      this.pending.el = p;
      this.renderPending();
    }
    this.scrollThreadToBottom();
  }

  private renderMessage(parent: HTMLElement, msg: ChatMessage, scope: Component): void {
    if (msg.role === "job" && msg.jobId) {
      const job = this.plugin.jobs.get(msg.jobId);
      const wrap = parent.createDiv({ cls: "pawn-msg is-job" });
      if (job) renderJobCard(wrap, job, this.plugin, scope, { compact: true });
      else wrap.createDiv({ cls: "pawn-job-card", text: `Job ${msg.jobId}` });
      return;
    }
    const el = parent.createDiv({ cls: `pawn-msg is-${msg.role}` });
    if (msg.role === "user") {
      el.createDiv({ cls: "pawn-msg-text", text: msg.content });
      const ctx = [
        ...(msg.selection ? ["selection"] : []),
        ...(msg.context ?? []).map((p) => p.replace(/\.md$/i, "").split("/").pop() ?? p),
      ];
      if (ctx.length) el.createDiv({ cls: "pawn-msg-context", text: ctx.join(" · ") });
      renderUserMessageActions(el, {
        text: msg.content,
        onReuse: () => this.reuseInComposer(msg.content),
      });
      return;
    }
    if (msg.progress?.length) {
      const det = el.createEl("details", { cls: "pawn-progress-log" });
      det.createEl("summary", { text: `${msg.progress.length} step(s)` });
      for (const line of msg.progress) det.createDiv({ text: line });
    }
    const body = el.createDiv({ cls: "pawn-msg-text markdown-rendered" });
    if (msg.role === "error") {
      body.setText(msg.content);
      return;
    }
    void MarkdownRenderer.render(this.app, msg.content, body, msg.notePath ?? "", scope).then(
      () => this.scrollThreadToBottom(),
    );
    renderMessageActions(el, {
      app: this.app,
      text: msg.content,
      selection: msg.selection,
      notePath: msg.notePath,
      saveFolder: `${this.plugin.settings.agentRoot}/Notes`,
    });
  }

  private reuseInComposer(text: string): void {
    this.draftText = text;
    if (this.tab !== "chat" || !this.composer) {
      this.tab = "chat";
      this.render();
      this.composer?.focus();
      return;
    }
    this.composer.value = text;
    this.autosize();
    this.composer.focus();
    this.composer.selectionStart = this.composer.selectionEnd = text.length;
  }

  private renderPending(): void {
    const p = this.pending;
    if (!p?.el) return;
    p.el.empty();
    const row = p.el.createDiv({ cls: "pawn-pending-row" });
    row.createSpan({ cls: "pawn-spinner" });
    row.createSpan({ text: p.progress[p.progress.length - 1] ?? "Thinking…" });
    this.scrollThreadToBottom(true);
  }

  private renderComposer(wrap: HTMLElement): void {
    this.switchBannerEl = wrap.createDiv({ cls: "pawn-switch-banner is-hidden" });
    this.renderComposerBanner();
    this.context.mount(wrap);
    this.syncImageChips();
    this.slashEl = wrap.createDiv({ cls: "pawn-slash is-hidden" });
    const ta = wrap.createEl("textarea", {
      cls: "pawn-input",
      attr: { rows: "2", placeholder: "Message Pawn…  (/ commands, @ notes)" },
    });
    this.composer = ta;
    this.restoreDraft();
    ta.addEventListener("input", () => this.onComposerInput());
    ta.addEventListener("keydown", (ev) => this.onComposerKey(ev));
    ta.addEventListener("paste", (ev) => this.onComposerPaste(ev));
    ta.addEventListener("focus", () => {
      this.context.refresh();
      this.scheduleKeyboardInset();
    });
    ta.addEventListener("blur", () => this.scheduleKeyboardInset());

    const row = wrap.createDiv({ cls: "pawn-composer-row" });
    const bgLabel = row.createEl("label", { cls: "pawn-bg-toggle" });
    const bg = bgLabel.createEl("input", { type: "checkbox" });
    bg.checked = this.background;
    bg.onchange = () => (this.background = bg.checked);
    bgLabel.appendText(" Background");
    bgLabel.setAttr("aria-label", "Run as a background job; the result shows up here and in Jobs");

    const modelButton = row.createEl("button", {
      cls: "pawn-model-btn",
      attr: { type: "button", "aria-haspopup": "dialog" },
    });
    const modelLabel = modelButton.createSpan({ cls: "pawn-model-btn-label", text: "Model" });
    const chevron = modelButton.createSpan({ cls: "pawn-model-btn-chevron" });
    setIcon(chevron, "chevron-down");
    this.modelButton = modelButton;
    this.modelButtonLabel = modelLabel;
    this.refreshModelButton();
    modelButton.onclick = () => this.openModelPop();

    row.createDiv({ cls: "pawn-spacer" });
    const reread = row.createEl("button", {
      cls: "clickable-icon pawn-attach pawn-reread-btn",
      attr: {
        type: "button",
        "aria-label": "Read images in attached notes again",
        "aria-pressed": "false",
      },
    });
    setIcon(reread, "refresh-cw");
    this.rereadButton = reread;
    reread.onclick = () => this.toggleReread();
    this.paintRereadButton();
    const attach = row.createEl("button", {
      cls: "clickable-icon pawn-attach",
      attr: { "aria-label": "Upload a file to Pawn" },
    });
    setIcon(attach, "paperclip");
    this.fileInput = row.createEl("input", { type: "file", cls: "pawn-hidden" });
    this.fileInput.multiple = true;
    attach.onclick = () => this.fileInput?.click();
    this.fileInput.onchange = () => {
      const files = Array.from(this.fileInput?.files ?? []);
      void this.acceptDroppedFiles(files);
      if (this.fileInput) this.fileInput.value = "";
    };

    if (this.pending) {
      const stop = row.createEl("button", { text: "Stop", cls: "mod-warning pawn-send" });
      stop.onclick = () => this.stop();
    } else {
      const send = row.createEl("button", { text: "Send", cls: "mod-cta pawn-send" });
      send.onclick = () => void this.sendFromComposer();
    }
  }

  private renderComposerBanner(): void {
    const el = this.switchBannerEl;
    if (!el) return;
    el.empty();
    if (!this.pendingSwitch) {
      el.addClass("is-hidden");
      return;
    }
    el.removeClass("is-hidden");
    el.createSpan({
      text: `Open note is “${this.pendingSwitch.label}” — Switch chat?`,
    });
    const actions = el.createDiv({ cls: "pawn-switch-banner-actions" });
    const go = actions.createEl("button", { text: "Switch", cls: "mod-cta" });
    go.onclick = () => this.acceptPendingSwitch();
    const dismiss = actions.createEl("button", { text: "Dismiss" });
    dismiss.onclick = () => this.dismissPendingSwitch();
  }

  private autosize(): void {
    const ta = this.composer;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 240)}px`;
  }

  // ── composer: slash commands and @ mentions ──────────────────────────────

  private onComposerInput(): void {
    const ta = this.composer;
    if (!ta) return;
    this.draftText = ta.value;
    this.autosize();
    if (!ta.value.trim() && this.pendingSwitch) {
      this.pendingSwitch = null;
      this.renderComposerBanner();
    }
    const pos = ta.selectionStart;
    if (pos > 0 && ta.value[pos - 1] === "@" && (pos === 1 || /\s/.test(ta.value[pos - 2]))) {
      ta.value = ta.value.slice(0, pos - 1) + ta.value.slice(pos);
      ta.selectionStart = ta.selectionEnd = pos - 1;
      this.draftText = ta.value;
      this.context.openPicker();
      return;
    }
    const m = ta.value.match(/^\/(\S*)$/);
    if (m) this.showSlash(m[1].toLowerCase());
    else this.hideSlash();
  }

  private slashCandidates(): SlashItem[] {
    const builtins: SlashItem[] = [
      {
        slug: "idea",
        label: "/idea",
        hint: "Save one line under Ideas/",
        run: () => this.prefillSlash("/idea "),
      },
      {
        slug: "goal",
        label: "/goal",
        hint: "Add an active goal thread",
        run: () => this.prefillSlash("/goal "),
      },
      {
        slug: "park",
        label: "/park",
        hint: "Park a goal thread",
        run: () => this.prefillSlash("/park "),
      },
      {
        slug: "goal-apply",
        label: "/goal apply",
        hint: "Write Goals.md from the latest proposal",
        run: () => void this.send("/goal apply"),
      },
      {
        slug: "goals",
        label: "/goals",
        hint: "List active and parked threads",
        run: () => void this.send("/goals"),
      },
      {
        slug: "inbox",
        label: "/inbox",
        hint: "List items that need a tap",
        run: () => void this.send("/inbox"),
      },
      { slug: "reset", label: "/reset", hint: "Clear this conversation", run: () => void this.send("/reset") },
      ...(this.modelIsVision()
        ? [
            {
              slug: "vision-refresh",
              label: "/vision refresh",
              hint: "Read attached images again on the next message",
              run: () => this.armReread(),
            },
          ]
        : []),
      {
        slug: "model",
        label: "/model",
        hint: "Show or set the background model",
        run: () => this.prefillSlash("/model "),
      },
      {
        slug: "new",
        label: "/new",
        hint: "Start a new chat",
        run: () => {
          this.switchConversation(this.newGlobalConversation(), true);
          this.render();
        },
      },
      {
        slug: "bg",
        label: "/bg",
        hint: "Toggle background mode",
        run: () => {
          this.background = !this.background;
          this.render();
        },
      },
      {
        slug: "jobs",
        label: "/jobs",
        hint: "Show background jobs",
        run: () => {
          this.tab = "jobs";
          this.render();
        },
      },
    ];
    const prompts: SlashItem[] = this.plugin.prompts.list().map((cmd) => ({
      slug: cmd.slug,
      label: `/${cmd.slug}`,
      hint: cmd.name,
      run: () => {
        const snap = this.context.snapshot(false);
        const ta = this.composer;
        if (!ta) return;
        ta.value = renderPrompt(cmd.prompt, {
          selection: snap.selection?.text,
          noteTitle: snap.activeNote?.basename,
        });
        this.draftText = ta.value;
        if (cmd.background) this.background = true;
        this.autosize();
        ta.focus();
      },
    }));
    return [...prompts, ...builtins];
  }

  private showSlash(filter: string): void {
    const el = this.slashEl;
    if (!el) return;
    this.slashItems = this.slashCandidates().filter(
      (i) => i.slug.includes(filter) || i.hint.toLowerCase().includes(filter),
    );
    this.slashIndex = Math.min(this.slashIndex, Math.max(0, this.slashItems.length - 1));
    el.empty();
    if (!this.slashItems.length) {
      this.hideSlash();
      return;
    }
    el.removeClass("is-hidden");
    this.slashItems.forEach((item, idx) => {
      const row = el.createDiv({ cls: "pawn-slash-item" });
      row.toggleClass("is-selected", idx === this.slashIndex);
      row.createSpan({ cls: "pawn-slash-label", text: item.label });
      row.createSpan({ cls: "pawn-slash-hint", text: item.hint });
      row.onmousedown = (ev) => {
        ev.preventDefault();
        this.pickSlash(idx);
      };
    });
  }

  private hideSlash(): void {
    this.slashItems = [];
    this.slashIndex = 0;
    this.slashEl?.addClass("is-hidden");
  }

  private pickSlash(idx: number): void {
    const item = this.slashItems[idx];
    if (!item) return;
    if (this.composer) {
      this.composer.value = "";
      this.draftText = "";
    }
    this.hideSlash();
    item.run();
  }

  private prefillSlash(text: string): void {
    const ta = this.composer;
    if (!ta) return;
    ta.value = text;
    this.draftText = text;
    this.autosize();
    ta.focus();
    ta.selectionStart = ta.selectionEnd = ta.value.length;
  }

  private onComposerKey(ev: KeyboardEvent): void {
    if (this.slashItems.length) {
      if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
        ev.preventDefault();
        const n = this.slashItems.length;
        this.slashIndex = (this.slashIndex + (ev.key === "ArrowDown" ? 1 : n - 1)) % n;
        this.showSlash((this.composer?.value ?? "").slice(1).toLowerCase());
        return;
      }
      if (ev.key === "Enter" || ev.key === "Tab") {
        ev.preventDefault();
        this.pickSlash(this.slashIndex);
        return;
      }
      if (ev.key === "Escape") {
        this.hideSlash();
        return;
      }
    }
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      void this.sendFromComposer();
    }
  }

  // ── drag and drop / uploads ──────────────────────────────────────────────

  private registerDrop(el: HTMLElement): void {
    el.addEventListener("dragover", (ev) => {
      ev.preventDefault();
      el.addClass("is-dragover");
    });
    el.addEventListener("dragleave", () => el.removeClass("is-dragover"));
    el.addEventListener("drop", (ev) => {
      ev.preventDefault();
      el.removeClass("is-dragover");
      const dt = ev.dataTransfer;
      if (!dt) return;
      if (dt.files && dt.files.length) {
        void this.acceptDroppedFiles(Array.from(dt.files));
        return;
      }
      const text = dt.getData("text/plain");
      let added = 0;
      for (const line of text.split(/\r?\n/)) {
        const file = resolveDroppedNote(this.app, line);
        if (file) {
          if (file.extension === "md") this.context.addFile(file);
          else void this.plugin.uploadVaultFile(file, this.conversationId);
          added++;
        }
      }
      if (!added && text.trim()) new Notice("Drop notes from the file explorer, or files from disk.");
    });
  }

  private modelIsVision(): boolean {
    return Boolean(this.currentChoice()?.vision);
  }

  private toggleReread(): void {
    if (!this.modelIsVision()) {
      new Notice("This model can't see images. Pick a vision model in the model menu.");
      return;
    }
    this.rereadImages = !this.rereadImages;
    this.paintRereadButton();
  }

  private armReread(): void {
    if (!this.modelIsVision()) {
      new Notice("This model can't see images. Pick a vision model in the model menu.");
      return;
    }
    this.rereadImages = true;
    this.paintRereadButton();
  }

  private paintRereadButton(): void {
    const button = this.rereadButton;
    if (!button) return;
    const vision = this.modelIsVision();
    button.toggleClass("is-hidden", !vision);
    button.toggleClass("is-active", vision && this.rereadImages);
    button.setAttr("aria-pressed", this.rereadImages ? "true" : "false");
    button.setAttr(
      "aria-label",
      this.rereadImages
        ? "Images on the next message will be read again"
        : "Read images in attached notes again",
    );
  }

  private syncImageChips(): void {
    this.context.setImageChips(
      this.pendingImages.map((image, index) => ({
        label: image.filename,
        title: image.filename,
        onRemove: () => {
          this.pendingImages.splice(index, 1);
          this.syncImageChips();
        },
      })),
    );
  }

  private onComposerPaste(ev: ClipboardEvent): void {
    const files = clipboardImageFiles(
      ev.clipboardData,
      this.pendingImages.map((image) => image.filename),
    );
    if (!files.length) return;
    ev.preventDefault();
    void this.attachImages(files).then(() => this.composer?.focus());
  }

  private async acceptDroppedFiles(files: File[]): Promise<void> {
    const images: File[] = [];
    const rest: File[] = [];
    for (const file of files) {
      if (isImageName(file.name, file.type)) images.push(file);
      else rest.push(file);
    }
    if (images.length) await this.attachImages(images);
    if (rest.length) await this.uploadFiles(rest);
  }

  private async attachImages(files: File[]): Promise<void> {
    if (!this.modelIsVision()) {
      new Notice("This model can't see images. Pick a vision model in the model menu.");
      return;
    }
    const maxBytes = 4 * 1024 * 1024;
    for (const file of files) {
      if (this.pendingImages.length >= 4) {
        new Notice("Four images can be attached to one message.");
        break;
      }
      if (file.size > maxBytes) {
        new Notice(`${file.name} is larger than 4MB.`);
        continue;
      }
      this.pendingImages.push({
        filename: file.name,
        mediaType: file.type || mimeForName(file.name),
        data: await file.arrayBuffer(),
      });
    }
    this.render();
  }

  private async uploadFiles(files: File[]): Promise<void> {
    for (const f of files) {
      const job = await this.plugin.uploadData(
        f.name,
        await f.arrayBuffer(),
        f.type,
        this.conversationId,
      );
      if (job) this.addJobMessage(job.id);
    }
  }

  addJobMessage(jobId: string): void {
    this.plugin.conversations.add(this.conversationId, {
      id: newId(),
      role: "job",
      content: "",
      jobId,
      createdAt: Date.now(),
    });
    this.renderThread();
  }

  // ── sending ──────────────────────────────────────────────────────────────

  private async sendFromComposer(): Promise<void> {
    const ta = this.composer;
    if (!ta) return;
    const text = ta.value.trim();
    if (!text && !this.pendingImages.length) return;
    if (this.background && this.pendingImages.length) {
      new Notice("Background jobs don't include images. Uncheck Background, or remove the images.");
      return;
    }
    if (this.pendingImages.length && !this.modelIsVision()) {
      new Notice("This model can't see images. Pick a vision model in the model menu.");
      return;
    }
    ta.value = "";
    this.draftText = "";
    this.pendingSwitch = null;
    this.autosize();
    this.hideSlash();
    await this.send(text, { background: this.background });
  }

  private async contextImages(files: TFile[]): Promise<ChatImage[]> {
    const out: ChatImage[] = [];
    const seen = new Set<string>();
    const diagrams: { file: TFile; fragment: string }[] = [];
    const seenDiagrams = new Set<string>();
    const maxBytes = 4 * 1024 * 1024;
    const queueDiagram = (file: TFile, fragment: string) => {
      const key = `${file.path}\0${fragment}`;
      if (seenDiagrams.has(key)) return;
      seenDiagrams.add(key);
      diagrams.push({ file, fragment });
    };
    for (const file of files) {
      if (fileIsExcalidraw(this.app, file)) queueDiagram(file, "");
      let body = "";
      try {
        body = await this.app.vault.cachedRead(file);
      } catch {
        continue;
      }
      for (const target of noteImageTargets(body)) {
        if (out.length >= 12) break;
        const dest =
          this.app.metadataCache.getFirstLinkpathDest(target, file.path) ??
          this.app.vault.getAbstractFileByPath(target);
        if (!(dest instanceof TFile) || seen.has(dest.path) || !isImageName(dest.name)) continue;
        seen.add(dest.path);
        const data = await this.app.vault.readBinary(dest);
        if (data.byteLength > maxBytes) {
          new Notice(`${dest.name} is larger than 4MB and was skipped.`);
          continue;
        }
        out.push({
          filename: dest.name,
          media_type: mimeForName(dest.name),
          data_base64: bytesToBase64(data),
          role: "context",
        });
      }
      for (const embed of noteExcalidrawTargets(body)) {
        const dest =
          this.app.metadataCache.getFirstLinkpathDest(embed.target, file.path) ??
          this.app.vault.getAbstractFileByPath(embed.target);
        if (!(dest instanceof TFile) || !fileIsExcalidraw(this.app, dest)) continue;
        queueDiagram(dest, embed.fragment);
      }
    }
    let missing = false;
    let failed = false;
    for (const item of diagrams) {
      if (out.length >= 12) break;
      const exported = await exportExcalidrawImage(this.app, item.file, item.fragment);
      if (exported.status === "ok") {
        out.push({
          filename: exported.filename,
          media_type: "image/png",
          data_base64: exported.dataBase64,
          role: "context",
        });
        continue;
      }
      if (exported.status === "missing") missing = true;
      else if (exported.status === "too-large") {
        new Notice(`${exported.name} is larger than 4MB and was skipped.`);
      } else if (exported.status === "failed") failed = true;
    }
    if (missing) {
      new Notice("Excalidraw is not enabled, so diagrams were not included.");
    } else if (failed) {
      new Notice("A diagram could not be exported and was skipped.");
    }
    return out;
  }

  private async noteContext(file: TFile): Promise<NoteContext> {
    if (!this.plugin.settings.sendLocalNoteContent) return { path: file.path };
    return { path: file.path, content: await this.app.vault.cachedRead(file) };
  }

  async send(text: string, opts: { background?: boolean } = {}): Promise<void> {
    const idea = text.trim().match(/^\/idea(?:\s+([\s\S]+))?$/i);
    if (idea) {
      const line = (idea[1] ?? "").trim();
      if (!line) {
        new Notice("Usage: /idea <one line>");
        return;
      }
      try {
        const path = await captureIdea(this.app, line);
        if (this.composer) this.composer.value = "";
        this.draftText = "";
        this.hideSlash();
        new Notice(`Captured ${path}`);
      } catch (e) {
        noticeError(e);
      }
      return;
    }

    if (this.pending) {
      new Notice("Pawn is still answering; stop it first or wait.");
      return;
    }
    this.stickToBottom = true;
    const conversation = this.conversationId;
    const convs = this.plugin.conversations;

    if (text === "/vision refresh") {
      this.armReread();
      if (this.composer) this.composer.value = "";
      this.draftText = "";
      return;
    }

    if (text === "/reset") {
      try {
        await this.plugin.client.chat({ conversation, message: "/reset" }, {});
        convs.clear(conversation);
        new Notice("Conversation reset.");
      } catch (e) {
        new Notice(`Reset failed: ${e instanceof Error ? e.message : e}`);
      }
      this.render();
      return;
    }

    const snap: ContextSnapshot = this.context.snapshot(true);
    const selection: SelectionRef | undefined = snap.selection ?? undefined;
    const notePath = snap.activeNote?.path ?? selection?.path;
    const contextPaths = snap.extra.map((f) => f.path);
    const questionImages = this.pendingImages.splice(0);
    const reread = this.rereadImages;
    this.rereadImages = false;
    const imageNote = questionImages.map((image) => image.filename).join(", ");
    const message = text || "Look at this image.";
    convs.add(conversation, {
      id: newId(),
      role: "user",
      content: imageNote ? (text ? `${text}\n\n(images: ${imageNote})` : `(images: ${imageNote})`) : text,
      createdAt: Date.now(),
      selection,
      notePath,
      context: [...(snap.activeNote ? [snap.activeNote.path] : []), ...contextPaths],
    });

    if (opts.background) {
      this.rereadImages = reread;
      await this.sendBackground(text, conversation, snap);
      return;
    }

    const abort = new AbortController();
    this.pending = { abort, progress: [], el: null };
    this.renderBody();
    let answered = false;
    try {
      const activeNote = snap.activeNote ? await this.noteContext(snap.activeNote) : undefined;
      const context = await Promise.all(snap.extra.map((f) => this.noteContext(f)));
      const model = this.selectedModel();
      const includeNotes = this.plugin.settings.includeNoteImages && this.modelIsVision();
      const noteFiles = [...(snap.activeNote ? [snap.activeNote] : []), ...snap.extra];
      const contextImages = includeNotes ? await this.contextImages(noteFiles) : [];
      const images: ChatImage[] = [
        ...questionImages.map((image) => ({
          filename: image.filename,
          media_type: image.mediaType || mimeForName(image.filename),
          data_base64: bytesToBase64(image.data),
          role: "question" as const,
        })),
        ...contextImages,
      ];
      await this.plugin.client.chat(
        {
          conversation,
          message,
          active_note: activeNote,
          selection: selection?.text,
          context,
          model: model || undefined,
          images: images.length ? images : undefined,
          force_caption: reread || undefined,
          include_note_images: includeNotes,
          ...this.tuningFields(),
        },
        {
          onProgress: (line) => {
            if (!this.pending || this.pending.abort !== abort) return;
            this.pending.progress.push(line);
            this.renderPending();
          },
          onAnswer: (content) => {
            answered = true;
            convs.add(conversation, {
              id: newId(),
              role: "assistant",
              content,
              createdAt: Date.now(),
              progress: this.pending?.progress.slice(),
              selection,
              notePath,
            });
          },
          onJob: (job) => {
            this.plugin.jobs.track(job);
            convs.add(conversation, {
              id: newId(),
              role: "job",
              content: "",
              jobId: job.id,
              createdAt: Date.now(),
            });
          },
          onError: (message) => {
            convs.add(conversation, {
              id: newId(),
              role: "error",
              content: `Pawn error: ${message}`,
              createdAt: Date.now(),
            });
          },
        },
        abort.signal,
      );
      if (abort.signal.aborted && !answered) {
        convs.add(conversation, {
          id: newId(),
          role: "error",
          content: "Stopped. The server may still finish this turn in the background.",
          createdAt: Date.now(),
        });
      }
    } catch (e) {
      const offline = e instanceof ServerUnreachable;
      convs.add(conversation, {
        id: newId(),
        role: "error",
        content: offline
          ? "Pawn server unreachable. Tick Background to queue this as a task note that runs after sync."
          : `Chat failed: ${e instanceof Error ? e.message : e}`,
        createdAt: Date.now(),
      });
    } finally {
      if (this.pending?.abort === abort) this.pending = null;
      if (this.containerEl.isConnected) this.renderBody();
    }
  }

  private async sendBackground(
    text: string,
    conversation: string,
    snap: ContextSnapshot,
  ): Promise<void> {
    try {
      const model = this.selectedModel();
      const job = await this.plugin.jobs.submitAsk({
        instruction: text,
        conversation,
        note_path: snap.activeNote?.path ?? snap.selection?.path,
        selection: snap.selection?.text,
        context_paths: snap.extra.map((f) => f.path),
        model: model || undefined,
        ...this.tuningFields(),
      });
      this.plugin.maybeInsertCallout(job);
      this.addJobMessage(job.id);
    } catch (e) {
      this.plugin.conversations.add(conversation, {
        id: newId(),
        role: "error",
        content: `Could not start job: ${e instanceof Error ? e.message : e}`,
        createdAt: Date.now(),
      });
      this.renderThread();
    }
  }

  private stop(): void {
    this.pending?.abort.abort();
  }
}

/** Image files on a paste. Text pastes are left for the textarea. */
function clipboardImageFiles(data: DataTransfer | null, taken: string[]): File[] {
  if (!data) return [];
  const found: File[] = [];
  for (const item of Array.from(data.items ?? [])) {
    if (item.kind !== "file" || !item.type.toLowerCase().startsWith("image/")) continue;
    const file = item.getAsFile();
    if (file) found.push(file);
  }
  if (!found.length) {
    for (const file of Array.from(data.files ?? [])) {
      if (isImageName(file.name, file.type)) found.push(file);
    }
  }
  const names = new Set(taken);
  return found.map((file) => namePastedImage(file, names));
}

function namePastedImage(file: File, names: Set<string>): File {
  const raw = file.name.trim();
  const generic = !raw || /^image\.(png|jpe?g|gif|webp)$/i.test(raw);
  let name = generic ? `pasted.${extensionForImage(file)}` : raw;
  if (!isImageName(name, file.type)) name = `pasted.${extensionForImage(file)}`;
  if (names.has(name)) {
    const dot = name.lastIndexOf(".");
    const stem = dot > 0 ? name.slice(0, dot) : name;
    const ext = dot > 0 ? name.slice(dot) : "";
    let n = 2;
    while (names.has(`${stem}-${n}${ext}`)) n += 1;
    name = `${stem}-${n}${ext}`;
  }
  names.add(name);
  if (name === file.name) return file;
  return new File([file], name, { type: file.type || mimeForName(name) });
}

function extensionForImage(file: File): string {
  const type = file.type.split(";", 1)[0].trim().toLowerCase();
  if (type === "image/jpeg") return "jpg";
  if (type === "image/gif") return "gif";
  if (type === "image/webp") return "webp";
  return "png";
}

function modelTail(id: string): string {
  const at = id.indexOf("@");
  const model = (at >= 0 ? id.slice(at + 1) : id).trim();
  const slash = model.lastIndexOf("/");
  return (slash >= 0 ? model.slice(slash + 1) : model) || id;
}

function modelProvider(id: string): string {
  const at = id.indexOf("@");
  return at >= 0 ? id.slice(0, at) : "";
}

/** Closed chip: model name only, plus the provider when two models share it. */
function modelChipLabel(id: string, ids: string[]): string {
  const tail = modelTail(id);
  const clash = ids.some((other) => other !== id && modelTail(other) === tail);
  if (!clash) return tail;
  const provider = modelProvider(id);
  return provider ? `${provider} · ${tail}` : id;
}

/** Menu row: model name, then the provider, so the full catalog id stays out of the row. */
function modelMenuLabel(id: string, vision = false): string {
  const provider = modelProvider(id);
  const tail = modelTail(id);
  const base = !provider || provider === tail ? tail : `${tail} · ${provider}`;
  return vision ? `${base} · Vision` : base;
}

const REASONING_LABEL: Record<string, string> = {
  none: "Off",
  low: "Low",
  medium: "Medium",
  high: "High",
};

const ROUTE_LABEL: Record<string, string> = {
  balanced: "Balanced",
  nitro: "Nitro",
  floor: "Floor",
  exacto: "Exacto",
};

function readKeyboardHeight(doc: Document): number {
  for (const el of [doc.documentElement, doc.body]) {
    if (!el) continue;
    const inline = parseFloat(el.style.getPropertyValue("--keyboard-height"));
    if (Number.isFinite(inline) && inline > 0) return inline;
  }
  const computed = parseFloat(
    getComputedStyle(doc.documentElement).getPropertyValue("--keyboard-height"),
  );
  return Number.isFinite(computed) ? Math.max(0, computed) : 0;
}
