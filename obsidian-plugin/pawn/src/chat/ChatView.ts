import {
  Component,
  ItemView,
  MarkdownRenderer,
  Notice,
  TFile,
  WorkspaceLeaf,
  setIcon,
} from "obsidian";
import { resolveActiveMarkdownFile } from "../active";
import { NoteContext, ServerUnreachable } from "../api";
import { PromptCommand, renderPrompt } from "../commands/PromptCommands";
import { JobFilter, renderJobCard, renderJobsList } from "../jobs/JobsView";
import type PawnPlugin from "../main";
import { ContextBar, ContextSnapshot, resolveDroppedNote } from "./ContextBar";
import { renderMessageActions } from "./MessageActions";
import {
  ChatMessage,
  conversationLabel,
  newId,
  noteConversationId,
  SelectionRef,
} from "./conversations";

export const PAWN_CHAT_VIEW = "pawn-chat";

type TabId = "chat" | "jobs";

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
  private fileInput: HTMLInputElement | null = null;

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
    this.containerEl.addClass("pawn-view");
    this.conversationId = this.defaultConversation();
    this.unsubscribeJobs = this.plugin.jobs.onChange(() => this.onJobsChanged());
    this.registerEvent(
      this.app.workspace.on("active-leaf-change", () => this.onActiveNoteChanged()),
    );
    let timer: number | null = null;
    this.registerDomEvent(document, "selectionchange", () => {
      if (timer != null) window.clearTimeout(timer);
      timer = window.setTimeout(() => this.context.refresh(), 300);
    });
    this.render();
  }

  async onClose(): Promise<void> {
    this.unsubscribeJobs?.();
    this.pending?.abort.abort();
    this.containerEl.empty();
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
    this.render();
    if (opts.prefill != null && this.composer) {
      this.composer.value = opts.prefill;
      this.autosize();
    }
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
    if (!id.startsWith("note:")) {
      this.plugin.data.lastGlobalConversation = id;
      this.plugin.persistSoon();
    }
  }

  private onActiveNoteChanged(): void {
    if (this.plugin.settings.conversationMode === "note" && !this.pinned && !this.pending) {
      const file = resolveActiveMarkdownFile(this.app);
      const next = file ? noteConversationId(file.path) : this.conversationId;
      if (next !== this.conversationId) {
        this.conversationId = next;
        this.render();
        return;
      }
    }
    this.context.refresh();
  }

  private onJobsChanged(): void {
    if (this.tab === "jobs") this.renderBody();
    else this.renderThread();
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
    const root = this.contentEl;
    root.empty();
    root.addClass("pawn-root");
    this.renderHeader(root.createDiv({ cls: "pawn-header" }));
    this.bodyEl = root.createDiv({ cls: "pawn-body" });
    this.renderBody();
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
    const dot = tabs.createSpan({
      cls: this.plugin.jobs.online ? "pawn-dot is-online" : "pawn-dot is-offline",
      attr: { "aria-label": this.plugin.jobs.online ? "Server reachable" : "Server offline" },
    });
    dot.setText("●");
  }

  private renderBody(): void {
    const body = this.bodyEl;
    if (!body) return;
    body.empty();
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
    this.renderThread();
    this.renderComposer(body.createDiv({ cls: "pawn-composer" }));
    this.registerDrop(body);
  }

  private renderThread(): void {
    const thread = this.threadEl;
    if (!thread || this.tab !== "chat") return;
    const stick = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 40;
    thread.empty();
    const scope = this.freshScope();
    const conv = this.plugin.conversations.get(this.conversationId);
    if (!conv.messages.length && !this.pending) {
      const empty = thread.createDiv({ cls: "pawn-empty" });
      empty.createEl("p", {
        text: this.conversationId.startsWith("note:")
          ? "Ask Pawn about this note. It's attached as context automatically."
          : "Ask Pawn anything. Add notes with @ or the + chip.",
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
    if (stick || this.pending) thread.scrollTop = thread.scrollHeight;
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
    void MarkdownRenderer.render(this.app, msg.content, body, msg.notePath ?? "", scope);
    renderMessageActions(el, {
      app: this.app,
      text: msg.content,
      selection: msg.selection,
      notePath: msg.notePath,
      saveFolder: `${this.plugin.settings.agentRoot}/Notes`,
    });
  }

  private renderPending(): void {
    const p = this.pending;
    if (!p?.el) return;
    p.el.empty();
    const row = p.el.createDiv({ cls: "pawn-pending-row" });
    row.createSpan({ cls: "pawn-spinner" });
    row.createSpan({ text: p.progress[p.progress.length - 1] ?? "Thinking…" });
  }

  private renderComposer(wrap: HTMLElement): void {
    this.context.mount(wrap);
    this.slashEl = wrap.createDiv({ cls: "pawn-slash is-hidden" });
    const ta = wrap.createEl("textarea", {
      cls: "pawn-input",
      attr: { rows: "2", placeholder: "Message Pawn…  (/ commands, @ notes)" },
    });
    this.composer = ta;
    ta.addEventListener("input", () => this.onComposerInput());
    ta.addEventListener("keydown", (ev) => this.onComposerKey(ev));
    ta.addEventListener("focus", () => this.context.refresh());

    const row = wrap.createDiv({ cls: "pawn-composer-row" });
    const bgLabel = row.createEl("label", { cls: "pawn-bg-toggle" });
    const bg = bgLabel.createEl("input", { type: "checkbox" });
    bg.checked = this.background;
    bg.onchange = () => (this.background = bg.checked);
    bgLabel.appendText(" Background");
    bgLabel.setAttr("aria-label", "Run as a background job; the result shows up here and in Jobs");

    const attach = row.createEl("button", {
      cls: "clickable-icon",
      attr: { "aria-label": "Upload a file to Pawn" },
    });
    setIcon(attach, "paperclip");
    this.fileInput = row.createEl("input", { type: "file", cls: "pawn-hidden" });
    this.fileInput.multiple = true;
    attach.onclick = () => this.fileInput?.click();
    this.fileInput.onchange = () => {
      const files = Array.from(this.fileInput?.files ?? []);
      void this.uploadFiles(files);
      if (this.fileInput) this.fileInput.value = "";
    };

    row.createDiv({ cls: "pawn-spacer" });
    if (this.pending) {
      const stop = row.createEl("button", { text: "Stop", cls: "mod-warning" });
      stop.onclick = () => this.stop();
    } else {
      const send = row.createEl("button", { text: "Send", cls: "mod-cta" });
      send.onclick = () => void this.sendFromComposer();
    }
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
    this.autosize();
    const pos = ta.selectionStart;
    if (pos > 0 && ta.value[pos - 1] === "@" && (pos === 1 || /\s/.test(ta.value[pos - 2]))) {
      ta.value = ta.value.slice(0, pos - 1) + ta.value.slice(pos);
      ta.selectionStart = ta.selectionEnd = pos - 1;
      this.context.openPicker();
      return;
    }
    const m = ta.value.match(/^\/(\S*)$/);
    if (m) this.showSlash(m[1].toLowerCase());
    else this.hideSlash();
  }

  private slashCandidates(): SlashItem[] {
    const builtins: SlashItem[] = [
      { slug: "reset", label: "/reset", hint: "Clear this conversation", run: () => void this.send("/reset") },
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
    if (this.composer) this.composer.value = "";
    this.hideSlash();
    item.run();
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
        void this.uploadFiles(Array.from(dt.files));
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
    if (!text) return;
    ta.value = "";
    this.autosize();
    this.hideSlash();
    await this.send(text, { background: this.background });
  }

  private async noteContext(file: TFile): Promise<NoteContext> {
    if (!this.plugin.settings.sendLocalNoteContent) return { path: file.path };
    return { path: file.path, content: await this.app.vault.cachedRead(file) };
  }

  async send(text: string, opts: { background?: boolean } = {}): Promise<void> {
    if (this.pending) {
      new Notice("Pawn is still answering; stop it first or wait.");
      return;
    }
    const conversation = this.conversationId;
    const convs = this.plugin.conversations;

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
    convs.add(conversation, {
      id: newId(),
      role: "user",
      content: text,
      createdAt: Date.now(),
      selection,
      notePath,
      context: [...(snap.activeNote ? [snap.activeNote.path] : []), ...contextPaths],
    });

    if (opts.background) {
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
      await this.plugin.client.chat(
        {
          conversation,
          message: text,
          active_note: activeNote,
          selection: selection?.text,
          context,
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
      const job = await this.plugin.jobs.submitAsk({
        instruction: text,
        conversation,
        note_path: snap.activeNote?.path ?? snap.selection?.path,
        selection: snap.selection?.text,
        context_paths: snap.extra.map((f) => f.path),
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
