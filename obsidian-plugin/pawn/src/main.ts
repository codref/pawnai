import { Editor, MarkdownView, Menu, Notice, Plugin, TAbstractFile, TFile, debounce } from "obsidian";
import { promptText, resolveActiveMarkdownFile } from "./active";
import { Job, PawnClient } from "./api";
import { insertCallout, shortTitle } from "./callout";
import { OpenChatOptions, PAWN_CHAT_VIEW, PawnChatView } from "./chat/ChatView";
import {
  Conversation,
  ConversationStore,
  newId,
  noteConversationId,
} from "./chat/conversations";
import { PromptCommand, PromptCommandRegistry, PromptPickerModal } from "./commands/PromptCommands";
import { applyGoalsFromActiveFile } from "./inbox/ApplyGoals";
import { InboxStore } from "./inbox/InboxView";
import { captureIdea, noticeError } from "./inbox/ideas";
import { QuickCaptureModal } from "./inbox/QuickCapture";
import { JobStore } from "./jobs/JobStore";
import { DEFAULT_SETTINGS, PawnSettings, PawnSettingTab } from "./settings";
import { VaultSync } from "./vaultSync";

const LEGACY_PANEL_VIEW = "pawn-panel";
const TEXT_UPLOAD = /\.(md|markdown|txt)$/i;

interface PawnData {
  settings: PawnSettings;
  conversations: Record<string, Conversation>;
  lastGlobalConversation: string;
}

export default class PawnPlugin extends Plugin {
  settings: PawnSettings = { ...DEFAULT_SETTINGS };
  data: PawnData = { settings: this.settings, conversations: {}, lastGlobalConversation: "" };
  client!: PawnClient;
  jobs!: JobStore;
  inbox!: InboxStore;
  vaultSync!: VaultSync;
  prompts!: PromptCommandRegistry;
  conversations!: ConversationStore;
  private statusEl: HTMLElement | null = null;

  persistSoon = debounce(() => void this.persist(), 1000, true);

  async onload(): Promise<void> {
    await this.loadPluginData();
    this.client = new PawnClient(() => this.settings);
    this.conversations = new ConversationStore(this.data.conversations, () => this.persistSoon());
    this.jobs = new JobStore(this, this.client);
    this.inbox = new InboxStore(this);
    this.vaultSync = new VaultSync(this, this.client);
    this.prompts = new PromptCommandRegistry(this);

    this.addSettingTab(new PawnSettingTab(this.app, this));
    this.registerView(PAWN_CHAT_VIEW, (leaf) => new PawnChatView(leaf, this));
    this.registerCommands();
    this.registerMenus();

    this.addRibbonIcon("chess-king", "Open Pawn chat", () => void this.openChat({}));
    this.statusEl = this.addStatusBarItem();
    this.statusEl.addClass("mod-clickable");
    this.statusEl.onclick = () => void this.openChat({ tab: "jobs" });

    this.app.workspace.onLayoutReady(() => {
      this.app.workspace.detachLeavesOfType(LEGACY_PANEL_VIEW);
      void this.prompts.reload();
      this.jobs.start();
      this.inbox.start();
      this.vaultSync.start();
      this.updateStatusBar();
    });

    const reloadPrompts = debounce(() => void this.prompts.reload(), 500, true);
    const onVaultChange = (file: TAbstractFile) => {
      if (this.prompts.isCommandFile(file.path)) reloadPrompts();
    };
    this.registerEvent(this.app.metadataCache.on("changed", onVaultChange));
    this.registerEvent(this.app.vault.on("delete", onVaultChange));
    this.registerEvent(
      this.app.vault.on("rename", (file, oldPath) => {
        if (this.prompts.isCommandFile(file.path) || this.prompts.isCommandFile(oldPath)) {
          reloadPrompts();
        }
      }),
    );
  }

  onunload(): void {
    this.jobs?.stop();
    this.inbox?.stop();
    this.vaultSync?.stop();
    void this.persist();
  }

  // ── persistence ──────────────────────────────────────────────────────────

  private async loadPluginData(): Promise<void> {
    const raw = ((await this.loadData()) ?? {}) as Partial<PawnData> & Partial<PawnSettings>;
    // 0.1 stored settings flat at the top level.
    const storedSettings = raw.settings ?? (raw as Partial<PawnSettings>);
    const settings: PawnSettings = { ...DEFAULT_SETTINGS };
    for (const key of Object.keys(DEFAULT_SETTINGS) as Array<keyof PawnSettings>) {
      if (storedSettings[key] !== undefined) {
        (settings as unknown as Record<string, unknown>)[key] = storedSettings[key];
      }
    }
    this.settings = settings;
    this.data = {
      settings,
      conversations: raw.conversations ?? {},
      lastGlobalConversation: raw.lastGlobalConversation ?? "",
    };
  }

  private async persist(): Promise<void> {
    this.data.settings = this.settings;
    this.data.conversations = this.conversations?.toJSON() ?? this.data.conversations;
    await this.saveData(this.data);
  }

  async saveSettings(): Promise<void> {
    await this.persist();
    this.updateStatusBar();
  }

  // ── commands and menus ───────────────────────────────────────────────────

  private registerCommands(): void {
    this.addCommand({
      id: "open-chat",
      name: "Open chat",
      callback: () => void this.openChat({ tab: "chat" }),
    });
    this.addCommand({
      id: "new-chat",
      name: "New chat",
      callback: async () => {
        const view = await this.openChat({ tab: "chat" });
        view?.applyOptions({ conversation: `chat:${newId()}` });
      },
    });
    this.addCommand({
      id: "ask-pawn",
      name: "Ask about selection or note",
      editorCallback: () => void this.openChat({ tab: "chat" }),
    });
    this.addCommand({
      id: "send-background",
      name: "Send to Pawn (background)",
      editorCallback: (editor, ctx) => {
        const file = ctx.file;
        if (file) void this.sendBackgroundFromEditor(editor, file);
      },
    });
    this.addCommand({
      id: "run-prompt",
      name: "Run prompt command…",
      callback: () =>
        new PromptPickerModal(this, (cmd) => void this.runPromptCommand(cmd)).open(),
    });
    this.addCommand({
      id: "show-jobs",
      name: "Show background jobs",
      callback: () => void this.openChat({ tab: "jobs" }),
    });
    this.addCommand({
      id: "show-inbox",
      name: "Show inbox",
      callback: () => void this.openChat({ tab: "inbox" }),
    });
    this.addCommand({
      id: "quick-capture",
      name: "Quick capture",
      callback: () => {
        new QuickCaptureModal(this.app, async (title) => {
          try {
            const path = await captureIdea(this.app, title);
            new Notice(`Captured ${path}`);
          } catch (e) {
            noticeError(e);
          }
        }).open();
      },
    });
    this.addCommand({
      id: "apply-goals",
      name: "Apply goals proposal",
      callback: () => void applyGoalsFromActiveFile(this.app),
    });
    this.addCommand({
      id: "add-note-context",
      name: "Add current note to chat context",
      checkCallback: (checking) => {
        const file = resolveActiveMarkdownFile(this.app);
        if (!file) return false;
        if (!checking) void this.addToContext(file);
        return true;
      },
    });
    this.addCommand({
      id: "upload-active-file",
      name: "Upload active file to Pawn",
      checkCallback: (checking) => {
        const file = this.app.workspace.getActiveFile();
        if (!file) return false;
        if (!checking) void this.uploadVaultFile(file);
        return true;
      },
    });
    this.addCommand({
      id: "create-default-prompts",
      name: "Create default prompt commands",
      callback: () => void this.prompts.writeDefaults(),
    });
  }

  private registerMenus(): void {
    this.registerEvent(
      this.app.workspace.on("editor-menu", (menu: Menu, editor: Editor, view) => {
        if (!(view instanceof MarkdownView)) return;
        menu.addSeparator();
        menu.addItem((i) =>
          i
            .setTitle("Ask Pawn")
            .setIcon("chess-king")
            .onClick(() => void this.openChat({ tab: "chat" })),
        );
        menu.addItem((i) =>
          i
            .setTitle("Send to Pawn (background)")
            .setIcon("clock")
            .onClick(() => {
              if (view.file) void this.sendBackgroundFromEditor(editor, view.file);
            }),
        );
        for (const cmd of this.prompts.list().filter((c) => c.contextMenu)) {
          menu.addItem((i) =>
            i
              .setTitle(`Pawn: ${cmd.name}`)
              .setIcon("sparkles")
              .onClick(() => void this.runPromptCommand(cmd)),
          );
        }
      }),
    );

    this.registerEvent(
      this.app.workspace.on("file-menu", (menu: Menu, file: TAbstractFile) => {
        if (!(file instanceof TFile)) return;
        if (file.extension === "md") {
          menu.addItem((i) =>
            i
              .setTitle("Add to Pawn chat context")
              .setIcon("chess-king")
              .onClick(() => void this.addToContext(file)),
          );
        } else {
          menu.addItem((i) =>
            i
              .setTitle("Upload to Pawn")
              .setIcon("upload")
              .onClick(() => void this.uploadVaultFile(file)),
          );
        }
      }),
    );
  }

  // ── views ────────────────────────────────────────────────────────────────

  async openChat(opts: OpenChatOptions): Promise<PawnChatView | null> {
    const { workspace } = this.app;
    let leaf = workspace.getLeavesOfType(PAWN_CHAT_VIEW)[0];
    if (!leaf) {
      const right = workspace.getRightLeaf(false);
      if (!right) return null;
      await right.setViewState({ type: PAWN_CHAT_VIEW, active: true });
      leaf = right;
    }
    await workspace.revealLeaf(leaf);
    await leaf.loadIfDeferred?.();
    const view = leaf.view;
    if (!(view instanceof PawnChatView)) {
      console.warn("Pawn: chat leaf has unexpected view", view?.getViewType?.());
      return null;
    }
    view.applyOptions(opts);
    return view;
  }

  async runPromptCommand(cmd: PromptCommand): Promise<void> {
    const view = await this.openChat({ tab: "chat" });
    await view?.runPrompt(cmd);
  }

  private async addToContext(file: TFile): Promise<void> {
    const view = await this.openChat({ tab: "chat" });
    view?.addContextFile(file);
  }

  // ── background jobs ──────────────────────────────────────────────────────

  private async sendBackgroundFromEditor(editor: Editor, file: TFile): Promise<void> {
    const selection = editor.getSelection();
    const instruction = await promptText(this.app, {
      title: "Background job for Pawn",
      placeholder: selection
        ? "What should Pawn do with the selected text?"
        : "What should Pawn do with this note?",
    });
    if (!instruction?.trim()) return;
    try {
      const job = await this.jobs.submitAsk({
        instruction: instruction.trim(),
        conversation: noteConversationId(file.path),
        note_path: file.path,
        selection: selection || undefined,
      });
      this.maybeInsertCallout(job, editor);
      new Notice(
        job.offline ? "Saved as task note (offline)." : "Pawn is working on it in the background.",
      );
    } catch (e) {
      new Notice(`Could not start job: ${e instanceof Error ? e.message : e}`);
    }
  }

  maybeInsertCallout(job: Job, editor?: Editor): void {
    if (!this.settings.insertCalloutForJobs || !job.task_key) return;
    const ed = editor ?? this.app.workspace.getActiveViewOfType(MarkdownView)?.editor;
    if (ed) insertCallout(ed, job.task_key, shortTitle(job.instruction));
  }

  async uploadVaultFile(file: TFile, conversation?: string): Promise<Job | null> {
    const data = await this.app.vault.readBinary(file);
    return this.uploadData(file.name, data, "", conversation, file.path);
  }

  async uploadData(
    filename: string,
    data: ArrayBuffer,
    contentType: string,
    conversation?: string,
    notePath?: string,
  ): Promise<Job | null> {
    try {
      const job = await this.client.upload({
        filename,
        data,
        contentType: contentType || undefined,
        conversation,
        notePath,
        index: TEXT_UPLOAD.test(filename),
      });
      this.jobs.track(job);
      new Notice(`Uploading ${filename} to Pawn…`);
      return job;
    } catch (e) {
      new Notice(`Upload failed: ${e instanceof Error ? e.message : e}`);
      return null;
    }
  }

  updateStatusBar(): void {
    const el = this.statusEl;
    if (!el || !this.jobs) return;
    el.empty();
    const online = this.jobs.online;
    el.createSpan({
      cls: online ? "pawn-dot is-online" : "pawn-dot is-offline",
      text: "● ",
    });
    const { active, review } = this.jobs.counts();
    const waiting = this.inbox?.attention() ?? 0;
    const parts = ["Pawn"];
    if (active) parts.push(`${active} running`);
    if (review) parts.push(`${review} to review`);
    if (waiting) parts.push(`${waiting} inbox`);
    el.createSpan({ text: parts.join(" · ") });
    el.setAttr("aria-label", online ? "Pawn server reachable" : "Pawn server offline");
  }
}
