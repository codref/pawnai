import {
  App,
  Editor,
  MarkdownView,
  Notice,
  Plugin,
  PluginSettingTab,
  Setting,
  WorkspaceLeaf,
} from "obsidian";
import { approveTask, createTask, healthCheck } from "./api";
import { PawnPanelView, PAWN_VIEW_TYPE } from "./panel";
import { DEFAULT_SETTINGS, type PawnSettings } from "./settings";
import {
  conversationForNote,
  createTaskNote,
  insertCallout,
  listTasksForNote,
  setApproved,
  uuid4,
  writeTaskResult,
} from "./tasks";

export default class PawnPlugin extends Plugin {
  settings: PawnSettings = DEFAULT_SETTINGS;
  statusEl: HTMLElement | null = null;
  private reachability = false;

  async onload(): Promise<void> {
    await this.loadSettings();

    this.registerView(
      PAWN_VIEW_TYPE,
      (leaf) => new PawnPanelView(leaf, this),
    );

    this.addCommand({
      id: "ask-pawn",
      name: "Ask Pawn",
      editorCallback: async (editor: Editor, view: MarkdownView) => {
        await this.askPawn(editor, view);
      },
    });

    this.addCommand({
      id: "open-pawn-panel",
      name: "Open Pawn panel",
      callback: async () => {
        await this.activatePanel();
      },
    });

    this.addCommand({
      id: "approve-pawn-task",
      name: "Approve Pawn task",
      editorCallback: async (_editor, view) => {
        if (!view.file) return;
        const tasks = await listTasksForNote(
          this.app,
          view.file.path,
          this.settings.agentRoot,
        );
        const review = tasks.find((t) => t.status === "review") || tasks[0];
        if (!review) {
          new Notice("No Pawn task for this note");
          return;
        }
        await setApproved(this.app, review.path, true);
        if (await healthCheck(this.settings)) {
          try {
            await approveTask(this.settings, review.id, review.result);
            new Notice("Approved and indexed");
          } catch {
            new Notice("Approved locally; watcher will index");
          }
        } else {
          new Notice("Approved locally; will sync");
        }
      },
    });

    this.registerEvent(
      this.app.workspace.on("editor-menu", (menu, editor, view) => {
        if (!(view instanceof MarkdownView)) return;
        menu.addItem((item) => {
          item
            .setTitle("Ask Pawn")
            .setIcon("bot")
            .onClick(async () => {
              await this.askPawn(editor, view);
            });
        });
      }),
    );

    this.addRibbonIcon("bot", "Open Pawn panel", async () => {
      await this.activatePanel();
    });

    this.addSettingTab(new PawnSettingTab(this.app, this));

    this.statusEl = this.addStatusBarItem();
    this.statusEl.setText("Pawn …");
    this.registerInterval(
      window.setInterval(() => {
        void this.refreshStatus();
      }, 15000),
    );
    await this.refreshStatus();
  }

  onunload(): void {
    this.app.workspace.detachLeavesOfType(PAWN_VIEW_TYPE);
  }

  async loadSettings(): Promise<void> {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
  }

  async activatePanel(): Promise<void> {
    const { workspace } = this.app;
    let leaf = workspace.getLeavesOfType(PAWN_VIEW_TYPE)[0];
    if (!leaf) {
      const right = workspace.getRightLeaf(false);
      leaf = right ?? workspace.getLeaf(true);
      await leaf.setViewState({ type: PAWN_VIEW_TYPE, active: true });
    }
    workspace.revealLeaf(leaf);
  }

  private async refreshStatus(): Promise<void> {
    this.reachability = await healthCheck(this.settings);
    const file = this.app.workspace.getActiveFile();
    let pending = 0;
    if (file) {
      const tasks = await listTasksForNote(
        this.app,
        file.path,
        this.settings.agentRoot,
      );
      pending = tasks.filter((t) => t.status === "review").length;
    }
    if (this.statusEl) {
      this.statusEl.setText(
        `Pawn ${this.reachability ? "online" : "offline"}${
          pending ? ` · ${pending} review` : ""
        }`,
      );
    }
  }

  private selectionOrParagraph(editor: Editor): string {
    if (editor.somethingSelected()) {
      return editor.getSelection();
    }
    const cursor = editor.getCursor();
    return editor.getLine(cursor.line);
  }

  async askPawn(editor: Editor, view: MarkdownView): Promise<void> {
    if (!view.file) {
      new Notice("Open a Markdown file first");
      return;
    }
    const context = this.selectionOrParagraph(editor).trim();
    const instruction =
      window.prompt(
        "Ask Pawn:",
        context ? `Regarding the selection:\n${context.slice(0, 200)}` : "",
      ) || "";
    if (!instruction.trim()) return;

    const id = uuid4();
    const notePath = view.file.path;
    const conversation = conversationForNote(notePath);
    const taskPath = await createTaskNote(this.app, this.settings, {
      id,
      instruction: instruction.trim(),
      notePath,
      context: context || undefined,
      conversation,
    });
    insertCallout(editor, taskPath, instruction.trim().split("\n")[0]);
    new Notice("Pawn task created");

    if (this.settings.alwaysQueue) {
      new Notice("Queued for vault sync (always-queue on)");
      return;
    }

    const online = await healthCheck(this.settings);
    if (!online) {
      new Notice("Server offline — task will run after sync");
      return;
    }

    new Notice("Asking Pawn…");
    try {
      const { status, data } = await createTask(this.settings, {
        id,
        instruction: instruction.trim(),
        note_path: notePath,
        context: context || undefined,
        conversation,
        timeout_seconds: Math.max(
          5,
          Math.floor(this.settings.fastPathTimeoutMs / 1000),
        ),
      });
      if ((status === 200 || status === 201) && data.result) {
        await writeTaskResult(this.app, taskPath, data.result, "review");
        new Notice("Pawn replied (review in panel)");
      } else if (status === 202) {
        new Notice("Accepted — result will arrive via sync");
      } else {
        new Notice(
          `Pawn: ${data.error_code || status} — left as todo for sync`,
        );
      }
    } catch (err) {
      new Notice(`Fast path failed (${String(err)}); left as todo`);
    }
    await this.refreshStatus();
  }
}

class PawnSettingTab extends PluginSettingTab {
  plugin: PawnPlugin;

  constructor(app: App, plugin: PawnPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h2", { text: "Pawn" });

    new Setting(containerEl)
      .setName("Server URL")
      .setDesc("pawn-server base URL (reachable from this device)")
      .addText((text) =>
        text
          .setPlaceholder("http://127.0.0.1:8000")
          .setValue(this.plugin.settings.serverUrl)
          .onChange(async (value) => {
            this.plugin.settings.serverUrl = value.trim();
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("API token")
      .setDesc("Same as api.token in pawnai.yaml")
      .addText((text) =>
        text
          .setPlaceholder("Bearer token")
          .setValue(this.plugin.settings.apiToken)
          .onChange(async (value) => {
            this.plugin.settings.apiToken = value.trim();
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Agent root")
      .setDesc("Vault folder Pawn owns (default Pawn)")
      .addText((text) =>
        text
          .setValue(this.plugin.settings.agentRoot)
          .onChange(async (value) => {
            this.plugin.settings.agentRoot = value.trim() || "Pawn";
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Fast-path timeout (ms)")
      .setDesc("Wait this long for a direct reply before falling back to sync")
      .addText((text) =>
        text
          .setValue(String(this.plugin.settings.fastPathTimeoutMs))
          .onChange(async (value) => {
            const n = Number(value);
            if (!Number.isNaN(n) && n >= 1000) {
              this.plugin.settings.fastPathTimeoutMs = n;
              await this.plugin.saveSettings();
            }
          }),
      );

    new Setting(containerEl)
      .setName("Always queue")
      .setDesc("Skip HTTP fast path; always leave tasks as todo for the watcher")
      .addToggle((toggle) =>
        toggle
          .setValue(this.plugin.settings.alwaysQueue)
          .onChange(async (value) => {
            this.plugin.settings.alwaysQueue = value;
            await this.plugin.saveSettings();
          }),
      );
  }
}
