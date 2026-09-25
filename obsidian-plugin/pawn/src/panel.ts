import {
  ItemView,
  MarkdownView,
  Notice,
  WorkspaceLeaf,
} from "obsidian";
import type PawnPlugin from "./main";
import {
  noticeIfNoNote,
  promptText,
  resolveActiveMarkdownFile,
  resolveActiveMarkdownView,
} from "./active";
import { approveTask, chatCompletions, healthCheck } from "./api";
import {
  appendInstructionFollowUp,
  listTasksForNote,
  setApproved,
  setTaskStatus,
  type TaskMeta,
} from "./tasks";

export const PAWN_VIEW_TYPE = "pawn-panel";

export class PawnPanelView extends ItemView {
  plugin: PawnPlugin;
  private tab: "tasks" | "chat" = "tasks";
  private chatLog: { role: "user" | "assistant"; content: string }[] = [];
  private busy = false;

  constructor(leaf: WorkspaceLeaf, plugin: PawnPlugin) {
    super(leaf);
    this.plugin = plugin;
  }

  getViewType(): string {
    return PAWN_VIEW_TYPE;
  }

  getDisplayText(): string {
    return "Pawn";
  }

  getIcon(): string {
    return "bot";
  }

  async onOpen(): Promise<void> {
    this.registerEvent(
      this.app.workspace.on("file-open", () => {
        void this.render();
      }),
    );
    this.registerEvent(
      this.app.workspace.on("active-leaf-change", () => {
        void this.render();
      }),
    );
    await this.render();
  }

  async onClose(): Promise<void> {
    this.containerEl.empty();
  }

  async refresh(): Promise<void> {
    await this.render();
  }

  private activeNotePath(): string | null {
    return resolveActiveMarkdownFile(this.app)?.path ?? null;
  }

  private markdownEditor(): MarkdownView | null {
    return resolveActiveMarkdownView(this.app);
  }

  private async render(): Promise<void> {
    const root = this.containerEl.children[1] as HTMLElement;
    root.empty();
    root.addClass("pawn-panel");

    const tabs = root.createDiv({ cls: "pawn-tabs" });
    const tasksTab = tabs.createEl("button", {
      text: "Tasks",
      cls: this.tab === "tasks" ? "is-active" : "",
    });
    const chatTab = tabs.createEl("button", {
      text: "Chat",
      cls: this.tab === "chat" ? "is-active" : "",
    });
    tasksTab.onclick = async () => {
      this.tab = "tasks";
      await this.render();
    };
    chatTab.onclick = async () => {
      this.tab = "chat";
      await this.render();
    };

    if (this.tab === "tasks") {
      await this.renderTasks(root);
    } else {
      await this.renderChat(root);
    }
  }

  private async renderTasks(root: HTMLElement): Promise<void> {
    const notePath = this.activeNotePath();
    const body = root.createDiv({ cls: "pawn-panel-body" });
    if (!notePath) {
      body.createEl("p", { text: "Open a Markdown note to see related tasks." });
      return;
    }
    body.createEl("p", {
      cls: "pawn-muted",
      text: `Tasks for ${notePath}`,
    });
    const tasks = await listTasksForNote(
      this.app,
      notePath,
      this.plugin.settings.agentRoot,
    );
    if (!tasks.length) {
      body.createEl("p", { text: "No Pawn tasks linked to this note yet." });
      return;
    }
    for (const task of tasks) {
      const card = body.createDiv({ cls: "pawn-task-card" });
      card.createEl("div", {
        cls: "pawn-task-title",
        text: `${task.status} · ${task.id.slice(0, 8)}`,
      });
      if (task.result) {
        card.createEl("pre", {
          cls: "pawn-task-result",
          text: task.result.slice(0, 800),
        });
      } else {
        card.createEl("p", {
          cls: "pawn-muted",
          text: task.instruction.slice(0, 200) || "(no instruction)",
        });
      }
      const actions = card.createDiv({ cls: "pawn-task-actions" });
      this.actionButton(actions, "Insert", () => this.insertResult(task));
      this.actionButton(actions, "Replace", () => this.replaceSelection(task));
      this.actionButton(actions, "Reply", () => this.replyToTask(task));
      this.actionButton(actions, "Approve", () => this.approve(task));
      this.actionButton(actions, "Open", () => {
        this.app.workspace.openLinkText(task.path, "", false);
      });
    }
  }

  private actionButton(
    parent: HTMLElement,
    label: string,
    onClick: () => void | Promise<void>,
  ): void {
    const btn = parent.createEl("button", { text: label });
    btn.onclick = async () => {
      try {
        await onClick();
      } catch (err) {
        new Notice(`Pawn: ${String(err)}`);
      }
    };
  }

  private insertResult(task: TaskMeta): void {
    const view = this.markdownEditor();
    if (!view?.editor || !task.result) {
      new Notice("No result to insert");
      return;
    }
    const editor = view.editor;
    const cursor = editor.getCursor();
    editor.replaceRange("\n" + task.result.trim() + "\n", cursor);
  }

  private replaceSelection(task: TaskMeta): void {
    const view = this.markdownEditor();
    if (!view?.editor || !task.result) {
      new Notice("No result to insert");
      return;
    }
    const editor = view.editor;
    if (editor.somethingSelected()) {
      editor.replaceSelection(task.result.trim());
    } else {
      this.insertResult(task);
    }
  }

  private async replyToTask(task: TaskMeta): Promise<void> {
    const followUp = await promptText(this.app, {
      title: "Follow-up for Pawn",
      placeholder: "Add a follow-up instruction…",
    });
    if (!followUp?.trim()) return;
    await appendInstructionFollowUp(this.app, task.path, followUp.trim());
    await setTaskStatus(this.app, task.path, "todo");
    new Notice("Follow-up queued (status: todo)");
    await this.render();
  }

  private async approve(task: TaskMeta): Promise<void> {
    await setApproved(this.app, task.path, true);
    const online = await healthCheck(this.plugin.settings);
    if (online) {
      try {
        await approveTask(this.plugin.settings, task.id, task.result);
        new Notice("Approved and indexed into Pawn memory");
      } catch (err) {
        new Notice(
          `Approved locally; server index failed: ${String(err)}. Watcher will retry.`,
        );
      }
    } else {
      new Notice("Approved locally; will sync for watcher indexing");
    }
    await this.render();
  }

  private async renderChat(root: HTMLElement): Promise<void> {
    const notePath = this.activeNotePath();
    const body = root.createDiv({ cls: "pawn-panel-body" });
    if (!notePath) {
      body.createEl("p", { text: "Open a note to chat about it." });
      return;
    }
    const online = await healthCheck(this.plugin.settings);
    if (!online) {
      body.createEl("p", {
        text: "Server unreachable. Chat needs pawn-server.",
      });
      return;
    }
    body.createEl("p", {
      cls: "pawn-muted",
      text: `Chat · note:${notePath}`,
    });
    const log = body.createDiv({ cls: "pawn-chat-log" });
    for (const msg of this.chatLog) {
      log.createEl("div", {
        cls: `pawn-chat-msg pawn-chat-${msg.role}`,
        text: msg.content,
      });
    }
    const form = body.createDiv({ cls: "pawn-chat-form" });
    const input = form.createEl("textarea", {
      attr: { rows: "3", placeholder: "Ask Pawn about this note…" },
    });
    const row = form.createDiv({ cls: "pawn-task-actions" });
    const send = row.createEl("button", { text: this.busy ? "…" : "Send" });
    send.disabled = this.busy;
    send.onclick = async () => {
      const text = input.value.trim();
      if (!text || this.busy) return;
      this.busy = true;
      this.chatLog.push({ role: "user", content: text });
      input.value = "";
      await this.render();
      try {
        const reply = await chatCompletions(this.plugin.settings, {
          messages: this.chatLog.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          user: `note:${notePath}`,
        });
        this.chatLog.push({ role: "assistant", content: reply });
      } catch (err) {
        new Notice(`Chat failed: ${String(err)}`);
      } finally {
        this.busy = false;
        await this.render();
      }
    };
    if (this.chatLog.length) {
      const last = this.chatLog[this.chatLog.length - 1];
      if (last.role === "assistant") {
        const insert = row.createEl("button", { text: "Insert into note" });
        insert.onclick = () => {
          const view = this.markdownEditor();
          if (!view?.editor) {
            noticeIfNoNote();
            return;
          }
          view.editor.replaceRange(
            "\n" + last.content.trim() + "\n",
            view.editor.getCursor(),
          );
        };
      }
    }
  }
}
