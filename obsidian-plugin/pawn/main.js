/* Pawn Obsidian plugin */
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/main.ts
var main_exports = {};
__export(main_exports, {
  default: () => PawnPlugin
});
module.exports = __toCommonJS(main_exports);
var import_obsidian3 = require("obsidian");

// src/api.ts
var import_obsidian = require("obsidian");
function authHeaders(settings) {
  const headers = {
    "Content-Type": "application/json"
  };
  if (settings.apiToken) {
    headers.Authorization = `Bearer ${settings.apiToken}`;
  }
  return headers;
}
function baseUrl(settings) {
  return settings.serverUrl.replace(/\/+$/, "");
}
async function healthCheck(settings) {
  try {
    const res = await (0, import_obsidian.requestUrl)({
      url: `${baseUrl(settings)}/health`,
      method: "GET",
      throw: false
    });
    return res.status >= 200 && res.status < 300;
  } catch {
    return false;
  }
}
async function createTask(settings, body) {
  const res = await (0, import_obsidian.requestUrl)({
    url: `${baseUrl(settings)}/v1/vault/tasks`,
    method: "POST",
    headers: authHeaders(settings),
    body: JSON.stringify(body),
    throw: false
  });
  let data;
  try {
    data = res.json;
  } catch {
    data = { task_id: body.id || "", status: "blocked", error_code: "bad_response" };
  }
  return { status: res.status, data };
}
async function approveTask(settings, id, result) {
  const res = await (0, import_obsidian.requestUrl)({
    url: `${baseUrl(settings)}/v1/vault/tasks/${encodeURIComponent(id)}/approve`,
    method: "POST",
    headers: authHeaders(settings),
    body: JSON.stringify(result ? { result } : {}),
    throw: false
  });
  return res.json;
}
async function chatCompletions(settings, opts) {
  const res = await (0, import_obsidian.requestUrl)({
    url: `${baseUrl(settings)}/v1/chat/completions`,
    method: "POST",
    headers: authHeaders(settings),
    body: JSON.stringify({
      model: "pawn",
      messages: opts.messages,
      user: opts.user,
      stream: false
    })
  });
  const data = res.json;
  return data.choices?.[0]?.message?.content ?? "";
}

// src/panel.ts
var import_obsidian2 = require("obsidian");

// src/tasks.ts
function yamlEscape(value) {
  if (/[:#{}[\],&*?|>!%@`]/.test(value) || value.includes("\n")) {
    return JSON.stringify(value);
  }
  return value;
}
function taskPathForId(agentRoot, id) {
  return `${agentRoot.replace(/\/+$/, "")}/Tasks/${id}.md`;
}
function conversationForNote(notePath) {
  return `note:${notePath.replace(/^\/+/, "")}`;
}
function renderTaskNote(opts) {
  const lines = [
    "---",
    "pawn: task",
    `id: ${opts.id}`,
    `status: ${opts.status}`,
    `approved: ${opts.approved ? "true" : "false"}`
  ];
  if (opts.conversation) {
    lines.push(`conversation: ${yamlEscape(opts.conversation)}`);
  }
  if (opts.notePath) {
    const wiki = opts.notePath.replace(/\.md$/i, "");
    lines.push(`note: "[[${wiki}]]"`);
  }
  lines.push("---", "", "## Instruction", opts.instruction.trim(), "");
  if (opts.context?.trim()) {
    lines.push("## Context", opts.context.trim(), "");
  }
  if (opts.result?.trim()) {
    lines.push("## Result", opts.result.trim(), "");
  }
  return lines.join("\n");
}
function getSection(body, name) {
  const re = new RegExp(
    `^##\\s+${name}\\s*\\n([\\s\\S]*?)(?=^##\\s+|$)`,
    "m"
  );
  const m = body.match(re);
  return m ? m[1].trim() : "";
}
function parseTaskFile(path, text) {
  if (!text.startsWith("---")) return null;
  const end = text.indexOf("\n---", 3);
  if (end < 0) return null;
  const fm = text.slice(3, end).trim();
  const body = text.slice(end + 4);
  if (!/^pawn:\s*task\b/m.test(fm)) return null;
  const id = (fm.match(/^id:\s*(.+)$/m)?.[1] || "").trim();
  const status = (fm.match(/^status:\s*(.+)$/m)?.[1] || "todo").trim();
  const approved = /^approved:\s*(true|yes|1)\b/im.test(fm);
  const conversation = (fm.match(/^conversation:\s*(.+)$/m)?.[1] || "").trim().replace(/^["']|["']$/g, "");
  let note = (fm.match(/^note:\s*(.+)$/m)?.[1] || "").trim();
  note = note.replace(/^["']|["']$/g, "");
  const wiki = note.match(/^\[\[([^\]|]+)/);
  if (wiki) note = wiki[1].trim();
  return {
    id,
    status,
    approved,
    note,
    conversation,
    path,
    instruction: getSection(body, "Instruction"),
    result: getSection(body, "Result")
  };
}
async function createTaskNote(app, settings, opts) {
  const path = taskPathForId(settings.agentRoot, opts.id);
  const folder = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "";
  if (folder && !app.vault.getAbstractFileByPath(folder)) {
    await app.vault.createFolder(folder).catch(() => void 0);
  }
  const content = renderTaskNote({
    id: opts.id,
    status: "todo",
    instruction: opts.instruction,
    context: opts.context,
    notePath: opts.notePath,
    conversation: opts.conversation || conversationForNote(opts.notePath),
    approved: false
  });
  const existing = app.vault.getAbstractFileByPath(path);
  if (existing) {
    await app.vault.modify(existing, content);
  } else {
    await app.vault.create(path, content);
  }
  return path;
}
async function writeTaskResult(app, taskPath, result, status) {
  const file = app.vault.getAbstractFileByPath(taskPath);
  if (!file || !("extension" in file)) return;
  const text = await app.vault.read(file);
  const meta = parseTaskFile(taskPath, text);
  if (!meta) return;
  const content = renderTaskNote({
    id: meta.id,
    status,
    instruction: meta.instruction,
    context: getSection(
      text.includes("---") ? text.slice(text.indexOf("\n---", 3) + 4) : text,
      "Context"
    ),
    result,
    notePath: meta.note,
    conversation: meta.conversation,
    approved: false
  });
  await app.vault.modify(file, content);
}
async function setApproved(app, taskPath, approved) {
  const file = app.vault.getAbstractFileByPath(taskPath);
  if (!file || !("extension" in file)) return;
  await app.fileManager.processFrontMatter(file, (fm) => {
    fm.approved = approved;
    if (approved) fm.status = "done";
  });
}
async function setTaskStatus(app, taskPath, status) {
  const file = app.vault.getAbstractFileByPath(taskPath);
  if (!file || !("extension" in file)) return;
  await app.fileManager.processFrontMatter(file, (fm) => {
    fm.status = status;
  });
}
async function appendInstructionFollowUp(app, taskPath, followUp) {
  const file = app.vault.getAbstractFileByPath(taskPath);
  if (!file || !("extension" in file)) return;
  const text = await app.vault.read(file);
  const meta = parseTaskFile(taskPath, text);
  if (!meta) return;
  const bodyStart = text.indexOf("\n---", 3) + 4;
  const body = text.slice(bodyStart);
  const context = getSection(body, "Context");
  const instruction = meta.instruction.trim() + "\n\n### Follow-up\n" + followUp.trim();
  const content = renderTaskNote({
    id: meta.id,
    status: "todo",
    instruction,
    context,
    result: meta.result,
    notePath: meta.note,
    conversation: meta.conversation,
    approved: false
  });
  await app.vault.modify(file, content);
}
async function listTasksForNote(app, notePath, agentRoot) {
  const folder = `${agentRoot.replace(/\/+$/, "")}/Tasks`;
  const files = app.vault.getMarkdownFiles().filter(
    (f) => f.path.startsWith(folder + "/")
  );
  const noteKey = notePath.replace(/\.md$/i, "");
  const out = [];
  for (const file of files) {
    const text = await app.vault.read(file);
    const meta = parseTaskFile(file.path, text);
    if (!meta) continue;
    const linked = meta.note.replace(/\.md$/i, "");
    if (linked === noteKey || meta.conversation === conversationForNote(notePath) || meta.conversation === `note:${notePath}`) {
      out.push(meta);
    }
  }
  return out;
}
function insertCallout(editor, taskPath, title) {
  const wiki = taskPath.replace(/\.md$/i, "");
  const short = (title || "Pawn task").replace(/\n/g, " ").slice(0, 60);
  const callout = `> [!pawn] [[${wiki}|${short}]]
`;
  const cursor = editor.getCursor();
  editor.replaceRange(callout, cursor);
}
function uuid4() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0;
    const v = c === "x" ? r : r & 3 | 8;
    return v.toString(16);
  });
}

// src/panel.ts
var PAWN_VIEW_TYPE = "pawn-panel";
var PawnPanelView = class extends import_obsidian2.ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.tab = "tasks";
    this.chatLog = [];
    this.busy = false;
    this.plugin = plugin;
  }
  getViewType() {
    return PAWN_VIEW_TYPE;
  }
  getDisplayText() {
    return "Pawn";
  }
  getIcon() {
    return "bot";
  }
  async onOpen() {
    await this.render();
  }
  async onClose() {
    this.containerEl.empty();
  }
  async refresh() {
    await this.render();
  }
  activeNotePath() {
    const view = this.app.workspace.getActiveViewOfType(import_obsidian2.MarkdownView);
    return view?.file?.path ?? null;
  }
  async render() {
    const root = this.containerEl.children[1];
    root.empty();
    root.addClass("pawn-panel");
    const tabs = root.createDiv({ cls: "pawn-tabs" });
    const tasksTab = tabs.createEl("button", {
      text: "Tasks",
      cls: this.tab === "tasks" ? "is-active" : ""
    });
    const chatTab = tabs.createEl("button", {
      text: "Chat",
      cls: this.tab === "chat" ? "is-active" : ""
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
  async renderTasks(root) {
    const notePath = this.activeNotePath();
    const body = root.createDiv({ cls: "pawn-panel-body" });
    if (!notePath) {
      body.createEl("p", { text: "Open a Markdown note to see related tasks." });
      return;
    }
    body.createEl("p", {
      cls: "pawn-muted",
      text: `Tasks for ${notePath}`
    });
    const tasks = await listTasksForNote(
      this.app,
      notePath,
      this.plugin.settings.agentRoot
    );
    if (!tasks.length) {
      body.createEl("p", { text: "No Pawn tasks linked to this note yet." });
      return;
    }
    for (const task of tasks) {
      const card = body.createDiv({ cls: "pawn-task-card" });
      card.createEl("div", {
        cls: "pawn-task-title",
        text: `${task.status} \xB7 ${task.id.slice(0, 8)}`
      });
      if (task.result) {
        card.createEl("pre", {
          cls: "pawn-task-result",
          text: task.result.slice(0, 800)
        });
      } else {
        card.createEl("p", {
          cls: "pawn-muted",
          text: task.instruction.slice(0, 200) || "(no instruction)"
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
  actionButton(parent, label, onClick) {
    const btn = parent.createEl("button", { text: label });
    btn.onclick = async () => {
      try {
        await onClick();
      } catch (err) {
        new import_obsidian2.Notice(`Pawn: ${String(err)}`);
      }
    };
  }
  insertResult(task) {
    const view = this.app.workspace.getActiveViewOfType(import_obsidian2.MarkdownView);
    if (!view?.editor || !task.result) {
      new import_obsidian2.Notice("No result to insert");
      return;
    }
    const editor = view.editor;
    const cursor = editor.getCursor();
    editor.replaceRange("\n" + task.result.trim() + "\n", cursor);
  }
  replaceSelection(task) {
    const view = this.app.workspace.getActiveViewOfType(import_obsidian2.MarkdownView);
    if (!view?.editor || !task.result) {
      new import_obsidian2.Notice("No result to insert");
      return;
    }
    const editor = view.editor;
    if (editor.somethingSelected()) {
      editor.replaceSelection(task.result.trim());
    } else {
      this.insertResult(task);
    }
  }
  async replyToTask(task) {
    const followUp = window.prompt("Follow-up for Pawn:");
    if (!followUp?.trim()) return;
    await appendInstructionFollowUp(this.app, task.path, followUp.trim());
    await setTaskStatus(this.app, task.path, "todo");
    new import_obsidian2.Notice("Follow-up queued (status: todo)");
    await this.render();
  }
  async approve(task) {
    await setApproved(this.app, task.path, true);
    const online = await healthCheck(this.plugin.settings);
    if (online) {
      try {
        await approveTask(this.plugin.settings, task.id, task.result);
        new import_obsidian2.Notice("Approved and indexed into Pawn memory");
      } catch (err) {
        new import_obsidian2.Notice(
          `Approved locally; server index failed: ${String(err)}. Watcher will retry.`
        );
      }
    } else {
      new import_obsidian2.Notice("Approved locally; will sync for watcher indexing");
    }
    await this.render();
  }
  async renderChat(root) {
    const notePath = this.activeNotePath();
    const body = root.createDiv({ cls: "pawn-panel-body" });
    if (!notePath) {
      body.createEl("p", { text: "Open a note to chat about it." });
      return;
    }
    const online = await healthCheck(this.plugin.settings);
    if (!online) {
      body.createEl("p", {
        text: "Server unreachable. Chat needs pawn-server."
      });
      return;
    }
    body.createEl("p", {
      cls: "pawn-muted",
      text: `Chat \xB7 note:${notePath}`
    });
    const log = body.createDiv({ cls: "pawn-chat-log" });
    for (const msg of this.chatLog) {
      log.createEl("div", {
        cls: `pawn-chat-msg pawn-chat-${msg.role}`,
        text: msg.content
      });
    }
    const form = body.createDiv({ cls: "pawn-chat-form" });
    const input = form.createEl("textarea", {
      attr: { rows: "3", placeholder: "Ask Pawn about this note\u2026" }
    });
    const row = form.createDiv({ cls: "pawn-task-actions" });
    const send = row.createEl("button", { text: this.busy ? "\u2026" : "Send" });
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
            content: m.content
          })),
          user: `note:${notePath}`
        });
        this.chatLog.push({ role: "assistant", content: reply });
      } catch (err) {
        new import_obsidian2.Notice(`Chat failed: ${String(err)}`);
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
          const view = this.app.workspace.getActiveViewOfType(import_obsidian2.MarkdownView);
          if (!view?.editor) return;
          view.editor.replaceRange(
            "\n" + last.content.trim() + "\n",
            view.editor.getCursor()
          );
        };
      }
    }
  }
};

// src/settings.ts
var DEFAULT_SETTINGS = {
  serverUrl: "http://127.0.0.1:8000",
  apiToken: "",
  agentRoot: "Pawn",
  fastPathTimeoutMs: 6e4,
  alwaysQueue: false
};

// src/main.ts
var PawnPlugin = class extends import_obsidian3.Plugin {
  constructor() {
    super(...arguments);
    this.settings = DEFAULT_SETTINGS;
    this.statusEl = null;
    this.reachability = false;
  }
  async onload() {
    await this.loadSettings();
    this.registerView(
      PAWN_VIEW_TYPE,
      (leaf) => new PawnPanelView(leaf, this)
    );
    this.addCommand({
      id: "ask-pawn",
      name: "Ask Pawn",
      editorCallback: async (editor, view) => {
        await this.askPawn(editor, view);
      }
    });
    this.addCommand({
      id: "open-pawn-panel",
      name: "Open Pawn panel",
      callback: async () => {
        await this.activatePanel();
      }
    });
    this.addCommand({
      id: "approve-pawn-task",
      name: "Approve Pawn task",
      editorCallback: async (_editor, view) => {
        if (!view.file) return;
        const tasks = await listTasksForNote(
          this.app,
          view.file.path,
          this.settings.agentRoot
        );
        const review = tasks.find((t) => t.status === "review") || tasks[0];
        if (!review) {
          new import_obsidian3.Notice("No Pawn task for this note");
          return;
        }
        await setApproved(this.app, review.path, true);
        if (await healthCheck(this.settings)) {
          try {
            await approveTask(this.settings, review.id, review.result);
            new import_obsidian3.Notice("Approved and indexed");
          } catch {
            new import_obsidian3.Notice("Approved locally; watcher will index");
          }
        } else {
          new import_obsidian3.Notice("Approved locally; will sync");
        }
      }
    });
    this.registerEvent(
      this.app.workspace.on("editor-menu", (menu, editor, view) => {
        if (!(view instanceof import_obsidian3.MarkdownView)) return;
        menu.addItem((item) => {
          item.setTitle("Ask Pawn").setIcon("bot").onClick(async () => {
            await this.askPawn(editor, view);
          });
        });
      })
    );
    this.addRibbonIcon("bot", "Open Pawn panel", async () => {
      await this.activatePanel();
    });
    this.addSettingTab(new PawnSettingTab(this.app, this));
    this.statusEl = this.addStatusBarItem();
    this.statusEl.setText("Pawn \u2026");
    this.registerInterval(
      window.setInterval(() => {
        void this.refreshStatus();
      }, 15e3)
    );
    await this.refreshStatus();
  }
  onunload() {
    this.app.workspace.detachLeavesOfType(PAWN_VIEW_TYPE);
  }
  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }
  async saveSettings() {
    await this.saveData(this.settings);
  }
  async activatePanel() {
    const { workspace } = this.app;
    let leaf = workspace.getLeavesOfType(PAWN_VIEW_TYPE)[0];
    if (!leaf) {
      const right = workspace.getRightLeaf(false);
      leaf = right ?? workspace.getLeaf(true);
      await leaf.setViewState({ type: PAWN_VIEW_TYPE, active: true });
    }
    workspace.revealLeaf(leaf);
  }
  async refreshStatus() {
    this.reachability = await healthCheck(this.settings);
    const file = this.app.workspace.getActiveFile();
    let pending = 0;
    if (file) {
      const tasks = await listTasksForNote(
        this.app,
        file.path,
        this.settings.agentRoot
      );
      pending = tasks.filter((t) => t.status === "review").length;
    }
    if (this.statusEl) {
      this.statusEl.setText(
        `Pawn ${this.reachability ? "online" : "offline"}${pending ? ` \xB7 ${pending} review` : ""}`
      );
    }
  }
  selectionOrParagraph(editor) {
    if (editor.somethingSelected()) {
      return editor.getSelection();
    }
    const cursor = editor.getCursor();
    return editor.getLine(cursor.line);
  }
  async askPawn(editor, view) {
    if (!view.file) {
      new import_obsidian3.Notice("Open a Markdown file first");
      return;
    }
    const context = this.selectionOrParagraph(editor).trim();
    const instruction = window.prompt(
      "Ask Pawn:",
      context ? `Regarding the selection:
${context.slice(0, 200)}` : ""
    ) || "";
    if (!instruction.trim()) return;
    const id = uuid4();
    const notePath = view.file.path;
    const conversation = conversationForNote(notePath);
    const taskPath = await createTaskNote(this.app, this.settings, {
      id,
      instruction: instruction.trim(),
      notePath,
      context: context || void 0,
      conversation
    });
    insertCallout(editor, taskPath, instruction.trim().split("\n")[0]);
    new import_obsidian3.Notice("Pawn task created");
    if (this.settings.alwaysQueue) {
      new import_obsidian3.Notice("Queued for vault sync (always-queue on)");
      return;
    }
    const online = await healthCheck(this.settings);
    if (!online) {
      new import_obsidian3.Notice("Server offline \u2014 task will run after sync");
      return;
    }
    new import_obsidian3.Notice("Asking Pawn\u2026");
    try {
      const { status, data } = await createTask(this.settings, {
        id,
        instruction: instruction.trim(),
        note_path: notePath,
        context: context || void 0,
        conversation,
        timeout_seconds: Math.max(
          5,
          Math.floor(this.settings.fastPathTimeoutMs / 1e3)
        )
      });
      if ((status === 200 || status === 201) && data.result) {
        await writeTaskResult(this.app, taskPath, data.result, "review");
        new import_obsidian3.Notice("Pawn replied (review in panel)");
      } else if (status === 202) {
        new import_obsidian3.Notice("Accepted \u2014 result will arrive via sync");
      } else {
        new import_obsidian3.Notice(
          `Pawn: ${data.error_code || status} \u2014 left as todo for sync`
        );
      }
    } catch (err) {
      new import_obsidian3.Notice(`Fast path failed (${String(err)}); left as todo`);
    }
    await this.refreshStatus();
  }
};
var PawnSettingTab = class extends import_obsidian3.PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }
  display() {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h2", { text: "Pawn" });
    new import_obsidian3.Setting(containerEl).setName("Server URL").setDesc("pawn-server base URL (reachable from this device)").addText(
      (text) => text.setPlaceholder("http://127.0.0.1:8000").setValue(this.plugin.settings.serverUrl).onChange(async (value) => {
        this.plugin.settings.serverUrl = value.trim();
        await this.plugin.saveSettings();
      })
    );
    new import_obsidian3.Setting(containerEl).setName("API token").setDesc("Same as api.token in pawnai.yaml").addText(
      (text) => text.setPlaceholder("Bearer token").setValue(this.plugin.settings.apiToken).onChange(async (value) => {
        this.plugin.settings.apiToken = value.trim();
        await this.plugin.saveSettings();
      })
    );
    new import_obsidian3.Setting(containerEl).setName("Agent root").setDesc("Vault folder Pawn owns (default Pawn)").addText(
      (text) => text.setValue(this.plugin.settings.agentRoot).onChange(async (value) => {
        this.plugin.settings.agentRoot = value.trim() || "Pawn";
        await this.plugin.saveSettings();
      })
    );
    new import_obsidian3.Setting(containerEl).setName("Fast-path timeout (ms)").setDesc("Wait this long for a direct reply before falling back to sync").addText(
      (text) => text.setValue(String(this.plugin.settings.fastPathTimeoutMs)).onChange(async (value) => {
        const n = Number(value);
        if (!Number.isNaN(n) && n >= 1e3) {
          this.plugin.settings.fastPathTimeoutMs = n;
          await this.plugin.saveSettings();
        }
      })
    );
    new import_obsidian3.Setting(containerEl).setName("Always queue").setDesc("Skip HTTP fast path; always leave tasks as todo for the watcher").addToggle(
      (toggle) => toggle.setValue(this.plugin.settings.alwaysQueue).onChange(async (value) => {
        this.plugin.settings.alwaysQueue = value;
        await this.plugin.saveSettings();
      })
    );
  }
};
