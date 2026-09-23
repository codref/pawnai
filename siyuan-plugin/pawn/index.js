/**
 * Pawn — SiYuan plugin
 *
 * Inserts a pawn/prompt custom block (slash inserts only; Send converts the
 * selection and posts its block id) to pawn-server POST /v1/siyuan/triggers
 * via /api/network/forwardProxy.
 */
const { Plugin, Setting, fetchSyncPost, showMessage } = require("siyuan");

const CONFIG_KEY = "config.json";
const ATTR_STATUS = "custom-agent-status";
const PROMPT_INFO = "pawn/prompt";
const BUSY_STATUSES = new Set(["queued", "claimed", "running"]);

const DEFAULTS = {
  serverUrl: "http://127.0.0.1:8000",
  apiToken: "",
  sendButtonLabel: "Send to Pawn",
  requestTimeoutMs: 15000,
};

function stripIal(text) {
  return String(text || "")
    .replace(/\{:[^}]*\}/g, "")
    .trim();
}

/** A line that is only ;;; would close the custom-block fence. */
function escapeFenceLines(text) {
  return String(text || "")
    .split("\n")
    .map((line) => (line.trim() === ";;;" ? line + " " : line))
    .join("\n");
}

function buildPromptMarkdown(content) {
  const body = escapeFenceLines(content).replace(/\s+$/, "");
  return ";;;" + PROMPT_INFO + "\n" + (body ? body + "\n" : "") + ";;;\n";
}

function extractPromptInner(kramdown) {
  const text = stripIal(kramdown);
  if (!text.startsWith(";;;" + PROMPT_INFO)) return null;
  const match = text.match(
    /^;;;pawn\/prompt[ \t]*\n([\s\S]*?)\n?;;;[ \t]*$/
  );
  if (!match) return null;
  return match[1].replace(/\n$/, "");
}

function readEditorText(editor) {
  const raw = editor && typeof editor.innerText === "string" ? editor.innerText : "";
  return escapeFenceLines(raw.replace(/\u00a0/g, " ")).replace(/\n$/, "");
}

function isPawnPrompt(el) {
  return (
    !!el &&
    el.getAttribute("data-type") === "NodeCustomBlock" &&
    el.getAttribute("data-info") === PROMPT_INFO
  );
}

function asIProtyle(protyle) {
  if (protyle && protyle.protyle && protyle.protyle.wysiwyg) return protyle.protyle;
  return protyle;
}

function getActiveBlockElement(protyle) {
  try {
    const range = protyle && protyle.toolbar && protyle.toolbar.range;
    let node = range && range.startContainer;
    if (!node) {
      const sel = window.getSelection();
      node = sel && sel.anchorNode;
    }
    const el = node && (node.nodeType === 3 ? node.parentElement : node);
    return (el && el.closest && el.closest("[data-node-id]")) || null;
  } catch (_e) {
    return null;
  }
}

function topLevelBlocks(elements) {
  const list = (elements || []).filter((el) => el && el.getAttribute("data-node-id"));
  const tops = list.filter(
    (el) => !list.some((other) => other !== el && other.contains(el))
  );
  return tops.sort((a, b) => {
    if (a === b) return 0;
    const pos = a.compareDocumentPosition(b);
    if (pos & Node.DOCUMENT_POSITION_FOLLOWING) return -1;
    if (pos & Node.DOCUMENT_POSITION_PRECEDING) return 1;
    return 0;
  });
}

function getSelectedBlockElements(protyle) {
  const root = protyle && protyle.wysiwyg && protyle.wysiwyg.element;
  const selected = root
    ? Array.from(root.querySelectorAll(".protyle-wysiwyg--select"))
    : [];
  const blocks = topLevelBlocks(selected);
  if (blocks.length) return blocks;
  const active = getActiveBlockElement(protyle);
  return active ? [active] : [];
}

function formatErrorDetail(value) {
  if (value == null || value === "") return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (item == null) return "";
        if (typeof item === "string") return item;
        if (typeof item === "object") {
          return item.msg || item.message || item.detail || JSON.stringify(item);
        }
        return String(item);
      })
      .filter(Boolean)
      .join("; ");
  }
  if (typeof value === "object") {
    return value.msg || value.message || value.detail || JSON.stringify(value);
  }
  return String(value);
}

module.exports = class PawnPlugin extends Plugin {
  constructor(...args) {
    super(...args);
    this.config = Object.assign({}, DEFAULTS);
  }

  async onload() {
    await this.loadConfig();

    const label =
      (this.i18n && this.i18n.pawnPrompt) || "Pawn prompt";
    this.protyleSlash = [
      {
        filter: ["pawn", "prompt"],
        id: "pawnPrompt",
        html:
          '<div class="b3-list-item__first"><span class="b3-list-item__text">' +
          label +
          "</span></div>",
        callback: (_protyle, nodeElement) => {
          this._insertEmptyPrompt(nodeElement).catch((e) => {
            console.error("pawn: insert prompt failed", e);
            showMessage(
              (this.i18n && this.i18n.insertFail) || "Could not insert Pawn prompt",
              5000,
              "error"
            );
          });
        },
      },
    ];
    this.customBlockRenders = {
      prompt: (options) => this._renderPrompt(options),
    };

    this.eventBus.on("click-blockicon", this._onBlockIcon);

    this.addCommand({
      langKey: "sendToPawn",
      hotkey: "⌥⌘P",
      editorCallback: (protyle) => {
        this._sendFromProtyle(protyle);
      },
    });
  }

  onunload() {
    this.eventBus.off("click-blockicon", this._onBlockIcon);
  }

  updateProtyleToolbar(toolbar) {
    const conf = this.config || DEFAULTS;
    const tip =
      conf.sendButtonLabel ||
      (this.i18n && this.i18n.sendToPawn) ||
      "Send to Pawn";
    toolbar.push("|");
    toolbar.push({
      name: "sendToPawn",
      icon: "iconSend",
      tipPosition: "n",
      tip,
      hotkey: "⌥⌘P",
      click: (protyle) => {
        this._sendFromProtyle(protyle);
      },
    });
    return toolbar;
  }

  _renderPrompt({ element, content, setContent }) {
    const editor = document.createElement("div");
    editor.className = "pawn-prompt__editor";
    editor.contentEditable = "plaintext-only";
    editor.spellcheck = true;
    editor.setAttribute(
      "data-placeholder",
      (this.i18n && this.i18n.promptPlaceholder) || "Pawn prompt"
    );
    if (content) editor.textContent = content;
    element.append(editor);

    const persist = () => {
      const text = readEditorText(editor);
      const stored = String(content || "").replace(/\n$/, "");
      if (text === stored) return;
      if (setContent(text)) return;
      const block = element.closest && element.closest("[data-node-id]");
      const id = block && block.getAttribute("data-node-id");
      if (!id) return;
      fetchSyncPost("/api/block/updateBlock", {
        id,
        dataType: "markdown",
        data: buildPromptMarkdown(text),
      }).catch((e) => console.warn("pawn: persist prompt failed", e));
    };
    editor.addEventListener("blur", persist);
  }

  _sendFromProtyle(protyle) {
    const blocks = getSelectedBlockElements(asIProtyle(protyle));
    this._sendBlocks(blocks).catch((e) => {
      console.error("pawn: send failed", e);
      showMessage(
        ((this.i18n && this.i18n.sendFail) || "Pawn trigger failed") +
          (e && e.message ? ": " + e.message : ""),
        5000,
        "error"
      );
    });
  }

  async loadConfig() {
    try {
      const data = await this.loadData(CONFIG_KEY);
      if (data && typeof data === "object") {
        this.config = Object.assign({}, DEFAULTS, data);
      }
    } catch (e) {
      console.warn("pawn: loadConfig failed", e);
    }
  }

  async saveConfig() {
    await this.saveData(CONFIG_KEY, this.config);
  }

  openSetting() {
    const i18n = this.i18n || {};
    const conf = this.config;
    let elServer;
    let elToken;
    let elLabel;
    let elTimeout;

    const setting = new Setting({
      confirmCallback: async () => {
        conf.serverUrl = (elServer.value || "").trim().replace(/\/+$/, "");
        conf.apiToken = elToken.value || "";
        conf.sendButtonLabel =
          (elLabel.value || DEFAULTS.sendButtonLabel).trim() ||
          DEFAULTS.sendButtonLabel;
        conf.requestTimeoutMs = Math.max(
          1000,
          parseInt(elTimeout.value, 10) || DEFAULTS.requestTimeoutMs
        );
        await this.saveConfig();
      },
    });

    setting.addItem({
      title: i18n.serverUrl || "Server URL",
      description: i18n.serverUrlDesc || "",
      createActionElement: () => {
        elServer = document.createElement("input");
        elServer.className = "b3-text-field fn__flex-center fn__size200";
        elServer.type = "url";
        elServer.value = conf.serverUrl || DEFAULTS.serverUrl;
        return elServer;
      },
    });
    setting.addItem({
      title: i18n.apiToken || "API token",
      description: i18n.apiTokenDesc || "",
      createActionElement: () => {
        elToken = document.createElement("input");
        elToken.className = "b3-text-field fn__flex-center fn__size200";
        elToken.type = "password";
        elToken.autocomplete = "off";
        elToken.value = conf.apiToken || "";
        return elToken;
      },
    });
    setting.addItem({
      title: i18n.sendButtonLabel || "Send button label",
      createActionElement: () => {
        elLabel = document.createElement("input");
        elLabel.className = "b3-text-field fn__flex-center fn__size200";
        elLabel.value = conf.sendButtonLabel || DEFAULTS.sendButtonLabel;
        return elLabel;
      },
    });
    setting.addItem({
      title: i18n.requestTimeoutMs || "Request timeout (ms)",
      createActionElement: () => {
        elTimeout = document.createElement("input");
        elTimeout.className = "b3-text-field fn__flex-center fn__size200";
        elTimeout.type = "number";
        elTimeout.min = "1000";
        elTimeout.value = String(
          conf.requestTimeoutMs ?? DEFAULTS.requestTimeoutMs
        );
        return elTimeout;
      },
    });

    this.setting = setting;
    this.setting.open(this.displayName || this.name || "Pawn");
  }

  _onBlockIcon = (event) => {
    const detail = event.detail || {};
    const menu = detail.menu;
    const blockElements = detail.blockElements || [];
    if (!menu || !blockElements.length) return;

    const label =
      this.config.sendButtonLabel ||
      (this.i18n && this.i18n.sendToPawn) ||
      "Send to Pawn";

    menu.addItem({
      icon: "iconSend",
      label,
      click: () => {
        this._sendBlocks(topLevelBlocks(Array.from(blockElements))).catch((e) => {
          console.error("pawn: send failed", e);
          showMessage(
            ((this.i18n && this.i18n.sendFail) || "Pawn trigger failed") +
              (e && e.message ? ": " + e.message : ""),
            5000,
            "error"
          );
        });
      },
    });
  };

  async _insertEmptyPrompt(nodeElement) {
    const id = nodeElement && nodeElement.getAttribute("data-node-id");
    if (!id) {
      showMessage(
        (this.i18n && this.i18n.noBlock) || "No block to turn into a Pawn prompt",
        4000,
        "error"
      );
      return;
    }
    await this._updateMarkdown(id, buildPromptMarkdown(""));
  }

  async _updateMarkdown(id, markdown) {
    const resp = await fetchSyncPost("/api/block/updateBlock", {
      id,
      dataType: "markdown",
      data: markdown,
    });
    if (resp && resp.code !== 0) {
      throw new Error(formatErrorDetail(resp.msg || resp) || "updateBlock failed");
    }
    return resp;
  }

  async _blockMarkdown(id) {
    const resp = await fetchSyncPost("/api/block/getBlockKramdown", { id });
    if (resp && resp.code !== 0) {
      throw new Error(formatErrorDetail(resp.msg || resp) || "getBlockKramdown failed");
    }
    return (resp && resp.data && resp.data.kramdown) || "";
  }

  async _deleteBlock(id) {
    const resp = await fetchSyncPost("/api/block/deleteBlock", { id });
    if (resp && resp.code !== 0) {
      throw new Error(formatErrorDetail(resp.msg || resp) || "deleteBlock failed");
    }
  }

  async _contentForBlock(el) {
    if (isPawnPrompt(el)) {
      const editor = el.querySelector(".pawn-prompt__editor");
      if (editor) return readEditorText(editor);
      const stored = el.getAttribute("data-content");
      if (stored != null) return String(stored).replace(/\n$/, "");
    }
    const id = el.getAttribute("data-node-id");
    const kd = await this._blockMarkdown(id);
    const inner = extractPromptInner(kd);
    if (inner != null) return inner;
    return stripIal(kd);
  }

  async _persistPromptElement(el) {
    const id = el.getAttribute("data-node-id");
    const editor = el.querySelector(".pawn-prompt__editor");
    if (!editor || !id) return id;
    // Always commit through the kernel before POST. setContent on blur is
    // local and may not have flushed yet.
    await this._updateMarkdown(id, buildPromptMarkdown(readEditorText(editor)));
    return id;
  }

  /**
   * One existing prompt is sent as-is. Any other selection is replaced by a
   * single pawn/prompt block, then sent.
   */
  async _ensurePrompt(blocks) {
    if (!blocks.length) return null;
    if (blocks.length === 1 && isPawnPrompt(blocks[0])) {
      return this._persistPromptElement(blocks[0]);
    }
    const parts = [];
    for (const el of blocks) {
      const text = await this._contentForBlock(el);
      if (text) parts.push(text);
    }
    const markdown = buildPromptMarkdown(parts.join("\n\n"));
    const firstId = blocks[0].getAttribute("data-node-id");
    await this._updateMarkdown(firstId, markdown);
    for (const el of blocks.slice(1)) {
      const id = el.getAttribute("data-node-id");
      if (id && id !== firstId) await this._deleteBlock(id);
    }
    return firstId;
  }

  async _sendBlocks(blocks) {
    const i18n = this.i18n || {};
    if (!blocks.length) {
      showMessage(i18n.noBlock || "No block to turn into a Pawn prompt", 4000, "error");
      return;
    }
    const serverUrl = (this.config.serverUrl || "").replace(/\/+$/, "");
    if (!serverUrl) {
      showMessage(i18n.notConfigured || "Configure Pawn server URL", 4000, "error");
      return;
    }

    const blockId = await this._ensurePrompt(blocks);
    if (!blockId) {
      showMessage(i18n.noBlock || "No block to turn into a Pawn prompt", 4000, "error");
      return;
    }

    const attrsResp = await fetchSyncPost("/api/attr/getBlockAttrs", { id: blockId });
    const attrs = (attrsResp && attrsResp.data) || {};
    const status = String(attrs[ATTR_STATUS] || "").toLowerCase();
    if (BUSY_STATUSES.has(status)) {
      showMessage(
        i18n.sendBusy || "Pawn is already working on this prompt",
        4000,
        "info"
      );
      return;
    }

    const headers = [];
    if (this.config.apiToken) {
      headers.push({ Authorization: "Bearer " + this.config.apiToken });
    }

    // Kernel default payloadEncoding is "json": the object is the request body.
    const proxy = await fetchSyncPost("/api/network/forwardProxy", {
      url: serverUrl + "/v1/siyuan/triggers",
      method: "POST",
      timeout: Math.max(1000, Number(this.config.requestTimeoutMs) || 15000),
      contentType: "application/json",
      headers,
      payload: { block_id: blockId },
      payloadEncoding: "json",
      responseEncoding: "text",
    });

    if (proxy && proxy.code !== 0) {
      showMessage(
        (i18n.sendFail || "Pawn trigger failed") +
          ": " +
          formatErrorDetail(proxy.msg || proxy) +
          " (proxy)",
        6000,
        "error"
      );
      return;
    }

    const data = proxy && proxy.data;
    const statusCode = data && data.status;
    let bodyText = (data && data.body) || "";
    if (bodyText && typeof bodyText !== "string") {
      bodyText = JSON.stringify(bodyText);
    }
    let parsed = null;
    try {
      parsed = bodyText ? JSON.parse(bodyText) : null;
    } catch (_e) {
      parsed = null;
    }

    if (statusCode === 202 || statusCode === 200) {
      showMessage(
        (i18n.sendOk || "Pawn accepted the request") +
          (parsed && parsed.request_id ? " (" + parsed.request_id + ")" : ""),
        4000,
        "info"
      );
      return;
    }

    const detail =
      formatErrorDetail(parsed && (parsed.detail || parsed.message)) ||
      bodyText ||
      "HTTP " + String(statusCode == null ? "?" : statusCode);
    showMessage(
      (i18n.sendFail || "Pawn trigger failed") + ": " + detail,
      6000,
      "error"
    );
  }
};
