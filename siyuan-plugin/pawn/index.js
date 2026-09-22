/**
 * Pawn — SiYuan plugin
 *
 * Wraps @pawn paragraphs into TIP callouts and sends the callout root to
 * pawn-server POST /v1/siyuan/triggers via /api/network/forwardProxy.
 */
const { Plugin, Setting, fetchSyncPost, showMessage } = require("siyuan");

const CONFIG_KEY = "config.json";
const ATTR_STATUS = "custom-agent-status";
const BUSY_STATUSES = new Set(["queued", "claimed", "running"]);

const DEFAULTS = {
  serverUrl: "http://127.0.0.1:8000",
  apiToken: "",
  mentionToken: "@pawn",
  autoWrapCallout: true,
  calloutIcon: "🤖",
  sendButtonLabel: "Send to Pawn",
  requestTimeoutMs: 15000,
};

function escapeRegExp(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function looksLikeTipCallout(text) {
  return /^\s*>\s*\[!TIP\]/im.test(text || "");
}

function stripIal(text) {
  return String(text || "")
    .replace(/\{:[^}]*\}/g, "")
    .trim();
}

function extractTitle(body, mentionToken, maxLen) {
  const token = mentionToken || "@pawn";
  let t = String(body || "")
    .replace(new RegExp("^\\s*" + escapeRegExp(token) + "\\b\\s*", "i"), "")
    .trim();
  t = t.replace(/\(\([^)]*$/, "").trim();
  t = t.split(/\s+--\s+|\n/)[0].trim();
  t = t.replace(/\s+/g, " ");
  if (!t) return "Pawn task";
  if (t.length > maxLen) {
    let cut = t.slice(0, maxLen - 1);
    if (cut.includes(" ")) cut = cut.slice(0, cut.lastIndexOf(" "));
    t = cut.replace(/[.,;:]+$/, "") + "…";
  }
  return t.charAt(0).toUpperCase() + t.slice(1);
}

function buildTipCalloutMarkdown(instruction, opts) {
  const mentionToken = (opts && opts.mentionToken) || "@pawn";
  const icon = (opts && opts.calloutIcon) || "🤖";
  const body = String(instruction || "").trim() || mentionToken + " (empty)";
  const title =
    (opts && opts.title) || extractTitle(body, mentionToken, 60);
  const safeTitle = String(title).replace(/\n/g, " ").trim();
  const head = "> [!TIP] " + icon + " " + safeTitle;
  const quoted = body
    .split("\n")
    .map((line) => (line ? "> " + line : ">"))
    .join("\n");
  return head + "\n" + quoted + "\n";
}

function getActiveBlockId(protyle) {
  try {
    const range = protyle && protyle.toolbar && protyle.toolbar.range;
    let node = range && range.startContainer;
    if (!node) {
      const sel = window.getSelection();
      node = sel && sel.anchorNode;
    }
    const el = node && (node.nodeType === 3 ? node.parentElement : node);
    const block = el && el.closest && el.closest("[data-node-id]");
    return block ? block.getAttribute("data-node-id") : null;
  } catch (_e) {
    return null;
  }
}

module.exports = class PawnPlugin extends Plugin {
  constructor(...args) {
    super(...args);
    // SiYuan may call updateProtyleToolbar during construction, before onload.
    this.config = Object.assign({}, DEFAULTS);
    this._wrapping = new Set();
  }

  async onload() {
    await this.loadConfig();

    this.eventBus.on("click-blockicon", this._onBlockIcon);

    // Command palette / optional hotkey (Settings → keymap → Pawn).
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

  /**
   * Adds "Send to Pawn" to the floating selection toolbar (the bar that
   * appears above selected text).
   */
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

  _sendFromProtyle(protyle) {
    const p = (protyle && protyle.protyle) || protyle;
    const blockId = getActiveBlockId(p);
    if (!blockId) {
      showMessage(
        (this.i18n && this.i18n.noCallout) || "No @pawn callout found",
        4000,
        "error"
      );
      return;
    }
    this._sendBlock(blockId).catch((e) => {
      console.error("pawn: send failed", e);
      showMessage(
        (this.i18n && this.i18n.sendFail) || "Pawn trigger failed",
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
    let elMention;
    let elAuto;
    let elIcon;
    let elLabel;
    let elTimeout;

    const setting = new Setting({
      confirmCallback: async () => {
        conf.serverUrl = (elServer.value || "").trim().replace(/\/+$/, "");
        conf.apiToken = elToken.value || "";
        conf.mentionToken = (elMention.value || "@pawn").trim() || "@pawn";
        conf.autoWrapCallout = !!elAuto.checked;
        conf.calloutIcon = (elIcon.value || DEFAULTS.calloutIcon).trim();
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
      title: i18n.mentionToken || "Mention token",
      description: i18n.mentionTokenDesc || "",
      createActionElement: () => {
        elMention = document.createElement("input");
        elMention.className = "b3-text-field fn__flex-center fn__size200";
        elMention.value = conf.mentionToken || DEFAULTS.mentionToken;
        return elMention;
      },
    });
    setting.addItem({
      title: i18n.autoWrapCallout || "Wrap as callout on send",
      description:
        i18n.autoWrapCalloutDesc ||
        "Convert a leading @pawn paragraph into a TIP callout when sending",
      createActionElement: () => {
        elAuto = document.createElement("input");
        elAuto.type = "checkbox";
        elAuto.className = "b3-switch fn__flex-center";
        elAuto.checked = conf.autoWrapCallout !== false;
        return elAuto;
      },
    });
    setting.addItem({
      title: i18n.calloutIcon || "Callout icon",
      createActionElement: () => {
        elIcon = document.createElement("input");
        elIcon.className = "b3-text-field fn__flex-center fn__size200";
        elIcon.value = conf.calloutIcon || DEFAULTS.calloutIcon;
        return elIcon;
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
    // Base Plugin.openSetting() calls setting.open(); we overrode it, so open here.
    this.setting.open(this.displayName || this.name || "Pawn");
  }

  /**
   * True when blockId sits under a TIP callout (or an in-flight wrap).
   */
  async _hasTipCalloutAncestor(blockId) {
    let current = blockId;
    const seen = new Set();
    while (current && !seen.has(current)) {
      seen.add(current);
      const rowResp = await fetchSyncPost("/api/query/sql", {
        stmt:
          "SELECT id, parent_id, root_id FROM blocks WHERE id = '" +
          current.replace(/'/g, "''") +
          "' LIMIT 1",
      });
      const rows = (rowResp && rowResp.data) || [];
      if (!rows.length) break;
      const row = rows[0];
      const parent = row.parent_id;
      if (!parent || parent === current || current === row.root_id) break;
      if (this._wrapping.has(parent)) return true;
      const kdResp = await fetchSyncPost("/api/block/getBlockKramdown", {
        id: parent,
      });
      const kd = (kdResp && kdResp.data && kdResp.data.kramdown) || "";
      if (looksLikeTipCallout(kd)) return true;
      current = parent;
    }
    return false;
  }

  /**
   * If *blockId* (or its resolved mention root) is a plain @pawn paragraph,
   * rewrite it into a TIP callout. Returns the callout/mention block id.
   */
  async _ensureWrapped(blockId) {
    const resolved = await this._resolveCalloutRoot(blockId);
    const id = resolved || blockId;
    if (!id || this._wrapping.has(id)) return id;

    const kdResp = await fetchSyncPost("/api/block/getBlockKramdown", {
      id,
    });
    const kramdown = stripIal(
      (kdResp && kdResp.data && kdResp.data.kramdown) || ""
    );
    if (!kramdown) return null;
    if (looksLikeTipCallout(kramdown)) return id;
    if (await this._hasTipCalloutAncestor(id)) {
      return (await this._resolveCalloutRoot(id)) || id;
    }

    const token = this.config.mentionToken || "@pawn";
    const re = new RegExp("^\\s*" + escapeRegExp(token) + "\\b", "i");
    if (!re.test(kramdown)) return null;

    this._wrapping.add(id);
    try {
      const md = buildTipCalloutMarkdown(kramdown, {
        mentionToken: token,
        calloutIcon: this.config.calloutIcon,
        title: extractTitle(kramdown, token, 60),
      });
      await fetchSyncPost("/api/block/updateBlock", {
        dataType: "markdown",
        data: md,
        id,
      });
      await fetchSyncPost("/api/attr/setBlockAttrs", {
        id,
        attrs: { [ATTR_STATUS]: "draft" },
      });
      return id;
    } finally {
      this._wrapping.delete(id);
    }
  }

  _onBlockIcon = (event) => {
    const detail = event.detail || {};
    const menu = detail.menu;
    const blockElements = detail.blockElements || [];
    if (!menu || !blockElements.length) return;
    const blockId = blockElements[0].getAttribute("data-node-id");
    if (!blockId) return;

    const label =
      this.config.sendButtonLabel ||
      (this.i18n && this.i18n.sendToPawn) ||
      "Send to Pawn";

    menu.addItem({
      icon: "iconSend",
      label,
      click: () => {
        this._sendBlock(blockId).catch((e) => {
          console.error("pawn: send failed", e);
          showMessage(
            (this.i18n && this.i18n.sendFail) || "Pawn trigger failed",
            5000,
            "error"
          );
        });
      },
    });
  };

  async _resolveCalloutRoot(blockId) {
    const token = this.config.mentionToken || "@pawn";
    const mentionRe = new RegExp(
      "(?:^|[\\s>])" + escapeRegExp(token) + "\\b",
      "i"
    );
    let current = blockId;
    const seen = new Set();
    let plainCandidate = null;

    while (current && !seen.has(current)) {
      seen.add(current);
      const rowResp = await fetchSyncPost("/api/query/sql", {
        stmt:
          "SELECT id, parent_id, root_id FROM blocks WHERE id = '" +
          current.replace(/'/g, "''") +
          "' LIMIT 1",
      });
      const rows = (rowResp && rowResp.data) || [];
      if (!rows.length) break;
      const row = rows[0];
      const kdResp = await fetchSyncPost("/api/block/getBlockKramdown", {
        id: current,
      });
      const kd = (kdResp && kdResp.data && kdResp.data.kramdown) || "";
      if (looksLikeTipCallout(kd) && mentionRe.test(kd)) {
        return current;
      }
      const plain = stripIal(kd);
      if (
        !plainCandidate &&
        new RegExp("^\\s*" + escapeRegExp(token) + "\\b", "i").test(plain)
      ) {
        plainCandidate = current;
      }
      const parent = row.parent_id;
      const root = row.root_id;
      if (!parent || parent === current || current === root) break;
      current = parent;
    }
    return plainCandidate;
  }

  async _sendBlock(blockId) {
    const i18n = this.i18n || {};
    const serverUrl = (this.config.serverUrl || "").replace(/\/+$/, "");
    if (!serverUrl) {
      showMessage(i18n.notConfigured || "Configure Pawn server URL", 4000, "error");
      return;
    }

    let rootId = null;
    if (this.config.autoWrapCallout !== false) {
      rootId = await this._ensureWrapped(blockId);
    }
    if (!rootId) {
      rootId = await this._resolveCalloutRoot(blockId);
    }
    if (!rootId) {
      showMessage(i18n.noCallout || "No @pawn callout found", 4000, "error");
      return;
    }

    const attrsResp = await fetchSyncPost("/api/attr/getBlockAttrs", {
      id: rootId,
    });
    const attrs = (attrsResp && attrsResp.data) || {};
    const status = String(attrs[ATTR_STATUS] || "").toLowerCase();
    if (BUSY_STATUSES.has(status)) {
      showMessage(
        i18n.sendBusy || "Pawn is already working on this callout",
        4000,
        "info"
      );
      return;
    }

    const headers = [{ "Content-Type": "application/json" }];
    if (this.config.apiToken) {
      headers.push({ Authorization: "Bearer " + this.config.apiToken });
    }

    const proxy = await fetchSyncPost("/api/network/forwardProxy", {
      url: serverUrl + "/v1/siyuan/triggers",
      method: "POST",
      timeout: Math.max(1000, Number(this.config.requestTimeoutMs) || 15000),
      contentType: "application/json",
      headers,
      payload: { block_id: rootId },
      payloadEncoding: "text",
      responseEncoding: "text",
    });

    const data = proxy && proxy.data;
    const statusCode = data && data.status;
    const bodyText = (data && data.body) || "";
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
      (parsed && (parsed.detail || parsed.message)) ||
      bodyText ||
      "HTTP " + String(statusCode);
    showMessage(
      (i18n.sendFail || "Pawn trigger failed") + ": " + detail,
      6000,
      "error"
    );
  }
};
