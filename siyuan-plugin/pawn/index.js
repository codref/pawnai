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
  wrapDebounceMs: 400,
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

module.exports = class PawnPlugin extends Plugin {
  async onload() {
    this.config = Object.assign({}, DEFAULTS);
    await this.loadConfig();
    this._wrapTimers = new Map();
    this._wrapping = new Set();

    this.eventBus.on("ws-main", this._onWsMain);
    this.eventBus.on("click-blockicon", this._onBlockIcon);
  }

  onunload() {
    this.eventBus.off("ws-main", this._onWsMain);
    this.eventBus.off("click-blockicon", this._onBlockIcon);
    for (const t of this._wrapTimers.values()) clearTimeout(t);
    this._wrapTimers.clear();
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
    let elDebounce;
    let elIcon;
    let elLabel;
    let elTimeout;

    const setting = new Setting({
      confirmCallback: async () => {
        conf.serverUrl = (elServer.value || "").trim().replace(/\/+$/, "");
        conf.apiToken = elToken.value || "";
        conf.mentionToken = (elMention.value || "@pawn").trim() || "@pawn";
        conf.autoWrapCallout = !!elAuto.checked;
        conf.wrapDebounceMs = Math.max(
          0,
          parseInt(elDebounce.value, 10) || DEFAULTS.wrapDebounceMs
        );
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
      title: i18n.autoWrapCallout || "Auto-wrap callout",
      description: i18n.autoWrapCalloutDesc || "",
      createActionElement: () => {
        elAuto = document.createElement("input");
        elAuto.type = "checkbox";
        elAuto.className = "b3-switch fn__flex-center";
        elAuto.checked = conf.autoWrapCallout !== false;
        return elAuto;
      },
    });
    setting.addItem({
      title: i18n.wrapDebounceMs || "Wrap debounce (ms)",
      createActionElement: () => {
        elDebounce = document.createElement("input");
        elDebounce.className = "b3-text-field fn__flex-center fn__size200";
        elDebounce.type = "number";
        elDebounce.min = "0";
        elDebounce.value = String(
          conf.wrapDebounceMs ?? DEFAULTS.wrapDebounceMs
        );
        return elDebounce;
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
  }

  _onWsMain = (event) => {
    if (!this.config.autoWrapCallout) return;
    const detail = event.detail || {};
    if (detail.cmd !== "transactions") return;
    const data = detail.data;
    if (!Array.isArray(data)) return;
    for (const tx of data) {
      const ops = (tx && tx.doOperations) || [];
      for (const op of ops) {
        if (!op || !op.id) continue;
        if (op.action !== "update" && op.action !== "insert") continue;
        this._scheduleMaybeWrap(op.id);
      }
    }
  };

  _scheduleMaybeWrap(blockId) {
    if (this._wrapping.has(blockId)) return;
    const prev = this._wrapTimers.get(blockId);
    if (prev) clearTimeout(prev);
    const ms = Math.max(0, Number(this.config.wrapDebounceMs) || 0);
    const timer = setTimeout(() => {
      this._wrapTimers.delete(blockId);
      this._maybeWrapBlock(blockId).catch((e) =>
        console.warn("pawn: wrap failed", blockId, e)
      );
    }, ms);
    this._wrapTimers.set(blockId, timer);
  }

  async _maybeWrapBlock(blockId) {
    if (this._wrapping.has(blockId)) return;
    const kdResp = await fetchSyncPost("/api/block/getBlockKramdown", {
      id: blockId,
    });
    const kramdown = stripIal(
      (kdResp && kdResp.data && kdResp.data.kramdown) || ""
    );
    if (!kramdown || looksLikeTipCallout(kramdown)) return;

    const token = this.config.mentionToken || "@pawn";
    // Require mention at start plus following whitespace (committed prefix).
    const re = new RegExp("^\\s*" + escapeRegExp(token) + "\\s+", "i");
    if (!re.test(kramdown)) return;

    this._wrapping.add(blockId);
    try {
      const md = buildTipCalloutMarkdown(kramdown, {
        mentionToken: token,
        calloutIcon: this.config.calloutIcon,
      });
      await fetchSyncPost("/api/block/updateBlock", {
        dataType: "markdown",
        data: md,
        id: blockId,
      });
      await fetchSyncPost("/api/attr/setBlockAttrs", {
        id: blockId,
        attrs: { [ATTR_STATUS]: "draft" },
      });
    } finally {
      this._wrapping.delete(blockId);
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

    const rootId = await this._resolveCalloutRoot(blockId);
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
