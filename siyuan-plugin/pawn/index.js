/**
 * Pawn — SiYuan plugin
 *
 * Sends the current block to pawn-server POST /v1/siyuan/triggers via
 * /api/network/forwardProxy. Wraps as a TIP callout on send by default.
 * @pawn is not required for Send (watcher discovery still uses it).
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
  // Match only at the start of THIS block's kramdown. Do not use /m — parent /
  // document kramdown includes children, and a tip elsewhere would false-positive
  // and skip wrapping the selected paragraph.
  return /^\s*>\s*\[!TIP\]/i.test(text || "");
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
  const body = String(instruction || "").trim() || "Pawn task";
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

function formatErrorDetail(value) {
  if (value == null || value === "") return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (item == null) return "";
        if (typeof item === "string") return item;
        if (typeof item === "object") {
          return (
            item.msg ||
            item.message ||
            item.detail ||
            JSON.stringify(item)
          );
        }
        return String(item);
      })
      .filter(Boolean)
      .join("; ");
  }
  if (typeof value === "object") {
    return (
      value.msg ||
      value.message ||
      value.detail ||
      JSON.stringify(value)
    );
  }
  return String(value);
}

/** Collect block ids from insertBlock / updateBlock API responses (order preserved). */
function operationIds(resp) {
  const data = resp && resp.data;
  const txs = Array.isArray(data)
    ? data
    : data && Array.isArray(data.transactions)
      ? data.transactions
      : [];
  const ids = [];
  const seen = new Set();
  const push = (id) => {
    if (!id || seen.has(id)) return;
    seen.add(id);
    ids.push(id);
  };
  for (const tx of txs) {
    const ops = (tx && tx.doOperations) || [];
    for (const op of ops) {
      if (!op) continue;
      push(op.id);
      // Nested ids in generated DOM (callout container + children).
      const html = typeof op.data === "string" ? op.data : "";
      const re = /data-node-id="([^"]+)"/g;
      let m;
      while ((m = re.exec(html))) push(m[1]);
    }
  }
  return ids;
}

/** Prefer a blockquote/callout container id from insert op DOM. */
function blockquoteIdFromInsert(resp) {
  const data = resp && resp.data;
  const txs = Array.isArray(data)
    ? data
    : data && Array.isArray(data.transactions)
      ? data.transactions
      : [];
  for (const tx of txs) {
    for (const op of (tx && tx.doOperations) || []) {
      const html = op && typeof op.data === "string" ? op.data : "";
      if (!html) continue;
      let m = html.match(
        /data-node-id="([^"]+)"[^>]*\bdata-type="NodeBlockquote"/
      );
      if (m) return m[1];
      m = html.match(
        /\bdata-type="NodeBlockquote"[^>]*data-node-id="([^"]+)"/
      );
      if (m) return m[1];
      m = html.match(/data-node-id="([^"]+)"[^>]*\bclass="[^"]*\bbq\b/);
      if (m) return m[1];
    }
  }
  return null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** UTF-8 string → base64 for SiYuan forwardProxy payloadEncoding=base64. */
function utf8ToBase64(text) {
  const s = String(text || "");
  if (typeof btoa === "function") {
    return btoa(unescape(encodeURIComponent(s)));
  }
  // Node / uncommon runtimes
  return Buffer.from(s, "utf8").toString("base64");
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
        (this.i18n && this.i18n.noCallout) || "No block content to send",
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
        // 0.1.10 forced wrap off in saved settings; re-enable once on upgrade.
        if (data.wrapPrefMigrated !== true) {
          this.config.autoWrapCallout = true;
          this.config.wrapPrefMigrated = true;
          await this.saveConfig();
        }
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
        conf.wrapPrefMigrated = true;
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
      description:
        i18n.mentionTokenDesc ||
        "Optional — used only by the server watcher SQL scan (must match siyuan_watcher.mention_token). Plugin Send does not require it.",
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
        "Convert the current paragraph into a TIP callout when sending (no @pawn required)",
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
   * True when *blockId* is a real TIP callout container (not a paragraph
   * whose text merely starts with "> [!TIP]").
   */
  async _isTipCalloutBlock(blockId) {
    const kdResp = await fetchSyncPost("/api/block/getBlockKramdown", {
      id: blockId,
    });
    const kd = stripIal(
      (kdResp && kdResp.data && kdResp.data.kramdown) || ""
    );
    if (!looksLikeTipCallout(kd)) return false;
    const rowResp = await fetchSyncPost("/api/query/sql", {
      stmt:
        "SELECT type FROM blocks WHERE id = '" +
        blockId.replace(/'/g, "''") +
        "' LIMIT 1",
    });
    const rows = (rowResp && rowResp.data) || [];
    const typ = rows[0] && String(rows[0].type || "").toLowerCase();
    // Paragraph that only contains tip markdown text is not a callout.
    if (typ === "p") return false;
    return true;
  }

  /**
   * Pick the TIP callout id from an insertBlock response. Prefers a real
   * callout container; falls back to walking parents of inserted ids.
   */
  async _pickInsertedCalloutId(ins) {
    const fromDom = blockquoteIdFromInsert(ins);
    if (fromDom && (await this._isTipCalloutBlock(fromDom))) return fromDom;
    if (fromDom) return fromDom;

    const ids = operationIds(ins);
    for (const id of ids) {
      if (await this._isTipCalloutBlock(id)) return id;
    }
    for (const id of ids) {
      const ancestor = await this._resolveCalloutRoot(id);
      if (ancestor && (await this._isTipCalloutBlock(ancestor))) {
        return ancestor;
      }
    }
    return ids[0] || null;
  }

  /** Poll until SiYuan SQL can see *blockId* (avoids trigger 404 on fresh inserts). */
  async _waitForBlock(blockId, attempts, delayMs) {
    const n = Math.max(1, attempts || 25);
    const ms = Math.max(20, delayMs || 80);
    for (let i = 0; i < n; i++) {
      const rowResp = await fetchSyncPost("/api/query/sql", {
        stmt:
          "SELECT id FROM blocks WHERE id = '" +
          String(blockId).replace(/'/g, "''") +
          "' LIMIT 1",
      });
      const rows = (rowResp && rowResp.data) || [];
      if (rows.length) return true;
      await sleep(ms);
    }
    return false;
  }

  /**
   * Wrap *blockId* into a TIP callout when it is not already one (and not
   * already inside one).
   *
   * SiYuan 3.8 rejects updateBlock for paragraph→TIP type changes, and
   * insert+delete left a brand-new id that pawn-server often 404'd on
   * (not indexed yet). Instead: insert an empty TIP shell, move the
   * original block into it, trigger the *original* id (server walks up
   * to the TIP).
   */
  async _ensureWrapped(blockId) {
    if (!blockId || this._wrapping.has(blockId)) return blockId;

    if (await this._hasTipCalloutAncestor(blockId)) {
      return (await this._resolveCalloutRoot(blockId)) || blockId;
    }

    const kdResp = await fetchSyncPost("/api/block/getBlockKramdown", {
      id: blockId,
    });
    const kramdown = stripIal(
      (kdResp && kdResp.data && kdResp.data.kramdown) || ""
    );
    if (!kramdown) return null;
    if (await this._isTipCalloutBlock(blockId)) return blockId;

    const token = this.config.mentionToken || "@pawn";
    const icon = this.config.calloutIcon || "🤖";
    const title = extractTitle(kramdown, token, 60).replace(/\n/g, " ").trim();
    // Empty body placeholder — original paragraph is moved in next.
    const md = "> [!TIP] " + icon + " " + title + "\n> \n";

    this._wrapping.add(blockId);
    try {
      const ins = await fetchSyncPost("/api/block/insertBlock", {
        dataType: "markdown",
        data: md,
        nextID: blockId,
      });
      if (ins && ins.code !== 0) {
        console.warn("pawn: insertBlock failed", blockId, ins);
        showMessage(
          "Could not wrap callout: " + formatErrorDetail(ins.msg || ins),
          5000,
          "error"
        );
        return blockId;
      }
      const calloutId = await this._pickInsertedCalloutId(ins);
      if (!calloutId) {
        console.warn("pawn: insertBlock returned no id", ins);
        showMessage("Could not wrap callout: no new block id", 5000, "error");
        return blockId;
      }

      const ready = await this._waitForBlock(calloutId, 25, 80);
      if (!ready) {
        console.warn("pawn: callout not indexed yet", calloutId);
      }

      const moved = await fetchSyncPost("/api/block/moveBlock", {
        id: blockId,
        parentID: calloutId,
      });
      if (moved && moved.code !== 0) {
        console.warn("pawn: moveBlock failed", blockId, calloutId, moved);
        // Fall back: keep shell + leave original sibling; still try original id.
        showMessage(
          "Could not move into callout: " +
            formatErrorDetail(moved.msg || moved),
          5000,
          "error"
        );
        return blockId;
      }

      // Wait until the original block is parented under the tip in SQL.
      for (let i = 0; i < 20; i++) {
        if (await this._hasTipCalloutAncestor(blockId)) break;
        await sleep(80);
      }

      // Drop empty placeholder children left by the tip shell.
      try {
        const kids = await fetchSyncPost("/api/block/getChildBlocks", {
          id: calloutId,
        });
        const list = (kids && kids.data) || [];
        for (const child of list) {
          const cid = child && child.id;
          if (!cid || cid === blockId) continue;
          const cKd = await fetchSyncPost("/api/block/getBlockKramdown", {
            id: cid,
          });
          const text = stripIal(
            (cKd && cKd.data && cKd.data.kramdown) || ""
          );
          if (!text) {
            await fetchSyncPost("/api/block/deleteBlock", { id: cid });
          }
        }
      } catch (e) {
        console.warn("pawn: cleanup placeholder children failed", e);
      }

      await fetchSyncPost("/api/attr/setBlockAttrs", {
        id: calloutId,
        attrs: { [ATTR_STATUS]: "draft" },
      });
      // Return the original id — it still exists as a tip child; the server
      // walks up to the TIP. Avoids 404 on a brand-new callout id.
      return blockId;
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

  /**
   * Prefer a TIP callout ancestor; else the starting block if non-empty.
   * @pawn is optional (watcher-only); plugin Send does not require it.
   */
  async _resolveCalloutRoot(blockId) {
    let current = blockId;
    const seen = new Set();
    let startKd = "";

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
      if (current === blockId) startKd = kd;
      if (looksLikeTipCallout(kd)) {
        return current;
      }
      const parent = row.parent_id;
      const root = row.root_id;
      if (!parent || parent === current || current === root) break;
      current = parent;
    }
    if (stripIal(startKd)) return blockId;
    return null;
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
    } else {
      console.info("pawn: autoWrapCallout is off — sending without wrap");
    }
    if (!rootId) {
      rootId = await this._resolveCalloutRoot(blockId);
    } else {
      // After wrap-in-place, prefer the TIP parent for attrs + trigger.
      rootId = (await this._resolveCalloutRoot(rootId)) || rootId;
    }
    if (!rootId) {
      showMessage(
        i18n.noCallout || "No block content to send",
        4000,
        "error"
      );
      return;
    }

    await this._waitForBlock(rootId, 15, 80);

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

    // Only Authorization here — Content-Type comes from contentType below.
    // Duplicating Content-Type in headers has been flaky with some kernels.
    const headers = [];
    if (this.config.apiToken) {
      headers.push({ Authorization: "Bearer " + this.config.apiToken });
    }

    // SiYuan 3.8 forwardProxy: payloadEncoding "text" sends no body; "json"
    // with an object is also unreliable across kernel builds. Base64 of the
    // JSON string always hits request.SetBody(decoded) in the kernel.
    const bodyJson = JSON.stringify({ block_id: rootId });
    let proxy = null;
    let statusCode = null;
    let bodyText = "";
    let parsed = null;

    // Fresh callouts can 404 once if the kernel index lags; retry briefly.
    for (let attempt = 0; attempt < 4; attempt++) {
      if (attempt > 0) await sleep(150 * attempt);
      proxy = await fetchSyncPost("/api/network/forwardProxy", {
        url: serverUrl + "/v1/siyuan/triggers",
        method: "POST",
        timeout: Math.max(1000, Number(this.config.requestTimeoutMs) || 15000),
        contentType: "application/json",
        headers,
        payload: utf8ToBase64(bodyJson),
        payloadEncoding: "base64",
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
      statusCode = data && data.status;
      bodyText = (data && data.body) || "";
      if (bodyText && typeof bodyText !== "string") {
        bodyText = JSON.stringify(bodyText);
      }
      parsed = null;
      try {
        parsed = bodyText ? JSON.parse(bodyText) : null;
      } catch (_e) {
        parsed = null;
      }

      if (statusCode === 202 || statusCode === 200) break;
      if (statusCode !== 404) break;
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
