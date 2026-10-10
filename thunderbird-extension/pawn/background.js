/** Background: menus, message → snippet, single-instance companion window. */

const MENU_MESSAGE = "pawn-capture-message";
const MENU_SELECTION = "pawn-capture-selection";
const MENU_ATTACHMENT = "pawn-capture-attachment";

const BODY_CHAR_CAP = 20000;
const IMAGE_BYTE_CAP = 4 * 1024 * 1024;
const DEFAULT_PANEL = { width: 380, height: 720, left: undefined, top: undefined };

/** @type {number|null} */
let panelWindowId = null;
/** Last known bounds while the companion window is open. */
let panelBoundsCache = null;

browser.runtime.onInstalled.addListener(() => {
  void setupMenus();
});

browser.runtime.onStartup?.addListener?.(() => {
  void setupMenus();
});

void setupMenus();

async function setupMenus() {
  try {
    await browser.menus.removeAll();
  } catch (_) {
    /* first run */
  }
  await browser.menus.create({
    id: MENU_MESSAGE,
    title: "Send to Pawn tray",
    contexts: ["message_list"],
  });
  await browser.menus.create({
    id: MENU_SELECTION,
    title: "Send selection to Pawn",
    contexts: ["selection"],
  });
  await browser.menus.create({
    id: MENU_ATTACHMENT,
    title: "Send image to Pawn",
    contexts: ["message_attachments"],
  });
}

browser.action.onClicked.addListener(async () => {
  await openPanel();
});

browser.messageDisplayAction.onClicked.addListener(async () => {
  await openPanel();
});

browser.menus.onClicked.addListener(async (info, tab) => {
  await openPanel();
  try {
    if (info.menuItemId === MENU_MESSAGE) {
      const messages = await collectMessageList(info.selectedMessages);
      if (!messages.length) {
        await pushPending({ error: "No messages selected" });
        return;
      }
      for (const msg of messages) {
        await enqueueMessage(msg);
      }
    } else if (info.menuItemId === MENU_SELECTION) {
      await captureSelection(info, tab);
    } else if (info.menuItemId === MENU_ATTACHMENT) {
      await captureAttachments(info, tab);
    }
  } catch (err) {
    await pushPending({ error: String(err?.message || err) });
  }
});

browser.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "pawn-add-displayed") {
    (async () => {
      try {
        await openPanel();
        const messages = await getDisplayedMessages();
        if (!messages.length) {
          sendResponse({ ok: false, error: "No message displayed" });
          return;
        }
        for (const m of messages) await enqueueMessage(m);
        sendResponse({ ok: true, count: messages.length });
      } catch (err) {
        const error = String(err?.message || err);
        await pushPending({ error });
        sendResponse({ ok: false, error });
      }
    })();
    return true;
  }
  if (msg?.type === "pawn-add-selected") {
    (async () => {
      try {
        await openPanel();
        const messages = await getSelectedMailMessages();
        if (!messages.length) {
          sendResponse({ ok: false, error: "No messages selected" });
          return;
        }
        for (const m of messages) await enqueueMessage(m);
        sendResponse({ ok: true, count: messages.length });
      } catch (err) {
        const error = String(err?.message || err);
        await pushPending({ error });
        sendResponse({ ok: false, error });
      }
    })();
    return true;
  }
  return false;
});

browser.windows.onRemoved.addListener((windowId) => {
  if (windowId !== panelWindowId) return;
  panelWindowId = null;
  if (panelBoundsCache) {
    void browser.storage.local.set({ panelBounds: panelBoundsCache });
  }
});

if (browser.windows.onBoundsChanged) {
  browser.windows.onBoundsChanged.addListener((window) => {
    if (window.id !== panelWindowId) return;
    panelBoundsCache = {
      left: window.left,
      top: window.top,
      width: window.width,
      height: window.height,
    };
  });
}

/** Open or focus the companion capture window. */
async function openPanel() {
  if (panelWindowId != null) {
    try {
      await browser.windows.update(panelWindowId, { focused: true });
      await refreshBoundsCache(panelWindowId);
      return;
    } catch (_) {
      panelWindowId = null;
    }
  }
  const stored = await browser.storage.local.get("panelBounds");
  const bounds = { ...DEFAULT_PANEL, ...(stored.panelBounds || {}) };
  const createData = {
    url: "panel.html",
    type: "popup",
    width: Math.max(320, bounds.width || DEFAULT_PANEL.width),
    height: Math.max(400, bounds.height || DEFAULT_PANEL.height),
  };
  if (Number.isFinite(bounds.left)) createData.left = bounds.left;
  if (Number.isFinite(bounds.top)) createData.top = bounds.top;
  const win = await browser.windows.create(createData);
  panelWindowId = win.id;
  panelBoundsCache = {
    left: win.left,
    top: win.top,
    width: win.width,
    height: win.height,
  };
}

async function refreshBoundsCache(windowId) {
  try {
    const win = await browser.windows.get(windowId);
    panelBoundsCache = {
      left: win.left,
      top: win.top,
      width: win.width,
      height: win.height,
    };
  } catch (_) {
    /* ignore */
  }
}

async function collectMessageList(list) {
  const out = [];
  let page = list;
  while (page) {
    for (const msg of page.messages || []) out.push(msg);
    if (!page.id) break;
    page = await browser.messages.continueList(page.id);
  }
  return out;
}

/** Resolve messages shown in a mail tab (not the companion popup). */
async function getDisplayedMessages() {
  for (const tabId of await mailTabIds()) {
    try {
      const listed = await browser.messageDisplay.getDisplayedMessages(tabId);
      const messages = await collectMessageList(asMessageList(listed));
      if (messages.length) return messages;
    } catch (_) {
      /* try next mail tab */
    }
  }
  try {
    const listed = await browser.messageDisplay.getDisplayedMessages();
    return await collectMessageList(asMessageList(listed));
  } catch (_) {
    return [];
  }
}

async function getSelectedMailMessages() {
  for (const tabId of await mailTabIds()) {
    try {
      const list = await browser.mailTabs.getSelectedMessages(tabId);
      const messages = await collectMessageList(list);
      if (messages.length) return messages;
    } catch (_) {
      /* try next mail tab */
    }
  }
  try {
    return await collectMessageList(await browser.mailTabs.getSelectedMessages());
  } catch (_) {
    return [];
  }
}

/** Active mail tabs first, then others — never the companion popup. */
async function mailTabIds() {
  try {
    const tabs = await browser.mailTabs.query({});
    const active = tabs.filter((t) => t.active);
    const rest = tabs.filter((t) => !t.active);
    return [...active, ...rest]
      .map((t) => t.tabId)
      .filter((id) => id != null);
  } catch (_) {
    return [];
  }
}

function asMessageList(listed) {
  if (Array.isArray(listed)) return { messages: listed, id: null };
  return listed;
}

async function captureSelection(info, tab) {
  const text = String(info.selectionText || "").trim();
  if (!text) {
    await pushPending({ error: "No text selected" });
    return;
  }
  let header = null;
  if (tab?.id != null) {
    try {
      const displayed = await browser.messageDisplay.getDisplayedMessages(tab.id);
      const messages = await collectMessageList(asMessageList(displayed));
      header = messages[0] || null;
    } catch (_) {
      /* selection may be outside a message pane */
    }
  }
  const subject = header?.subject || "selection";
  const mid = header?.headerMessageId ? `mid:${header.headerMessageId}` : "";
  const block = header
    ? `${formatHeaderBlock(header)}\n\n--- Selection ---\n\n${text}`
    : text;
  await enqueueSnippet({
    kind: "text",
    text: truncateBody(block),
    source_url: mid,
    page_title: subject,
  });
}

async function captureAttachments(info, tab) {
  const attachments = info.attachments || [];
  if (!attachments.length) {
    await pushPending({ error: "No attachment selected" });
    return;
  }
  let messageId = null;
  let header = null;
  if (tab?.id != null) {
    const displayed = await browser.messageDisplay.getDisplayedMessages(tab.id);
    const messages = await collectMessageList(asMessageList(displayed));
    header = messages[0] || null;
    messageId = header?.id;
  }
  if (messageId == null) {
    await pushPending({ error: "Could not resolve message for attachment" });
    return;
  }
  let added = 0;
  for (const att of attachments) {
    const ct = String(att.contentType || "").toLowerCase();
    if (!ct.startsWith("image/")) {
      await pushPending({
        error: `Skipped non-image attachment: ${att.name || att.partName}`,
      });
      continue;
    }
    if (att.size != null && att.size > IMAGE_BYTE_CAP) {
      await pushPending({
        error: `Skipped ${att.name || "image"} (>4MB)`,
      });
      continue;
    }
    const file = await browser.messages.getAttachmentFile(messageId, att.partName);
    if (file.size > IMAGE_BYTE_CAP) {
      await pushPending({
        error: `Skipped ${att.name || "image"} (>4MB)`,
      });
      continue;
    }
    const data_base64 = await fileToBase64(file);
    const mid = header?.headerMessageId ? `mid:${header.headerMessageId}` : "";
    await enqueueSnippet({
      kind: "image",
      data_base64,
      media_type: ct || file.type || "image/png",
      source_url: mid,
      page_title: header?.subject || att.name || "attachment",
      text: att.name ? `Attachment: ${att.name}` : "",
    });
    added += 1;
  }
  if (!added) {
    await pushPending({ error: "No image attachments queued" });
  }
}

async function enqueueMessage(msg) {
  const header = msg.id != null ? await browser.messages.get(msg.id) : msg;
  const body = await extractBody(header.id);
  const text = truncateBody(`${formatHeaderBlock(header)}\n\n${body}`.trim());
  const mid = header.headerMessageId ? `mid:${header.headerMessageId}` : "";
  await enqueueSnippet({
    kind: "text",
    text,
    source_url: mid,
    page_title: header.subject || "(no subject)",
  });
}

function formatHeaderBlock(header) {
  const lines = [
    `Subject: ${header.subject || "(no subject)"}`,
    `From: ${header.author || ""}`,
  ];
  if (header.recipients?.length) {
    lines.push(`To: ${header.recipients.join(", ")}`);
  }
  if (header.ccList?.length) {
    lines.push(`Cc: ${header.ccList.join(", ")}`);
  }
  if (header.date) {
    const d = header.date instanceof Date ? header.date : new Date(header.date);
    lines.push(`Date: ${d.toISOString()}`);
  }
  if (header.headerMessageId) {
    lines.push(`Message-ID: ${header.headerMessageId}`);
  }
  return lines.join("\n");
}

async function extractBody(messageId) {
  try {
    if (browser.messages.listInlineTextParts) {
      const parts = await browser.messages.listInlineTextParts(messageId);
      const plain = parts.find((p) =>
        String(p.contentType || "").toLowerCase().startsWith("text/plain"),
      );
      if (plain?.content?.trim()) {
        return stripQuotedTail(plain.content.trim());
      }
      const html = parts.find((p) =>
        String(p.contentType || "").toLowerCase().startsWith("text/html"),
      );
      if (html?.content) {
        const converted = await htmlToPlain(html.content);
        return stripQuotedTail(converted);
      }
    }
  } catch (_) {
    /* fall through to getFull */
  }
  try {
    const full = await browser.messages.getFull(messageId);
    const { plain, html } = walkParts(full);
    if (plain.trim()) return stripQuotedTail(plain.trim());
    if (html.trim()) return stripQuotedTail(await htmlToPlain(html));
  } catch (_) {
    /* empty body */
  }
  return "(no body)";
}

function walkParts(part, acc = { plain: "", html: "" }) {
  if (!part) return acc;
  const ct = String(part.contentType || "").toLowerCase();
  const body = typeof part.body === "string" ? part.body : "";
  if (ct.startsWith("text/plain") && body && !acc.plain) acc.plain = body;
  if (ct.startsWith("text/html") && body && !acc.html) acc.html = body;
  for (const child of part.parts || []) walkParts(child, acc);
  return acc;
}

async function htmlToPlain(html) {
  if (browser.messengerUtilities?.convertToPlainText) {
    try {
      return String(
        await browser.messengerUtilities.convertToPlainText(html),
      ).trim();
    } catch (_) {
      /* fall through */
    }
  }
  return String(html || "")
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/p>/gi, "\n\n")
    .replace(/<\/div>/gi, "\n")
    .replace(/<\/tr>/gi, "\n")
    .replace(/<\/li>/gi, "\n")
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** Drop trailing quoted reply blocks (best-effort). */
function stripQuotedTail(text) {
  const lines = String(text || "").split(/\r?\n/);
  let cut = lines.length;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^On .+ wrote:\s*$/i.test(line) || /^-+ ?Original Message ?-+$/i.test(line)) {
      cut = i;
      break;
    }
  }
  let body = lines.slice(0, cut);
  while (body.length && /^>/.test(body[body.length - 1])) body.pop();
  while (body.length && !body[body.length - 1].trim()) body.pop();
  return body.join("\n").trim() || String(text || "").trim();
}

function truncateBody(text) {
  const s = String(text || "");
  if (s.length <= BODY_CHAR_CAP) return s;
  return `${s.slice(0, BODY_CHAR_CAP)}\n\n…[truncated at ${BODY_CHAR_CAP} characters]`;
}

function newId() {
  if (crypto?.randomUUID) return crypto.randomUUID().replace(/-/g, "").slice(0, 16);
  return `s${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

async function enqueueSnippet(partial) {
  const snippet = {
    id: newId(),
    kind: partial.kind,
    text: partial.text || "",
    data_base64: partial.data_base64 || "",
    media_type: partial.media_type || "",
    source_url: partial.source_url || "",
    page_title: partial.page_title || "",
    captured_at: new Date().toISOString(),
    saved: false,
  };
  const { pendingSnippets = [] } = await browser.storage.local.get("pendingSnippets");
  pendingSnippets.push(snippet);
  await browser.storage.local.set({
    pendingSnippets,
    lastPageTitle: partial.page_title || "",
    lastPageUrl: partial.source_url || "",
  });
}

async function pushPending(payload) {
  await browser.storage.local.set({ pendingNotice: payload });
}

async function fileToBase64(file) {
  const buf = await file.arrayBuffer();
  const bytes = new Uint8Array(buf);
  const chunk = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}
