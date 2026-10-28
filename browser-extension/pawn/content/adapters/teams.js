/** Teams web adapter: message roots, ids, deep links, soft-match fallbacks. */

(function () {
  if (window.__pawnTeams) return;

  const CHAT_ID_RE = /(19:[^/?#\s]+@(?:thread\.v2|thread\.tacv2|unq\.gbl\.spaces))/i;
  const MESSAGE_LINK_RE =
    /\/l\/message\/(19:[^/]+)\/(\d+)(?:\?|$)/i;
  const MESSAGE_ID_ATTRS = [
    "data-mid",
    "data-message-id",
    "data-item-id",
    "data-tid-message-id",
  ];

  function norm(s) {
    return String(s || "").replace(/\s+/g, " ").trim();
  }

  function isTeamsHost(hostname) {
    const h = String(hostname || "").toLowerCase();
    return (
      /(^|\.)teams\.(microsoft|live)\.com$/.test(h) ||
      /(^|\.)teams\.cloud\.microsoft$/.test(h)
    );
  }

  function isTeamsPage() {
    try {
      return isTeamsHost(location.hostname);
    } catch (_) {
      return false;
    }
  }

  function pageKeyFromHref(href) {
    const s = String(href || "");
    const link = s.match(MESSAGE_LINK_RE);
    if (link) return decodeURIComponent(link[1]);
    const chat = s.match(CHAT_ID_RE);
    if (chat) return chat[1];
    try {
      const u = new URL(s, location.origin);
      for (const key of ["chatId", "conversationId", "threadId"]) {
        const v = u.searchParams.get(key);
        if (v && CHAT_ID_RE.test(v)) return v.match(CHAT_ID_RE)[1];
      }
      const pathChat = u.pathname.match(CHAT_ID_RE);
      if (pathChat) return pathChat[1];
      if (u.hash) {
        const hashChat = decodeURIComponent(u.hash).match(CHAT_ID_RE);
        if (hashChat) return hashChat[1];
      }
    } catch (_) {
      /* ignore */
    }
    return "";
  }

  function currentPageKey() {
    return (
      pageKeyFromHref(location.href) ||
      findChatIdInDom() ||
      ""
    );
  }

  function findChatIdInDom() {
    const anchors = document.querySelectorAll('a[href*="/l/message/"], a[href*="/l/chat/"]');
    for (const a of anchors) {
      const pk = pageKeyFromHref(a.href || "");
      if (pk) return pk;
    }
    const withAttr = document.querySelector(
      "[data-conversation-id], [data-chat-id], [data-thread-id]",
    );
    if (withAttr) {
      for (const name of ["data-conversation-id", "data-chat-id", "data-thread-id"]) {
        const v = withAttr.getAttribute(name) || "";
        const m = v.match(CHAT_ID_RE);
        if (m) return m[1];
      }
    }
    return "";
  }

  function looksLikeMessageRoot(el) {
    if (!el || el.nodeType !== 1) return false;
    const tid = (el.getAttribute("data-tid") || "").toLowerCase();
    if (
      /message|chat-pane-message|message-wrapper|message-body|message-list-item/.test(
        tid,
      )
    ) {
      return true;
    }
    if (el.getAttribute("data-mid") || el.getAttribute("data-message-id")) return true;
    const role = (el.getAttribute("role") || "").toLowerCase();
    if (role === "listitem" || role === "article") {
      if (el.querySelector?.("[data-tid*='message'], [data-mid], time, [datetime]")) {
        return true;
      }
    }
    return false;
  }

  function findMessageRoot(fromEl) {
    let el = fromEl;
    if (el && el.nodeType === 3) el = el.parentElement;
    let best = null;
    for (let cur = el; cur && cur !== document.body; cur = cur.parentElement) {
      if (looksLikeMessageRoot(cur)) best = cur;
    }
    if (best) return best;
    // Soft: climb to a block that has an author-ish header + body text.
    for (let cur = el; cur && cur !== document.body; cur = cur.parentElement) {
      const text = norm(cur.innerText || "").slice(0, 500);
      if (text.length < 8) continue;
      if (cur.querySelector?.("img[alt], [data-tid*='avatar'], time, [datetime]")) {
        return cur;
      }
    }
    return el || null;
  }

  function readMessageId(root) {
    if (!root) return "";
    for (const name of MESSAGE_ID_ATTRS) {
      const v = root.getAttribute?.(name);
      if (v && /^\d{10,}$/.test(String(v).trim())) return String(v).trim();
    }
    const nested = root.querySelector?.(
      "[data-mid], [data-message-id], [data-item-id], [id*='message']",
    );
    if (nested) {
      for (const name of MESSAGE_ID_ATTRS) {
        const v = nested.getAttribute?.(name);
        if (v && /^\d{10,}$/.test(String(v).trim())) return String(v).trim();
      }
      const id = nested.id || "";
      const m = id.match(/(\d{10,})/);
      if (m) return m[1];
    }
    const link = root.querySelector?.('a[href*="/l/message/"]');
    if (link?.href) {
      const m = link.href.match(MESSAGE_LINK_RE);
      if (m) return m[2];
    }
    // data-tid / aria sometimes embeds createdTime
    const blob = `${root.getAttribute?.("data-tid") || ""} ${root.id || ""}`;
    const mid = blob.match(/\b(1\d{12,})\b/);
    return mid ? mid[1] : "";
  }

  function readAuthor(root) {
    if (!root) return "";
    const selectors = [
      "[data-tid*='message-author']",
      "[data-tid*='author']",
      "[data-tid*='display-name']",
      "[data-tid*='sender']",
      "strong",
      "h1",
      "h2",
      "h3",
    ];
    for (const sel of selectors) {
      const node = root.querySelector?.(sel);
      const t = norm(node?.textContent || "");
      if (
        t &&
        t.length < 80 &&
        !/^(am|pm|\d+:\d+)/i.test(t) &&
        !/^(you|me|edited|like|reply)$/i.test(t)
      ) {
        return t;
      }
    }
    return "";
  }

  function readApproxTime(root) {
    if (!root) return "";
    const time = root.querySelector?.("time, [datetime], [data-tid*='timestamp'], [data-tid*='time']");
    if (time) {
      const label = norm(time.getAttribute("aria-label") || time.textContent || "");
      if (label) return label.slice(0, 40);
      const dt = time.getAttribute("datetime");
      if (dt) return dt;
    }
    const m = norm(root.innerText || "").match(/\b(\d{1,2}:\d{2}\s*(?:AM|PM)?)\b/i);
    return m ? m[1] : "";
  }

  function readBodyText(root) {
    if (!root) return "";
    const body =
      root.querySelector?.(
        "[data-tid*='message-body'], [data-tid*='message-content'], [data-tid*='message-text']",
      ) || root;
    return norm(body.innerText || body.textContent || "").slice(0, 400);
  }

  function buildDeepLink(pageKey, messageId) {
    if (!pageKey || !messageId) return "";
    const ctx = encodeURIComponent('{"contextType":"chat"}');
    return `https://teams.microsoft.com/l/message/${pageKey}/${messageId}?context=${ctx}`;
  }

  function readMessageIds(root) {
    const page_key = currentPageKey() || pageKeyFromHref(root?.querySelector?.("a[href*='/l/message/']")?.href || "");
    const message_id = readMessageId(root);
    const author = readAuthor(root);
    const approx_time = readApproxTime(root);
    const text_quote = readBodyText(root).slice(0, 160);
    const deep_link = buildDeepLink(page_key, message_id);
    const fallback = {};
    if (author) fallback.author = author;
    if (text_quote) fallback.text_quote = text_quote;
    if (approx_time) fallback.approx_time = approx_time;
    const anchor = {
      site: "teams",
      page_key: page_key || undefined,
      message_id: message_id || undefined,
      deep_link: deep_link || undefined,
      fallback: Object.keys(fallback).length ? fallback : undefined,
    };
    if (!anchor.page_key && !anchor.message_id && !anchor.fallback) return null;
    return anchor;
  }

  function listVisibleMessages() {
    const roots = new Set();
    const candidates = document.querySelectorAll(
      [
        "[data-tid*='chat-pane-message']",
        "[data-tid*='message-wrapper']",
        "[data-tid*='message-body']",
        "[data-mid]",
        "[data-message-id]",
        "[role='listitem']",
        "[role='article']",
      ].join(","),
    );
    for (const node of candidates) {
      const root = findMessageRoot(node);
      if (!root) continue;
      const rect = root.getBoundingClientRect?.();
      if (!rect || rect.height < 8 || rect.width < 8) continue;
      if (rect.bottom < 0 || rect.top > window.innerHeight) continue;
      roots.add(root);
    }
    return [...roots];
  }

  function captureFromSelection() {
    const sel = window.getSelection?.();
    const text = norm(sel?.toString() || "");
    if (!text) {
      return { ok: false, error: "No text selected" };
    }
    let node = sel.anchorNode || sel.focusNode;
    if (node && node.nodeType === 3) node = node.parentElement;
    const root = findMessageRoot(node);
    const anchor = root ? readMessageIds(root) : null;
    if (anchor?.fallback && text) {
      anchor.fallback = { ...anchor.fallback, text_quote: text.slice(0, 160) };
    } else if (anchor && text && !anchor.fallback) {
      anchor.fallback = { text_quote: text.slice(0, 160) };
    }
    return {
      ok: true,
      text,
      url: (anchor && anchor.deep_link) || location.href,
      title: document.title || "",
      anchor: anchor || undefined,
      page_key: currentPageKey(),
    };
  }

  function captureFromPoint(clientX, clientY) {
    const x = Number(clientX);
    const y = Number(clientY);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return { ok: false, error: "Invalid point" };
    }
    const el = document.elementFromPoint(x, y);
    const root = findMessageRoot(el);
    if (!root) return { ok: true, anchor: undefined, page_key: currentPageKey() };
    const anchor = readMessageIds(root);
    return {
      ok: true,
      anchor: anchor || undefined,
      page_key: currentPageKey(),
      url: (anchor && anchor.deep_link) || location.href,
    };
  }

  function matchMessage(root, anchor) {
    if (!root || !anchor) return false;
    const mid = String(anchor.message_id || "").trim();
    if (mid) {
      const got = readMessageId(root);
      if (got && got === mid) return true;
    }
    const fb = anchor.fallback || {};
    const quote = norm(fb.text_quote || "");
    const author = norm(fb.author || "");
    if (!quote && !author) return false;
    const body = readBodyText(root);
    const who = readAuthor(root);
    if (quote && body.includes(quote.slice(0, Math.min(48, quote.length)))) {
      if (!author || !who || who.toLowerCase() === author.toLowerCase()) return true;
    }
    if (author && who && who.toLowerCase() === author.toLowerCase() && quote) {
      const head = quote.slice(0, 24);
      if (head && body.includes(head)) return true;
    }
    return false;
  }

  const api = {
    isTeamsPage,
    isTeamsHost,
    currentPageKey,
    pageKeyFromHref,
    findMessageRoot,
    readMessageIds,
    listVisibleMessages,
    captureFromSelection,
    captureFromPoint,
    matchMessage,
    buildDeepLink,
  };

  window.__pawnTeams = api;

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg || !String(msg.type || "").startsWith("pawn-teams-")) return false;
    try {
      if (msg.type === "pawn-teams-page-key") {
        sendResponse({ ok: true, page_key: currentPageKey(), href: location.href });
        return false;
      }
      if (msg.type === "pawn-teams-capture-selection") {
        sendResponse(captureFromSelection());
        return false;
      }
      if (msg.type === "pawn-teams-capture-point") {
        sendResponse(captureFromPoint(msg.clientX, msg.clientY));
        return false;
      }
      if (msg.type === "pawn-teams-list-messages") {
        const items = listVisibleMessages().map((root) => ({
          anchor: readMessageIds(root),
        }));
        sendResponse({ ok: true, page_key: currentPageKey(), items });
        return false;
      }
    } catch (err) {
      sendResponse({ ok: false, error: String(err?.message || err) });
      return false;
    }
    return false;
  });
})();
