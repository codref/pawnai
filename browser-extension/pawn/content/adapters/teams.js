/** Teams web adapter: message roots, ids, deep links, soft-match fallbacks. */

(function () {
  if (window.__pawnTeams) return;

  const CHAT_ID_RE =
    /(19:[A-Za-z0-9._\-:@]+@(?:thread\.v2|thread\.tacv2|unq\.gbl\.spaces|thread\.skype))/i;
  const MESSAGE_LINK_RE = /\/l\/message\/([^/?#]+)\/(\d+)(?:\?|$)/i;
  const MESSAGE_ID_ATTRS = [
    "data-mid",
    "data-message-id",
    "data-item-id",
    "data-tid-message-id",
  ];

  function norm(s) {
    return String(s || "").replace(/\s+/g, " ").trim();
  }

  function decodeMaybe(s) {
    try {
      return decodeURIComponent(String(s || ""));
    } catch (_) {
      return String(s || "");
    }
  }

  function isTeamsHost(hostname) {
    const h = String(hostname || "").toLowerCase();
    if (!h) return false;
    if (/(^|\.)teams\.(microsoft|live)\.com$/.test(h)) return true;
    if (/(^|\.)teams\.microsoft\.us$/.test(h)) return true;
    if (h === "teams.cloud.microsoft" || h.endsWith(".teams.cloud.microsoft")) return true;
    if (h.endsWith(".cloud.microsoft") && /(^|\.)teams\./.test(h)) return true;
    return false;
  }

  /** querySelectorAll that also walks open shadow roots (Teams Fluent UI). */
  function deepQueryAll(root, selector, out = []) {
    if (!root) return out;
    try {
      root.querySelectorAll?.(selector)?.forEach((el) => out.push(el));
    } catch (_) {
      /* invalid selector in this root */
    }
    const walk = root.querySelectorAll ? root.querySelectorAll("*") : [];
    for (const el of walk) {
      if (el.shadowRoot) deepQueryAll(el.shadowRoot, selector, out);
    }
    return out;
  }

  function isTeamsPage() {
    try {
      return isTeamsHost(location.hostname);
    } catch (_) {
      return false;
    }
  }

  function pageKeyFromHref(href) {
    const s = decodeMaybe(href || "");
    const link = s.match(MESSAGE_LINK_RE);
    if (link) return decodeMaybe(link[1]);
    const chat = s.match(CHAT_ID_RE);
    if (chat) return chat[1];
    try {
      const u = new URL(s, location.origin);
      for (const key of ["chatId", "conversationId", "threadId", "threadId2"]) {
        const v = u.searchParams.get(key);
        if (!v) continue;
        const m = decodeMaybe(v).match(CHAT_ID_RE);
        if (m) return m[1];
      }
      const pathChat = decodeMaybe(u.pathname + u.hash).match(CHAT_ID_RE);
      if (pathChat) return pathChat[1];
    } catch (_) {
      /* ignore */
    }
    return "";
  }

  function findChatIdInDom() {
    const anchors = document.querySelectorAll(
      'a[href*="/l/message/"], a[href*="/l/chat/"], a[href*="thread.v2"]',
    );
    for (const a of anchors) {
      const pk = pageKeyFromHref(a.href || "");
      if (pk) return pk;
    }
    for (const el of document.querySelectorAll(
      "[data-conversation-id], [data-chat-id], [data-thread-id], [data-tid]",
    )) {
      for (const name of [
        "data-conversation-id",
        "data-chat-id",
        "data-thread-id",
        "data-tid",
      ]) {
        const v = el.getAttribute(name) || "";
        const m = v.match(CHAT_ID_RE);
        if (m) return m[1];
      }
    }
    // New Teams often keeps the id only in the message list markup.
    const pane =
      document.querySelector(
        "[data-tid*='message-pane'], [data-tid*='chat-pane-list'], [data-tid*='chat-pane']",
      ) || document.body;
    const blob = String(pane?.innerHTML || "").slice(0, 400000);
    const m = blob.match(CHAT_ID_RE);
    return m ? m[1] : "";
  }

  function currentPageKey() {
    return pageKeyFromHref(location.href) || findChatIdInDom() || "";
  }

  function isPaneChrome(el) {
    if (!el || el.nodeType !== 1) return true;
    const tid = (el.getAttribute("data-tid") || "").toLowerCase();
    // "message-pane-layout" must NOT count as a chat bubble (regex /message/ matched it).
    if (
      /message-pane|chat-pane(?!-message)|titlebar|app-bar|sidebar|composer|ck-editor|toolbar|layout/.test(
        tid,
      )
    ) {
      return true;
    }
    const role = (el.getAttribute("role") || "").toLowerCase();
    if (role === "main" || role === "navigation" || role === "banner") return true;
    return false;
  }

  function isPlausibleBubble(el) {
    if (!el || isPaneChrome(el)) return false;
    const r = el.getBoundingClientRect?.();
    if (!r || r.height < 28 || r.width < 80) return false;
    // Whole chat pane / virtualizer rows that fill the viewport are not bubbles.
    if (r.height > Math.min(420, window.innerHeight * 0.55)) return false;
    if (r.width > window.innerWidth * 0.92) return false;
    return true;
  }

  function looksLikeMessageRoot(el) {
    if (!el || el.nodeType !== 1 || isPaneChrome(el) || !isPlausibleBubble(el)) {
      return false;
    }
    const tid = (el.getAttribute("data-tid") || "").toLowerCase();
    if (
      /chat-pane-message|message-wrapper|message-body|message-list-item|chat-message|chat-item|message-content/.test(
        tid,
      )
    ) {
      return true;
    }
    // Avoid bare /message/ — it matches message-pane-layout.
    if (/(^|[^a-z])message([^a-z]|$)/.test(tid) && !/pane|layout|list$/.test(tid)) {
      return true;
    }
    if (el.getAttribute("data-mid") || el.getAttribute("data-message-id")) return true;
    const role = (el.getAttribute("role") || "").toLowerCase();
    if (role === "listitem" || role === "article") {
      if (
        el.querySelector?.(
          "[data-tid*='message-body'], [data-tid*='message-content'], [data-mid], time, [datetime], [data-tid*='avatar']",
        )
      ) {
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
    return best;
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
      const m = decodeMaybe(link.href).match(MESSAGE_LINK_RE);
      if (m) return m[2];
    }
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
      "[data-tid*='participant']",
      "strong",
    ];
    for (const sel of selectors) {
      const node = root.querySelector?.(sel);
      const t = norm(node?.textContent || "");
      if (
        t &&
        t.length < 80 &&
        !/^(am|pm|\d+:\d+)/i.test(t) &&
        !/^(you|me|edited|like|reply|react)$/i.test(t)
      ) {
        return t;
      }
    }
    return "";
  }

  function readApproxTime(root) {
    if (!root) return "";
    const time = root.querySelector?.(
      "time, [datetime], [data-tid*='timestamp'], [data-tid*='time']",
    );
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
    const page_key =
      currentPageKey() ||
      pageKeyFromHref(root?.querySelector?.("a[href*='/l/message/']")?.href || "");
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
    // Keep selectors specific — avoid [data-tid*='message'] (matches message-pane-layout).
    const selector = [
      "[data-tid*='chat-pane-message']",
      "[data-tid*='message-wrapper']",
      "[data-tid*='message-body']",
      "[data-tid*='message-content']",
      "[data-tid*='chat-message']",
      "[data-mid]",
      "[data-message-id]",
      "[role='listitem']",
      "[role='article']",
    ].join(",");
    const candidates = deepQueryAll(document, selector);
    for (const node of candidates) {
      if (isPaneChrome(node)) continue;
      const root = findMessageRoot(node) || (looksLikeMessageRoot(node) ? node : null);
      if (!root || !isPlausibleBubble(root)) continue;
      const rect = root.getBoundingClientRect?.();
      if (!rect) continue;
      if (rect.bottom < 0 || rect.top > window.innerHeight) continue;
      roots.add(root);
    }
    return [...roots];
  }

  function messageRootAtPoint(clientX, clientY) {
    const x = Number(clientX);
    const y = Number(clientY);
    const stack =
      typeof document.elementsFromPoint === "function"
        ? document.elementsFromPoint(x, y)
        : [document.elementFromPoint(x, y)].filter(Boolean);
    for (const el of stack) {
      if (el?.id === "pawn-region-root" || el?.closest?.("#pawn-region-root")) continue;
      const root = findMessageRoot(el);
      if (root) return root;
    }
    // Containment, then nearest-by-center (region crops often miss the bubble edge).
    let best = null;
    let bestScore = Infinity;
    for (const root of listVisibleMessages()) {
      const r = root.getBoundingClientRect();
      const cx = (r.left + r.right) / 2;
      const cy = (r.top + r.bottom) / 2;
      const contained =
        x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
      const dist = Math.hypot(x - cx, y - cy);
      const score = contained ? dist * 0.25 : dist;
      if (score < bestScore) {
        bestScore = score;
        best = root;
      }
    }
    // Reject nearest matches that are clearly far from the click (other pane).
    if (best && bestScore > 240) return null;
    return best;
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
    } else if (!anchor && text) {
      // Soft page-level fallback so Save still registers something paintable.
      const page_key = currentPageKey();
      return {
        ok: true,
        text,
        url: location.href,
        title: document.title || "",
        anchor: {
          site: "teams",
          page_key: page_key || undefined,
          fallback: { text_quote: text.slice(0, 160) },
        },
        page_key,
      };
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
    const root = messageRootAtPoint(x, y);
    const page_key = currentPageKey();
    if (!root) {
      return {
        ok: true,
        anchor: page_key
          ? { site: "teams", page_key, fallback: { text_quote: "region capture" } }
          : undefined,
        page_key,
        url: location.href,
        matched: false,
      };
    }
    const anchor = readMessageIds(root);
    return {
      ok: true,
      anchor: anchor || undefined,
      page_key,
      url: (anchor && anchor.deep_link) || location.href,
      matched: Boolean(anchor),
    };
  }

  function foldText(s) {
    return norm(s)
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s]/gu, " ")
      .replace(/\s+/g, " ")
      .trim();
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
    const when = norm(fb.approx_time || "");
    if (!quote && !author) return false;
    if (quote === "region capture" || quote === "capture") return false;
    const body = foldText(readBodyText(root));
    const who = foldText(readAuthor(root));
    const time = foldText(readApproxTime(root));
    const q = foldText(quote);
    const a = foldText(author);
    if (q && q.length >= 8) {
      const needle = q.slice(0, Math.min(48, q.length));
      if (body.includes(needle)) {
        if (!a || !who || who === a) return true;
      }
    }
    if (a && who && who === a) {
      if (q) {
        const head = q.slice(0, 24);
        if (head && body.includes(head)) return true;
      }
      const w = foldText(when);
      if (w && time && time.includes(w.slice(0, 5))) return true;
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
    messageRootAtPoint,
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
