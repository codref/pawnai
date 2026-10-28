/** Re-paint Pawn vault badges on Teams message bubbles. */

(function () {
  if (window.__pawnOverlaysReady) return;
  window.__pawnOverlaysReady = true;

  const ATTR = "data-pawn-overlay";
  const CHAT_BADGE_ID = "pawn-overlay-chat-badge";
  const TIP_ID = "pawn-overlay-tip";
  const POS_KEY = "pawnOverlayChatPos";
  /** Compact pawn icon — SVG so Teams font CSS cannot turn it into a giant glyph. */
  const PAWN_SVG =
    '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false">' +
    '<path fill="currentColor" d="M12 2a3 3 0 0 0-1.5 5.6L9.6 10H8a1 1 0 1 0 0 2h8a1 1 0 1 0 0-2h-1.6l-.9-2.4A3 3 0 0 0 12 2zm-4.5 12h9a1.5 1.5 0 0 1 1.5 1.5V16c0 2.2-2.5 4-6 4s-6-1.8-6-4v-.5A1.5 1.5 0 0 1 7.5 14z"/>' +
    "</svg>";
  const CHAT_SHADOW_CSS = `
:host {
  all: initial;
  position: fixed !important;
  z-index: 2147483640 !important;
  width: 40px !important;
  height: 40px !important;
  display: block !important;
  pointer-events: auto !important;
}
button.chip {
  all: unset;
  box-sizing: border-box;
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border: 2px solid #4a9e8e;
  border-radius: 50%;
  background: #14352f;
  color: #e8f7f3;
  cursor: grab;
  box-shadow: 0 3px 14px rgba(0,0,0,.45);
  touch-action: none;
  user-select: none;
}
button.chip:hover, button.chip:focus-visible {
  background: #1f4f46;
  outline: 2px solid #6bc4b4;
  outline-offset: 2px;
}
button.chip.is-dragging { cursor: grabbing; }
button.chip svg { width: 20px; height: 20px; display: block; fill: currentColor; pointer-events: none; }
span.count {
  position: absolute;
  top: -5px;
  right: -5px;
  box-sizing: border-box;
  min-width: 18px;
  height: 18px;
  padding: 0 4px;
  border-radius: 999px;
  background: #c45c26;
  color: #fff;
  font: 700 10px/18px system-ui, -apple-system, "Segoe UI", sans-serif;
  text-align: center;
  pointer-events: none;
  box-shadow: 0 1px 2px rgba(0,0,0,.35);
}
`;
  let anchors = [];
  let pageKey = "";
  let paintTimer = 0;
  let observer = null;
  let keepAlive = 0;
  let tipTimer = 0;
  let dragState = null;
  let userDocked = false;

  function teams() {
    return window.__pawnTeams || null;
  }

  function schedulePaint(delayMs) {
    if (paintTimer) clearTimeout(paintTimer);
    paintTimer = setTimeout(() => {
      paintTimer = 0;
      void paint();
    }, delayMs == null ? 120 : delayMs);
  }

  async function refreshAnchors() {
    const t = teams();
    if (!t?.isTeamsPage?.()) return;
    const nextKey = t.currentPageKey() || "";
    if (nextKey) pageKey = nextKey;
    try {
      const resp = await chrome.runtime.sendMessage({
        type: "pawn-anchors-get",
        page_key: pageKey,
        site: "teams",
      });
      const rows = Array.isArray(resp?.data)
        ? resp.data
        : Array.isArray(resp?.cached)
          ? resp.cached
          : [];
      // Never wipe a good cache on a transient empty response (Teams SPA churn).
      if (rows.length) {
        anchors = rows;
      } else if (!anchors.length && Array.isArray(resp?.cached) && resp.cached.length) {
        anchors = resp.cached;
      }
    } catch (_) {
      /* background unavailable — keep prior anchors */
    }
  }

  function ensureHost(root) {
    if (!root) return;
    if (!root.classList.contains("pawn-overlay-host")) {
      root.classList.add("pawn-overlay-host");
    }
  }

  function liveBadgeIds() {
    const ids = new Set();
    document.querySelectorAll(`[${ATTR}]`).forEach((el) => {
      if (!el.isConnected) {
        el.remove();
        return;
      }
      const id = el.getAttribute(ATTR);
      if (id) ids.add(id);
    });
    return ids;
  }

  function removeStale(validIds) {
    document.querySelectorAll(`[${ATTR}]`).forEach((el) => {
      const id = el.getAttribute(ATTR);
      const host = el.parentElement;
      const hostTid = (host?.getAttribute?.("data-tid") || "").toLowerCase();
      const badHost =
        /message-pane|chat-pane(?!-message)|layout|titlebar|composer/.test(hostTid) ||
        (() => {
          const r = host?.getBoundingClientRect?.();
          return r && r.height > Math.min(420, window.innerHeight * 0.55);
        })();
      if (!validIds.has(id) || !el.isConnected || badHost) el.remove();
    });
  }

  function findMount(root) {
    // Mount on the message card itself (not timestamp/reactions) so absolute
    // top-right lands on the bubble, not the reaction strip.
    return root;
  }

  function chatPaneRect() {
    const pane =
      document.querySelector(
        "[data-tid*='message-pane'], [data-tid*='chat-pane-list'], [data-tid*='chat-pane'], [role='main']",
      ) || document.querySelector("main");
    const r = pane?.getBoundingClientRect?.();
    if (r && r.width > 160 && r.height > 160) return r;
    return null;
  }

  /** Default dock: lower-right of the chat pane (above compose), not the window chrome. */
  function defaultChatDock(size) {
    const s = size || 40;
    const pad = 16;
    const pane = chatPaneRect();
    if (pane) {
      return {
        left: Math.max(pad, pane.right - s - pad),
        top: Math.max(pad, pane.bottom - s - 96),
      };
    }
    return {
      left: Math.max(pad, window.innerWidth - s - 28),
      top: Math.max(pad, window.innerHeight - s - 120),
    };
  }

  function isBadDock(pos, size) {
    if (!pos || !Number.isFinite(pos.left) || !Number.isFinite(pos.top)) return true;
    const s = size || 40;
    // Old default sat in the Teams header; treat that band as invalid unless userDragged.
    if (!pos.userDragged && pos.top < 88) return true;
    if (pos.left < 0 || pos.top < 0) return true;
    if (pos.left > window.innerWidth - s / 2) return true;
    if (pos.top > window.innerHeight - s / 2) return true;
    return false;
  }

  function openVault(path, title) {
    void chrome.runtime.sendMessage({
      type: "pawn-overlay-open",
      path: path || "",
      title: title || "",
    });
  }

  function hideTip() {
    if (tipTimer) {
      clearTimeout(tipTimer);
      tipTimer = 0;
    }
    document.getElementById(TIP_ID)?.remove();
  }

  function scheduleHideTip(delayMs) {
    if (tipTimer) clearTimeout(tipTimer);
    tipTimer = setTimeout(() => {
      tipTimer = 0;
      hideTip();
    }, delayMs == null ? 220 : delayMs);
  }

  function formatCaptureWhen(iso) {
    const raw = String(iso || "").trim();
    if (!raw) return "";
    const d = new Date(raw);
    if (Number.isNaN(d.getTime())) return raw.slice(0, 16);
    try {
      return new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(d);
    } catch (_) {
      return d.toLocaleString();
    }
  }

  function showTip(anchorEl, rows) {
    if (tipTimer) {
      clearTimeout(tipTimer);
      tipTimer = 0;
    }
    document.getElementById(TIP_ID)?.remove();
    if (!anchorEl || !rows?.length) return;

    const sorted = [...rows].sort((a, b) => {
      const ta = Date.parse(a.captured_at || "") || 0;
      const tb = Date.parse(b.captured_at || "") || 0;
      return tb - ta;
    });

    const tip = document.createElement("div");
    tip.id = TIP_ID;
    tip.className = "pawn-overlay-tip";
    tip.setAttribute("role", "menu");
    tip.setAttribute("aria-label", "Pawn captures");

    const strong = document.createElement("strong");
    strong.textContent =
      sorted.length === 1
        ? "Pawn capture"
        : `${sorted.length} Pawn captures`;
    tip.appendChild(strong);

    const list = document.createElement("div");
    list.className = "pawn-overlay-tip-list";
    for (const row of sorted.slice(0, 12)) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pawn-overlay-tip-item";
      btn.setAttribute("role", "menuitem");
      const when = formatCaptureWhen(row.captured_at);
      const title = row.title || row.path || "capture";
      const dateEl = document.createElement("span");
      dateEl.className = "pawn-overlay-tip-date";
      dateEl.textContent = when || "Date unknown";
      const titleEl = document.createElement("span");
      titleEl.className = "pawn-overlay-tip-title";
      titleEl.textContent = title;
      btn.append(dateEl, titleEl);
      btn.title = when ? `${title}\n${when}` : title;
      btn.addEventListener(
        "click",
        (ev) => {
          ev.preventDefault();
          ev.stopPropagation();
          hideTip();
          openVault(row.path, title);
        },
        true,
      );
      list.appendChild(btn);
    }
    if (sorted.length > 12) {
      const more = document.createElement("div");
      more.className = "pawn-overlay-tip-hint";
      more.textContent = `…and ${sorted.length - 12} more in the vault`;
      list.appendChild(more);
    }
    tip.appendChild(list);

    const hint = document.createElement("div");
    hint.className = "pawn-overlay-tip-hint";
    hint.textContent =
      sorted.length > 1
        ? "Click a row to open · drag the chip to move"
        : "Click to open · drag the chip to move";
    tip.appendChild(hint);

    tip.addEventListener("mouseenter", () => {
      if (tipTimer) {
        clearTimeout(tipTimer);
        tipTimer = 0;
      }
    });
    tip.addEventListener("mouseleave", () => scheduleHideTip(180));

    document.documentElement.appendChild(tip);
    const r = anchorEl.getBoundingClientRect();
    const tw = tip.offsetWidth || 260;
    const th = tip.offsetHeight || 100;
    let left = r.left + r.width / 2 - tw / 2;
    let top = r.top - th - 10;
    if (top < 8) top = r.bottom + 10;
    left = Math.max(8, Math.min(left, window.innerWidth - tw - 8));
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }

  function isSafeBadgeHost(root) {
    if (!root?.isConnected) return false;
    // Refuse pane-level hosts (was mounting on message-pane-layout).
    const tid = (root.getAttribute?.("data-tid") || "").toLowerCase();
    if (/message-pane|chat-pane(?!-message)|layout|titlebar|composer/.test(tid)) {
      return false;
    }
    const r = root.getBoundingClientRect?.();
    if (!r) return false;
    if (r.height > Math.min(420, window.innerHeight * 0.55)) return false;
    if (r.width > window.innerWidth * 0.92) return false;
    return true;
  }

  function paintOne(root, row) {
    const sid = String(row.snippet_id || "").trim();
    if (!sid || !root || !root.isConnected) return false;
    if (!isSafeBadgeHost(root)) return false;
    const sel =
      typeof CSS !== "undefined" && CSS.escape ? CSS.escape(sid) : sid;
    const existing = root.querySelector?.(`[${ATTR}="${sel}"]`);
    if (existing?.isConnected) return true;
    existing?.remove();
    ensureHost(root);
    const mount = findMount(root);
    if (!mount?.isConnected || !isSafeBadgeHost(mount)) return false;
    ensureHost(mount);
    const a = document.createElement("a");
    a.className = "pawn-overlay-badge";
    a.setAttribute(ATTR, sid);
    a.href = "#";
    a.setAttribute(
      "aria-label",
      row.title ? `Open in Obsidian: ${row.title}` : "Open in Obsidian",
    );
    a.innerHTML = PAWN_SVG;
    Object.assign(a.style, {
      position: "absolute",
      top: "8px",
      right: "10px",
      width: "24px",
      height: "24px",
      zIndex: "2147483000",
    });
    if (!row.message_id) a.setAttribute("data-pawn-soft", "1");
    a.addEventListener(
      "mouseenter",
      () => {
        if (tipTimer) clearTimeout(tipTimer);
        tipTimer = setTimeout(() => {
          tipTimer = 0;
          showTip(a, [row]);
        }, 220);
      },
      true,
    );
    a.addEventListener("mouseleave", () => scheduleHideTip(220), true);
    a.addEventListener(
      "click",
      (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        hideTip();
        openVault(row.path, row.title || "");
      },
      true,
    );
    mount.appendChild(a);
    return a.isConnected;
  }

  function stableBadgeHost() {
    // documentElement survives Teams SPA remounts; main/chat-pane do not.
    return document.documentElement;
  }

  async function loadChatPos() {
    try {
      const data = await chrome.storage.local.get(POS_KEY);
      const pos = data?.[POS_KEY];
      if (pos && !isBadDock(pos)) {
        userDocked = Boolean(pos.userDragged);
        return pos;
      }
      if (pos && isBadDock(pos)) {
        await chrome.storage.local.remove(POS_KEY);
      }
    } catch (_) {
      /* ignore */
    }
    return null;
  }

  async function saveChatPos(left, top) {
    try {
      userDocked = true;
      await chrome.storage.local.set({
        [POS_KEY]: { left, top, savedAt: Date.now(), userDragged: true },
      });
    } catch (_) {
      /* ignore */
    }
  }

  function applyChatPos(host, pos) {
    if (!host) return;
    const size = 40;
    const dock =
      pos && !isBadDock(pos, size) ? pos : defaultChatDock(size);
    host.style.setProperty("left", `${dock.left}px`, "important");
    host.style.setProperty("top", `${dock.top}px`, "important");
    host.style.setProperty("right", "auto", "important");
    host.style.setProperty("bottom", "auto", "important");
    host.style.setProperty("position", "fixed", "important");
    host.style.setProperty("width", "40px", "important");
    host.style.setProperty("height", "40px", "important");
    host.style.setProperty("z-index", "2147483640", "important");
  }

  function redockIfNeeded(host) {
    if (!host || userDocked || dragState) return;
    applyChatPos(host, null);
  }

  function chatButton(host) {
    return host?.shadowRoot?.querySelector("button.chip") || null;
  }

  function ensureChatHost() {
    let host = document.getElementById(CHAT_BADGE_ID);
    if (host?.isConnected && host.shadowRoot?.querySelector("button.chip")) {
      return host;
    }
    host?.remove();
    host = document.createElement("div");
    host.id = CHAT_BADGE_ID;
    host.setAttribute("data-pawn-chat-chip", "1");
    const shadow = host.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = CHAT_SHADOW_CSS;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.innerHTML = PAWN_SVG;
    shadow.append(style, btn);
    wireChatDrag(host, btn);
    stableBadgeHost().appendChild(host);
    void loadChatPos().then((pos) => applyChatPos(host, pos));
    return host;
  }

  function wireChatDrag(host, btn) {
    if (btn.__pawnDragWired) return;
    btn.__pawnDragWired = true;
    let moved = false;

    const onMove = (ev) => {
      if (!dragState) return;
      const dx = ev.clientX - dragState.startX;
      const dy = ev.clientY - dragState.startY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) moved = true;
      const size = 40;
      let left = dragState.originLeft + dx;
      let top = dragState.originTop + dy;
      left = Math.max(4, Math.min(left, window.innerWidth - size - 4));
      top = Math.max(4, Math.min(top, window.innerHeight - size - 4));
      applyChatPos(host, { left, top, userDragged: true });
      dragState.left = left;
      dragState.top = top;
      hideTip();
    };

    const onUp = () => {
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      btn.classList.remove("is-dragging");
      if (dragState && moved) {
        void saveChatPos(dragState.left, dragState.top);
      }
      btn.__pawnDidDrag = moved;
      dragState = null;
      moved = false;
    };

    btn.addEventListener("pointerdown", (ev) => {
      if (ev.button != null && ev.button !== 0) return;
      const r = host.getBoundingClientRect();
      dragState = {
        startX: ev.clientX,
        startY: ev.clientY,
        originLeft: r.left,
        originTop: r.top,
        left: r.left,
        top: r.top,
      };
      moved = false;
      btn.__pawnDidDrag = false;
      btn.classList.add("is-dragging");
      hideTip();
      try {
        btn.setPointerCapture?.(ev.pointerId);
      } catch (_) {
        /* ignore */
      }
      window.addEventListener("pointermove", onMove, true);
      window.addEventListener("pointerup", onUp, true);
      ev.preventDefault();
      ev.stopPropagation();
    });

    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      if (btn.__pawnDidDrag) {
        btn.__pawnDidDrag = false;
        return;
      }
      const rows = readHostRows(host);
      hideTip();
      if (rows.length > 1) {
        showTip(host, rows);
        return;
      }
      openVault(
        host.getAttribute("data-path") || "",
        host.getAttribute("data-title") || "",
      );
    });

    btn.addEventListener("mouseenter", () => {
      if (dragState) return;
      const rows = readHostRows(host);
      if (tipTimer) clearTimeout(tipTimer);
      tipTimer = setTimeout(() => {
        tipTimer = 0;
        showTip(host, rows);
      }, 220);
    });
    btn.addEventListener("mouseleave", () => scheduleHideTip(220));
  }

  function readHostRows(host) {
    try {
      const rows = JSON.parse(host.getAttribute("data-rows") || "[]");
      return Array.isArray(rows) ? rows : [];
    } catch (_) {
      return [];
    }
  }

  function paintChatBadge(rows) {
    const t = teams();
    const allow =
      window === window.top || (t?.listVisibleMessages?.() || []).length > 0;
    let host = document.getElementById(CHAT_BADGE_ID);
    if (!allow || !rows.length) {
      hideTip();
      host?.remove();
      return;
    }
    const first = rows[0];
    host = ensureChatHost();
    redockIfNeeded(host);
    host.setAttribute("data-path", first.path || "");
    host.setAttribute("data-title", first.title || "");
    // Keep payload lean — tip only needs path/title/captured_at.
    const lean = rows.map((r) => ({
      path: r.path || "",
      title: r.title || "",
      captured_at: r.captured_at || "",
      snippet_id: r.snippet_id || "",
    }));
    host.setAttribute("data-rows", JSON.stringify(lean));
    const btn = chatButton(host);
    if (btn) {
      btn.setAttribute(
        "aria-label",
        rows.length === 1
          ? `Pawn: open ${first.title || "capture"}`
          : `Pawn: ${rows.length} captures in this chat`,
      );
      let count = btn.querySelector("span.count");
      if (rows.length > 1) {
        if (!count) {
          count = document.createElement("span");
          count.className = "count";
          btn.appendChild(count);
        }
        count.textContent = rows.length > 99 ? "99+" : String(rows.length);
      } else {
        count?.remove();
      }
    }
  }

  function relevantAnchors() {
    const out = [];
    for (const row of anchors) {
      const sid = String(row.snippet_id || "").trim();
      if (!sid) continue;
      const rowPk = String(row.page_key || "").trim();
      // If both sides know a page_key and they disagree, skip — but keep
      // empty-page_key rows (soft) and keep all when our pageKey is still unknown.
      if (pageKey && rowPk && rowPk !== pageKey) continue;
      out.push(row);
    }
    return out;
  }

  async function paint() {
    const t = teams();
    if (!t?.isTeamsPage?.()) return;
    const nextKey = t.currentPageKey() || "";
    if (nextKey && nextKey !== pageKey) {
      pageKey = nextKey;
      await refreshAnchors();
    }
    const relevant = relevantAnchors();
    const validIds = new Set(
      relevant.map((row) => String(row.snippet_id || "").trim()).filter(Boolean),
    );
    if (!relevant.length) {
      removeStale(validIds);
      paintChatBadge([]);
      return;
    }
    const roots = t.listVisibleMessages();
    const live = liveBadgeIds();
    for (const row of relevant) {
      const sid = String(row.snippet_id || "").trim();
      if (!sid) continue;
      if (live.has(sid)) continue;
      for (const root of roots) {
        if (t.matchMessage(root, row)) {
          if (paintOne(root, row)) live.add(sid);
          break;
        }
      }
    }
    removeStale(validIds);
    // Chat chip for anything not currently sitting on a connected message node.
    // Survives Teams remounting the chat pane (badge lives on <html>).
    const stillLive = liveBadgeIds();
    const needsChip = relevant.filter(
      (row) => !stillLive.has(String(row.snippet_id || "").trim()),
    );
    // Floating chip for captures not currently on a live message node.
    // If nothing matched at all, still show the chip for all relevant rows.
    paintChatBadge(
      needsChip.length ? needsChip : stillLive.size ? [] : relevant,
    );
  }

  function hookSpa() {
    const wrap = (name) => {
      const orig = history[name];
      if (typeof orig !== "function") return;
      history[name] = function (...args) {
        const ret = orig.apply(this, args);
        schedulePaint(150);
        void refreshAnchors().then(() => schedulePaint(60));
        return ret;
      };
    };
    wrap("pushState");
    wrap("replaceState");
    window.addEventListener("popstate", () => schedulePaint(150));
    window.addEventListener("hashchange", () => schedulePaint(150));
  }

  function observe() {
    if (observer) {
      try {
        observer.disconnect();
      } catch (_) {
        /* ignore */
      }
    }
    // Observe a stable root — Teams replaces <main> / chat-pane on hydrate.
    observer = new MutationObserver(() => schedulePaint(200));
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
    });
    window.addEventListener("scroll", () => schedulePaint(200), true);
    if (keepAlive) clearInterval(keepAlive);
    // Backup tick: catches remounts that detach our observer target mid-flight.
    keepAlive = setInterval(() => schedulePaint(0), 2500);
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg?.type === "pawn-overlays-refresh") {
      void (async () => {
        if (Array.isArray(msg.anchors) && msg.anchors.length) {
          const byId = new Map(
            anchors.map((a) => [String(a.snippet_id || ""), a]),
          );
          for (const row of msg.anchors) {
            const sid = String(row.snippet_id || "");
            if (sid) byId.set(sid, { ...byId.get(sid), ...row });
          }
          anchors = [...byId.values()];
        } else {
          await refreshAnchors();
        }
        await paint();
        sendResponse({ ok: true, count: anchors.length });
      })();
      return true;
    }
    return false;
  });

  function debugSnapshot() {
    const chip = document.getElementById(CHAT_BADGE_ID);
    const msgBadges = [...document.querySelectorAll(`[${ATTR}]`)].map((el) => {
      const r = el.getBoundingClientRect();
      return {
        id: el.getAttribute(ATTR),
        soft: el.getAttribute("data-pawn-soft"),
        text: el.textContent?.trim(),
        rect: { top: r.top, left: r.left, w: r.width, h: r.height },
        connected: el.isConnected,
      };
    });
    return {
      pageKey,
      userDocked,
      anchors: anchors.length,
      relevant: relevantAnchors().length,
      chip: chip
        ? {
            rect: chip.getBoundingClientRect(),
            shadow: Boolean(chip.shadowRoot),
            count:
              chip.shadowRoot?.querySelector("span.count")?.textContent || null,
            rows: chip.getAttribute("data-rows"),
          }
        : null,
      msgBadges,
      pane: chatPaneRect(),
      href: location.href,
    };
  }

  window.__pawnOverlayDebug = {
    snapshot: debugSnapshot,
    refresh: () => refreshAnchors().then(() => paint()),
    resetDock: async () => {
      userDocked = false;
      try {
        await chrome.storage.local.remove(POS_KEY);
      } catch (_) {
        /* ignore */
      }
      const el = document.getElementById(CHAT_BADGE_ID);
      if (el) applyChatPos(el, null);
    },
  };

  async function boot() {
    const t = teams();
    if (!t?.isTeamsPage?.()) return;
    try {
      document.documentElement.setAttribute("data-pawn-overlays", "1");
    } catch (_) {
      /* ignore */
    }
    pageKey = t.currentPageKey() || "";
    await refreshAnchors();
    hookSpa();
    observe();
    schedulePaint(200);
    setTimeout(() => schedulePaint(0), 800);
    setTimeout(() => {
      void refreshAnchors().then(() => schedulePaint(0));
    }, 2000);
    setTimeout(() => schedulePaint(0), 5000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => void boot());
  } else {
    void boot();
  }
})();
