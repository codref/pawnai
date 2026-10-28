/** Re-paint Pawn vault badges on Teams message bubbles. */

(function () {
  if (window.__pawnOverlaysReady) return;
  window.__pawnOverlaysReady = true;

  const ATTR = "data-pawn-overlay";
  const CHAT_BADGE_ID = "pawn-overlay-chat-badge";
  const TIP_ID = "pawn-overlay-tip";
  const POS_KEY = "pawnOverlayChatPos";
  /** Chess-pawn glyph (simple silhouette). */
  const PAWN_SVG =
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
    '<path d="M12 3a3.2 3.2 0 0 0-1.6 6l-.7 2.2H8.2a1 1 0 0 0 0 2h7.6a1 1 0 0 0 0-2h-1.5L13.6 9A3.2 3.2 0 0 0 12 3zm-5 12.2c0-.7.5-1.2 1.2-1.2h9.6c.7 0 1.2.5 1.2 1.2V16c0 2.3-2.4 3.8-6 3.8s-6-1.5-6-3.8v-.8z"/>' +
    "</svg>";
  let anchors = [];
  let pageKey = "";
  let paintTimer = 0;
  let observer = null;
  let keepAlive = 0;
  let tipTimer = 0;
  let dragState = null;

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
      if (!validIds.has(id) || !el.isConnected) el.remove();
    });
  }

  function findMount(root) {
    const header =
      root.querySelector?.(
        "[data-tid*='message-header'], [data-tid*='timestamp'], time",
      )?.parentElement || root;
    return header || root;
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

  function showTip(anchorEl, rows) {
    hideTip();
    if (!anchorEl || !rows?.length) return;
    const tip = document.createElement("div");
    tip.id = TIP_ID;
    tip.className = "pawn-overlay-tip";
    tip.setAttribute("role", "tooltip");
    const n = rows.length;
    const strong = document.createElement("strong");
    strong.textContent =
      n === 1
        ? "1 Pawn capture in this chat"
        : `${n} Pawn captures in this chat`;
    tip.appendChild(strong);
    const ul = document.createElement("ul");
    for (const row of rows.slice(0, 6)) {
      const li = document.createElement("li");
      li.textContent = row.title || row.path || "capture";
      ul.appendChild(li);
    }
    if (rows.length > 6) {
      const li = document.createElement("li");
      li.textContent = `…and ${rows.length - 6} more`;
      ul.appendChild(li);
    }
    tip.appendChild(ul);
    const hint = document.createElement("div");
    hint.className = "pawn-overlay-tip-hint";
    hint.textContent = "Click to open · drag to move";
    tip.appendChild(hint);
    document.documentElement.appendChild(tip);
    const r = anchorEl.getBoundingClientRect();
    const tw = tip.offsetWidth || 200;
    const th = tip.offsetHeight || 80;
    let left = r.left + r.width / 2 - tw / 2;
    let top = r.top - th - 10;
    if (top < 8) top = r.bottom + 10;
    left = Math.max(8, Math.min(left, window.innerWidth - tw - 8));
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }

  function paintOne(root, row) {
    const sid = String(row.snippet_id || "").trim();
    if (!sid || !root || !root.isConnected) return false;
    const sel =
      typeof CSS !== "undefined" && CSS.escape ? CSS.escape(sid) : sid;
    const existing = root.querySelector?.(`[${ATTR}="${sel}"]`);
    if (existing?.isConnected) return true;
    existing?.remove();
    ensureHost(root);
    const mount = findMount(root);
    if (!mount?.isConnected) return false;
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
    if (!row.message_id) a.setAttribute("data-pawn-soft", "1");
    a.addEventListener(
      "mouseenter",
      () => {
        tipTimer = setTimeout(() => showTip(a, [row]), 280);
      },
      true,
    );
    a.addEventListener("mouseleave", () => hideTip(), true);
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
      if (
        pos &&
        Number.isFinite(pos.left) &&
        Number.isFinite(pos.top) &&
        pos.left >= 0 &&
        pos.top >= 0
      ) {
        return pos;
      }
    } catch (_) {
      /* ignore */
    }
    return null;
  }

  async function saveChatPos(left, top) {
    try {
      await chrome.storage.local.set({
        [POS_KEY]: { left, top, savedAt: Date.now() },
      });
    } catch (_) {
      /* ignore */
    }
  }

  function applyChatPos(el, pos) {
    if (!el) return;
    if (pos && Number.isFinite(pos.left) && Number.isFinite(pos.top)) {
      el.style.left = `${pos.left}px`;
      el.style.top = `${pos.top}px`;
      el.style.right = "auto";
      el.style.bottom = "auto";
    } else {
      el.style.left = "";
      el.style.top = "";
      el.style.right = "";
      el.style.bottom = "";
    }
  }

  function wireChatDrag(el) {
    if (el.__pawnDragWired) return;
    el.__pawnDragWired = true;
    let moved = false;

    const onMove = (ev) => {
      if (!dragState) return;
      const dx = ev.clientX - dragState.startX;
      const dy = ev.clientY - dragState.startY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) moved = true;
      const size = el.getBoundingClientRect().width || 36;
      let left = dragState.originLeft + dx;
      let top = dragState.originTop + dy;
      left = Math.max(4, Math.min(left, window.innerWidth - size - 4));
      top = Math.max(4, Math.min(top, window.innerHeight - size - 4));
      el.style.left = `${left}px`;
      el.style.top = `${top}px`;
      el.style.right = "auto";
      el.style.bottom = "auto";
      dragState.left = left;
      dragState.top = top;
      hideTip();
    };

    const onUp = () => {
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      el.classList.remove("is-dragging");
      if (dragState && moved) {
        void saveChatPos(dragState.left, dragState.top);
      }
      el.__pawnDidDrag = moved;
      dragState = null;
      moved = false;
    };

    el.addEventListener(
      "pointerdown",
      (ev) => {
        if (ev.button != null && ev.button !== 0) return;
        const r = el.getBoundingClientRect();
        dragState = {
          startX: ev.clientX,
          startY: ev.clientY,
          originLeft: r.left,
          originTop: r.top,
          left: r.left,
          top: r.top,
        };
        moved = false;
        el.__pawnDidDrag = false;
        el.classList.add("is-dragging");
        hideTip();
        try {
          el.setPointerCapture?.(ev.pointerId);
        } catch (_) {
          /* ignore */
        }
        window.addEventListener("pointermove", onMove, true);
        window.addEventListener("pointerup", onUp, true);
        ev.preventDefault();
        ev.stopPropagation();
      },
      true,
    );

    el.addEventListener(
      "click",
      (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        if (el.__pawnDidDrag) {
          el.__pawnDidDrag = false;
          return;
        }
        hideTip();
        openVault(el.getAttribute("data-path") || "", el.getAttribute("data-title") || "");
      },
      true,
    );

    el.addEventListener(
      "mouseenter",
      () => {
        if (dragState) return;
        const raw = el.getAttribute("data-rows") || "[]";
        let rows = [];
        try {
          rows = JSON.parse(raw);
        } catch (_) {
          rows = [];
        }
        tipTimer = setTimeout(() => showTip(el, rows), 280);
      },
      true,
    );
    el.addEventListener("mouseleave", () => hideTip(), true);
  }

  function paintChatBadge(rows) {
    const t = teams();
    const allow =
      window === window.top || (t?.listVisibleMessages?.() || []).length > 0;
    let el = document.getElementById(CHAT_BADGE_ID);
    if (!allow || !rows.length) {
      hideTip();
      el?.remove();
      return;
    }
    const first = rows[0];
    const created = !el || !el.isConnected;
    if (created) {
      el?.remove();
      el = document.createElement("a");
      el.id = CHAT_BADGE_ID;
      el.className = "pawn-overlay-badge pawn-overlay-chat";
      el.href = "#";
      el.innerHTML = PAWN_SVG;
      wireChatDrag(el);
      stableBadgeHost().appendChild(el);
      void loadChatPos().then((pos) => applyChatPos(el, pos));
    }
    el.setAttribute("data-path", first.path || "");
    el.setAttribute("data-title", first.title || "");
    el.setAttribute("data-rows", JSON.stringify(rows));
    el.setAttribute(
      "aria-label",
      rows.length === 1
        ? `Pawn: open ${first.title || "capture"}`
        : `Pawn: ${rows.length} captures in this chat`,
    );
    // Count badge = how many captures this chip represents (not matched to a bubble).
    let count = el.querySelector(".pawn-overlay-count");
    if (rows.length > 1) {
      if (!count) {
        count = document.createElement("span");
        count.className = "pawn-overlay-count";
        el.appendChild(count);
      }
      count.textContent = rows.length > 99 ? "99+" : String(rows.length);
    } else {
      count?.remove();
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
