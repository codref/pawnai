/** Re-paint Pawn vault badges on Teams message bubbles. */

(function () {
  if (window.__pawnOverlaysReady) return;
  window.__pawnOverlaysReady = true;

  const ATTR = "data-pawn-overlay";
  const CHAT_BADGE_ID = "pawn-overlay-chat-badge";
  let anchors = [];
  let pageKey = "";
  let paintTimer = 0;
  let observer = null;
  let keepAlive = 0;

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
    a.title = row.title ? `Open in Obsidian: ${row.title}` : "Open in Obsidian";
    a.textContent = "Pawn";
    if (!row.message_id) a.setAttribute("data-pawn-soft", "1");
    a.addEventListener(
      "click",
      (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        void chrome.runtime.sendMessage({
          type: "pawn-overlay-open",
          path: row.path,
          title: row.title || "",
        });
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

  function paintChatBadge(rows) {
    const t = teams();
    const allow =
      window === window.top || (t?.listVisibleMessages?.() || []).length > 0;
    let el = document.getElementById(CHAT_BADGE_ID);
    if (!allow || !rows.length) {
      el?.remove();
      return;
    }
    const first = rows[0];
    if (!el || !el.isConnected) {
      el?.remove();
      el = document.createElement("a");
      el.id = CHAT_BADGE_ID;
      el.className = "pawn-overlay-badge pawn-overlay-chat";
      el.href = "#";
      el.addEventListener(
        "click",
        (ev) => {
          ev.preventDefault();
          ev.stopPropagation();
          const path = el.getAttribute("data-path") || "";
          void chrome.runtime.sendMessage({
            type: "pawn-overlay-open",
            path,
            title: el.getAttribute("data-title") || "",
          });
        },
        true,
      );
      stableBadgeHost().appendChild(el);
    }
    el.setAttribute("data-path", first.path || "");
    el.setAttribute("data-title", first.title || "");
    el.title = first.title
      ? `Open in Obsidian: ${first.title}`
      : "Open capture in Obsidian";
    el.textContent = rows.length === 1 ? "Pawn" : `Pawn · ${rows.length}`;
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
