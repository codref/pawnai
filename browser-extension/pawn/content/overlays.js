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

  function teams() {
    return window.__pawnTeams || null;
  }

  function schedulePaint(delayMs) {
    if (paintTimer) clearTimeout(paintTimer);
    paintTimer = setTimeout(() => {
      paintTimer = 0;
      void paint();
    }, delayMs == null ? 80 : delayMs);
  }

  async function refreshAnchors() {
    const t = teams();
    if (!t?.isTeamsPage?.()) return;
    pageKey = t.currentPageKey() || pageKey || "";
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
      anchors = rows;
    } catch (_) {
      /* background unavailable */
    }
  }

  function ensureHost(root) {
    if (!root) return;
    if (!root.classList.contains("pawn-overlay-host")) {
      root.classList.add("pawn-overlay-host");
    }
  }

  function removeStale(validIds) {
    document.querySelectorAll(`[${ATTR}]`).forEach((el) => {
      const id = el.getAttribute(ATTR);
      if (!validIds.has(id)) el.remove();
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
    if (!sid || !root) return false;
    const existing = root.querySelector?.(
      `[${ATTR}="${typeof CSS !== "undefined" && CSS.escape ? CSS.escape(sid) : sid}"]`,
    );
    if (existing) return true;
    ensureHost(root);
    const mount = findMount(root);
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
    return true;
  }

  function paintChatBadge(unmatched) {
    // Only top frame (or a frame that actually hosts messages) — avoid N badges
    // when content_scripts run with all_frames on cloud.microsoft.
    const t = teams();
    const allow =
      window === window.top || (t?.listVisibleMessages?.() || []).length > 0;
    let el = document.getElementById(CHAT_BADGE_ID);
    if (!allow || !unmatched.length) {
      el?.remove();
      return;
    }
    const first = unmatched[0];
    if (!el) {
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
      const host =
        document.querySelector(
          "[data-tid*='message-pane'], [data-tid*='chat-pane'], main",
        ) || document.body;
      if (getComputedStyle(host).position === "static") {
        host.classList.add("pawn-overlay-host");
      }
      host.appendChild(el);
    }
    el.setAttribute("data-path", first.path || "");
    el.setAttribute("data-title", first.title || "");
    el.title = first.title
      ? `Open in Obsidian: ${first.title}`
      : "Open capture in Obsidian";
    el.textContent =
      unmatched.length === 1 ? "Pawn" : `Pawn · ${unmatched.length}`;
  }

  async function paint() {
    const t = teams();
    if (!t?.isTeamsPage?.()) return;
    const nextKey = t.currentPageKey() || "";
    if (nextKey !== pageKey) {
      pageKey = nextKey;
      await refreshAnchors();
    }
    const validIds = new Set(
      anchors
        .map((row) => String(row.snippet_id || "").trim())
        .filter(Boolean),
    );
    if (!anchors.length) {
      removeStale(validIds);
      paintChatBadge([]);
      return;
    }
    const roots = t.listVisibleMessages();
    const unmatched = [];
    for (const row of anchors) {
      const sid = String(row.snippet_id || "").trim();
      if (!sid) continue;
      // Prefer same page_key when both sides know it.
      const rowPk = String(row.page_key || "").trim();
      if (pageKey && rowPk && rowPk !== pageKey) continue;
      let hit = false;
      for (const root of roots) {
        if (t.matchMessage(root, row)) {
          hit = paintOne(root, row);
          break;
        }
      }
      if (!hit) unmatched.push(row);
    }
    removeStale(validIds);
    paintChatBadge(unmatched);
  }

  function hookSpa() {
    const wrap = (name) => {
      const orig = history[name];
      if (typeof orig !== "function") return;
      history[name] = function (...args) {
        const ret = orig.apply(this, args);
        schedulePaint(120);
        void refreshAnchors().then(() => schedulePaint(40));
        return ret;
      };
    };
    wrap("pushState");
    wrap("replaceState");
    window.addEventListener("popstate", () => schedulePaint(120));
    window.addEventListener("hashchange", () => schedulePaint(120));
  }

  function observe() {
    if (observer) return;
    const target =
      document.querySelector(
        "[data-tid*='message-pane'], [data-tid*='chat-pane'], main",
      ) || document.body;
    observer = new MutationObserver(() => schedulePaint(100));
    observer.observe(target, { childList: true, subtree: true });
    window.addEventListener("scroll", () => schedulePaint(150), true);
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
    setTimeout(() => schedulePaint(0), 1000);
    setTimeout(() => {
      void refreshAnchors().then(() => schedulePaint(0));
    }, 3000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => void boot());
  } else {
    void boot();
  }
})();
