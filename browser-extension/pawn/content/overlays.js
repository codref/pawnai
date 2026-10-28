/** Re-paint Pawn vault badges on Teams message bubbles. */

(function () {
  if (window.__pawnOverlaysReady) return;
  window.__pawnOverlaysReady = true;

  const ATTR = "data-pawn-overlay";
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
    pageKey = t.currentPageKey() || pageKey;
    try {
      const resp = await chrome.runtime.sendMessage({
        type: "pawn-anchors-get",
        page_key: pageKey,
        site: "teams",
      });
      if (resp?.ok && Array.isArray(resp.data)) {
        anchors = resp.data;
      } else if (resp?.cached && Array.isArray(resp.cached)) {
        anchors = resp.cached;
      }
    } catch (_) {
      /* background unavailable */
    }
  }

  function ensureHost(root) {
    if (!root) return;
    if (!root.classList.contains("pawn-overlay-host")) {
      const style = window.getComputedStyle(root);
      if (style.position === "static") {
        root.classList.add("pawn-overlay-host");
      } else {
        root.classList.add("pawn-overlay-host");
      }
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
    if (!sid || !root) return;
    const existing = root.querySelector?.(`[${ATTR}="${CSS.escape(sid)}"]`);
    if (existing) return;
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
  }

  async function paint() {
    const t = teams();
    if (!t?.isTeamsPage?.()) return;
    const nextKey = t.currentPageKey() || "";
    if (nextKey && nextKey !== pageKey) {
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
      return;
    }
    const roots = t.listVisibleMessages();
    for (const row of anchors) {
      const sid = String(row.snippet_id || "").trim();
      if (!sid) continue;
      for (const root of roots) {
        if (t.matchMessage(root, row)) {
          paintOne(root, row);
          break;
        }
      }
    }
    // Drop badges whose vault anchor disappeared (virtualized hosts remove themselves).
    removeStale(validIds);
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
      document.querySelector("[data-tid*='message-pane'], [data-tid*='chat-pane'], main") ||
      document.body;
    observer = new MutationObserver(() => schedulePaint(100));
    observer.observe(target, { childList: true, subtree: true });
    window.addEventListener(
      "scroll",
      () => schedulePaint(150),
      true,
    );
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg?.type === "pawn-overlays-refresh") {
      void (async () => {
        if (Array.isArray(msg.anchors)) {
          anchors = msg.anchors;
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
    pageKey = t.currentPageKey() || "";
    await refreshAnchors();
    hookSpa();
    observe();
    schedulePaint(200);
    // Teams hydrates slowly — a couple of follow-up paints.
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
