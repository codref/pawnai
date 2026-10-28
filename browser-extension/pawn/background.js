/** Background: context menus, snip handoff, side panel / sidebar open.
 *
 * Chrome MV3 loads this as a service worker; Firefox MV3 as background.scripts.
 */

const MENU_SELECTION = "pawn-capture-selection";
const MENU_REGION = "pawn-capture-region";

/** Only one region / visible clip at a time (side panel + context menu). */
let clipBusy = false;

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: MENU_SELECTION,
      title: "Send selection to Pawn",
      contexts: ["selection"],
    });
    chrome.contextMenus.create({
      id: MENU_REGION,
      title: "Capture region to Pawn",
      contexts: ["page", "frame"],
    });
  });
  if (chrome.sidePanel?.setPanelBehavior) {
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});
  }
});

/** Open Chrome side panel or Firefox sidebar (best effort). */
async function openUi(tab) {
  if (tab?.id != null && chrome.sidePanel?.open) {
    try {
      await chrome.sidePanel.open({ tabId: tab.id });
      return;
    } catch (_) {
      /* fall through */
    }
  }
  if (chrome.sidebarAction?.open) {
    try {
      await chrome.sidebarAction.open();
    } catch (_) {
      /* user can open View → Sidebar → Pawn Capture */
    }
  }
}

/** Prefer the focused content tab (sidebar queries confuse currentWindow). */
async function activeContentTab() {
  const focused = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  let tab = focused.find((t) => t.id != null && isCapturableUrl(t.url));
  if (tab) return tab;
  const active = await chrome.tabs.query({ active: true });
  tab = active.find((t) => t.id != null && isCapturableUrl(t.url));
  if (tab) return tab;
  return focused[0] || active[0] || null;
}

function isCapturableUrl(url) {
  if (!url) return false;
  return /^(https?:|file:)/i.test(url);
}

function hostPermissionError(url, err) {
  const msg = String(err?.message || err || "");
  if (/host permission|missing host|cannot access/i.test(msg)) {
    return (
      "Missing host permission for this tab. Reload the Firefox add-on from " +
      "pawn-capture-firefox.zip (v0.1.1+), or grant Access your data for all websites " +
      "under about:addons → Pawn Capture → Permissions. Tab: " +
      (url || "?")
    );
  }
  return msg;
}

/** Ensure we can inject / capture on *tab* (Firefox may not grant host perms until asked). */
async function ensureTabAccess(tab) {
  if (!tab?.id) throw new Error("No active tab");
  if (!isCapturableUrl(tab.url)) {
    throw new Error(
      "Cannot capture this page (open an http(s) tab, then use Selection / Region).",
    );
  }
  if (!chrome.permissions?.contains) return;
  try {
    const origin = new URL(tab.url).origin + "/*";
    const hasOrigin = await chrome.permissions.contains({ origins: [origin] });
    const hasAll = await chrome.permissions.contains({ origins: ["<all_urls>"] });
    if (hasOrigin || hasAll) return;
    if (!chrome.permissions.request) {
      throw new Error(
        "Missing host permission. Grant it under about:addons → Pawn Capture → Permissions.",
      );
    }
    const granted = await chrome.permissions.request({ origins: ["<all_urls>"] });
    if (!granted) {
      throw new Error(
        "Host permission denied. Allow Access your data for all websites for Pawn Capture.",
      );
    }
  } catch (err) {
    if (/Missing host permission|Host permission denied|Cannot capture/i.test(String(err.message))) {
      throw err;
    }
    // permissions API quirks — continue and let scripting fail with a clear message
  }
}

chrome.action.onClicked.addListener(async (tab) => {
  await openUi(tab);
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!tab?.id) return;
  await openUi(tab);
  try {
    if (info.menuItemId === MENU_SELECTION) {
      await captureSelection(tab);
    } else if (info.menuItemId === MENU_REGION) {
      await captureRegion(tab);
    }
  } catch (err) {
    const error = hostPermissionError(tab.url, err);
    if (!isBenignClipError(error)) await pushPending({ error });
  }
});

/**
 * Side-panel snips use a long-lived port (not one-shot sendMessage).
 * Full-viewport capture often outlives the one-shot reply channel in MV3,
 * which surfaces as "The message port closed before a response was received."
 * A connected port keeps the service worker alive until we post the result.
 */
chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "pawn-snip") return;
  port.onMessage.addListener((msg) => {
    const mode =
      msg?.type === "pawn-snip-selection"
        ? "selection"
        : msg?.type === "pawn-snip-region"
          ? "region"
          : msg?.type === "pawn-snip-visible"
            ? "visible"
            : null;
    if (!mode) {
      safePortPost(port, { ok: false, error: "Unknown capture mode" });
      return;
    }
    void runSnip(mode).then((result) => safePortPost(port, result));
  });
});

function safePortPost(port, msg) {
  try {
    port.postMessage(msg);
  } catch (_) {
    /* port already disconnected — snippet/error still land via storage */
  }
}

function isBenignClipError(error) {
  return /already in progress|cancelled/i.test(String(error || ""));
}

async function runSnip(mode) {
  const tab = await activeContentTab();
  if (!tab) return { ok: false, error: "No active tab" };
  try {
    if (mode === "selection") await captureSelection(tab);
    else if (mode === "region") await captureRegion(tab);
    else if (mode === "visible") await captureVisible(tab);
    else return { ok: false, error: "Unknown capture mode" };
    return { ok: true };
  } catch (err) {
    const error = hostPermissionError(tab.url, err);
    if (!isBenignClipError(error)) {
      try {
        await pushPending({ error });
      } catch (_) {
        /* ignore storage failures while reporting */
      }
    }
    return { ok: false, error };
  }
}

async function captureSelection(tab) {
  await ensureTabAccess(tab);
  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      const text = String(window.getSelection?.()?.toString() || "").trim();
      return {
        text,
        url: location.href,
        title: document.title || "",
      };
    },
  });
  if (!result?.text) {
    await pushPending({ error: "No text selected" });
    return;
  }
  await enqueueSnippet({
    kind: "text",
    text: result.text,
    source_url: result.url || tab.url || "",
    page_title: result.title || tab.title || "",
  });
}

async function captureRegion(tab) {
  if (clipBusy) throw new Error("Region capture already in progress");
  clipBusy = true;
  try {
    await ensureTabAccess(tab);
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
      format: "png",
    });
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content/region.js"],
    });
    const response = await chrome.tabs.sendMessage(tab.id, {
      type: "pawn-region-pick",
      dataUrl,
    });
    if (!response?.ok) {
      throw new Error(response?.error || "Region capture cancelled");
    }
    await enqueueSnippet({
      kind: "image",
      data_base64: response.data_base64,
      media_type: "image/png",
      source_url: tab.url || "",
      page_title: tab.title || "",
    });
  } finally {
    clipBusy = false;
  }
}

async function captureVisible(tab) {
  if (clipBusy) throw new Error("Region capture already in progress");
  clipBusy = true;
  try {
    await ensureTabAccess(tab);
    // JPEG keeps full-viewport captures under storage pressure vs PNG.
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
      format: "jpeg",
      quality: 90,
    });
    const b64 = String(dataUrl || "").split(",", 2)[1] || "";
    if (!b64) throw new Error("Visible capture failed");
    await enqueueSnippet({
      kind: "image",
      data_base64: b64,
      media_type: "image/jpeg",
      source_url: tab.url || "",
      page_title: tab.title || "",
    });
  } finally {
    clipBusy = false;
  }
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
  const { pendingSnippets = [] } = await chrome.storage.local.get("pendingSnippets");
  pendingSnippets.push(snippet);
  await chrome.storage.local.set({
    pendingSnippets,
    lastPageTitle: partial.page_title || "",
    lastPageUrl: partial.source_url || "",
  });
}

async function pushPending(payload) {
  await chrome.storage.local.set({ pendingNotice: payload });
}

const OUTLOOK_HOST_RE =
  /(^|\.)outlook\.(office|office365|live)\.com$|(^|\.)outlook\.cloud\.microsoft$/i;

function isOutlookUrl(url) {
  try {
    const host = new URL(url || "").hostname;
    return OUTLOOK_HOST_RE.test(host);
  } catch (_) {
    return false;
  }
}

/** Best-effort event fields from an open Outlook calendar detail page. */
async function scrapeOutlookEvent(tab) {
  await ensureTabAccess(tab);
  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      const norm = (s) => String(s || "").replace(/\s+/g, " ").trim();
      const isStubTitle = (t) =>
        /^(calendar|mail|outlook)\b/i.test(t) ||
        /-\s*outlook\s*$/i.test(t) ||
        /^calendar\s*-/i.test(t);

      const candidates = [];
      const push = (t) => {
        const v = norm(t);
        if (v && v.length >= 2 && v.length < 200) candidates.push(v);
      };
      for (const sel of [
        'input[aria-label*="Add a title" i]',
        'input[aria-label*="title" i]',
        'textarea[aria-label*="title" i]',
        '[aria-label="Event"] [role="heading"]',
        '[data-app-section="CalendarItemComposition"] h1',
        '[data-app-section="CalendarItemComposition"] [role="heading"]',
        '[role="heading"][aria-level="1"]',
        '[role="heading"][aria-level="2"]',
        "h1",
        "h2",
      ]) {
        document.querySelectorAll(sel).forEach((n) => {
          push(n.value || n.getAttribute("aria-label") || n.textContent);
        });
      }
      // Invite subject often sits above "Microsoft Teams meeting".
      const bodyText = norm(document.body?.innerText || "").slice(0, 8000);
      const teamsIdx = bodyText.search(/Microsoft Teams meeting/i);
      if (teamsIdx > 0) {
        const before = bodyText.slice(Math.max(0, teamsIdx - 180), teamsIdx);
        const lines = before
          .split(/\n| {2,}/)
          .map(norm)
          .filter(Boolean);
        if (lines.length) push(lines[lines.length - 1]);
      }

      let title = "";
      for (const c of candidates) {
        if (!isStubTitle(c)) {
          title = c;
          break;
        }
      }
      if (!title) title = candidates.find((c) => !isStubTitle(c)) || "";
      if (!title || isStubTitle(title)) {
        const page = norm(document.title || "");
        title = isStubTitle(page) ? title || page : page;
      }
      title = title.replace(/\s+[–—|-]\s+Mail.*$/i, "").trim();

      // Prefer the visible range: "10/28/2026, 11:00 AM – 11:30 AM"
      let start = "";
      let end = "";
      const range = bodyText.match(
        /(\d{1,2}\/\d{1,2}\/\d{4}),?\s+(\d{1,2}:\d{2}\s*[AP]M)\s*[–—-]\s*(\d{1,2}:\d{2}\s*[AP]M)/i,
      );
      if (range) {
        start = `${range[1]}, ${range[2]}`;
        end = `${range[1]}, ${range[3]}`;
      } else {
        const timeBits = Array.from(
          document.querySelectorAll(
            "time, input[aria-label*='Start' i], input[aria-label*='End' i]",
          ),
        )
          .map((n) => {
            if (n.tagName === "TIME") {
              return n.getAttribute("datetime") || n.textContent || "";
            }
            return n.value || n.getAttribute("aria-label") || n.textContent || "";
          })
          .map(norm)
          .filter(Boolean);
        start = timeBits[0] || "";
        end = timeBits[1] || "";
      }

      const attendees = Array.from(
        document.querySelectorAll(
          '[aria-label*="Attendee" i], [aria-label*="Participant" i], [data-testid*="attendee" i]',
        ),
      )
        .map((n) => norm(n.textContent))
        .filter((s) => s && s.length < 80)
        .slice(0, 12);
      const selection = norm(window.getSelection?.()?.toString() || "");
      return {
        title,
        start,
        end,
        attendees,
        selection,
        url: location.href,
        page_title: document.title || "",
      };
    },
  });
  return result || null;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type !== "pawn-scrape-outlook") return;
  void (async () => {
    try {
      const tab = await activeContentTab();
      if (!tab) {
        sendResponse({ ok: false, error: "No active tab" });
        return;
      }
      if (!isOutlookUrl(tab.url || "")) {
        sendResponse({
          ok: false,
          error: "Open an Outlook calendar event first",
          url: tab.url || "",
          title: tab.title || "",
        });
        return;
      }
      const scraped = await scrapeOutlookEvent(tab);
      sendResponse({
        ok: true,
        outlook: true,
        ...(scraped || {}),
        url: scraped?.url || tab.url || "",
        page_title: scraped?.page_title || tab.title || "",
      });
    } catch (err) {
      sendResponse({ ok: false, error: String(err?.message || err) });
    }
  })();
  return true;
});
