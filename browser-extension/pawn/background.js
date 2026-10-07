/** Background: context menus, snip handoff, side panel / sidebar open.
 *
 * Chrome MV3 loads this as a service worker; Firefox MV3 as background.scripts.
 */

const MENU_SELECTION = "pawn-capture-selection";
const MENU_REGION = "pawn-capture-region";

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
    await pushPending({ error: hostPermissionError(tab.url, err) });
  }
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "pawn-snip-selection") {
    (async () => {
      const tab = await activeContentTab();
      if (!tab) {
        sendResponse({ ok: false, error: "No active tab" });
        return;
      }
      try {
        await captureSelection(tab);
        sendResponse({ ok: true });
      } catch (err) {
        const error = hostPermissionError(tab.url, err);
        await pushPending({ error });
        sendResponse({ ok: false, error });
      }
    })();
    return true;
  }
  if (msg?.type === "pawn-snip-region") {
    (async () => {
      const tab = await activeContentTab();
      if (!tab) {
        sendResponse({ ok: false, error: "No active tab" });
        return;
      }
      try {
        await captureRegion(tab);
        sendResponse({ ok: true });
      } catch (err) {
        const error = hostPermissionError(tab.url, err);
        await pushPending({ error });
        sendResponse({ ok: false, error });
      }
    })();
    return true;
  }
  return false;
});

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
    await pushPending({
      error: response?.error || "Region capture cancelled",
    });
    return;
  }
  await enqueueSnippet({
    kind: "image",
    data_base64: response.data_base64,
    media_type: "image/png",
    source_url: tab.url || "",
    page_title: tab.title || "",
  });
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
