/** Service worker: context menus, snip handoff, side panel open. */

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

chrome.action.onClicked.addListener(async (tab) => {
  if (tab?.id != null && chrome.sidePanel?.open) {
    try {
      await chrome.sidePanel.open({ tabId: tab.id });
    } catch (_) {
      /* Firefox / unsupported */
    }
  }
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!tab?.id) return;
  try {
    if (chrome.sidePanel?.open) {
      await chrome.sidePanel.open({ tabId: tab.id });
    }
  } catch (_) {
    /* ignore */
  }
  if (info.menuItemId === MENU_SELECTION) {
    await captureSelection(tab);
  } else if (info.menuItemId === MENU_REGION) {
    await captureRegion(tab);
  }
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "pawn-snip-selection") {
    chrome.tabs.query({ active: true, currentWindow: true }).then(async (tabs) => {
      const tab = tabs[0];
      if (!tab) {
        sendResponse({ ok: false, error: "No active tab" });
        return;
      }
      try {
        await captureSelection(tab);
        sendResponse({ ok: true });
      } catch (err) {
        sendResponse({ ok: false, error: String(err?.message || err) });
      }
    });
    return true;
  }
  if (msg?.type === "pawn-snip-region") {
    chrome.tabs.query({ active: true, currentWindow: true }).then(async (tabs) => {
      const tab = tabs[0];
      if (!tab) {
        sendResponse({ ok: false, error: "No active tab" });
        return;
      }
      try {
        await captureRegion(tab);
        sendResponse({ ok: true });
      } catch (err) {
        sendResponse({ ok: false, error: String(err?.message || err) });
      }
    });
    return true;
  }
  return false;
});

async function captureSelection(tab) {
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
