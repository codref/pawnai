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

const TEAMS_HOST_RE =
  /(^|\.)teams\.(microsoft|live)\.com$|(^|\.)teams\.cloud\.microsoft$/i;

function isTeamsUrl(url) {
  try {
    return TEAMS_HOST_RE.test(new URL(url || "").hostname);
  } catch (_) {
    return false;
  }
}

async function ensureTeamsScripts(tab) {
  if (!tab?.id || !isTeamsUrl(tab.url || "")) return;
  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ["content/adapters/teams.js", "content/overlays.js"],
  });
  try {
    await chrome.scripting.insertCSS({
      target: { tabId: tab.id },
      files: ["content/overlay.css"],
    });
  } catch (_) {
    /* already injected via content_scripts */
  }
}

async function teamsTabMessage(tab, message) {
  if (!tab?.id || !isTeamsUrl(tab.url || "")) return null;
  try {
    return await chrome.tabs.sendMessage(tab.id, message);
  } catch (_) {
    try {
      await ensureTeamsScripts(tab);
      return await chrome.tabs.sendMessage(tab.id, message);
    } catch (err) {
      return { ok: false, error: String(err?.message || err) };
    }
  }
}

async function captureSelection(tab) {
  await ensureTabAccess(tab);
  let result = null;
  if (isTeamsUrl(tab.url || "")) {
    const teams = await teamsTabMessage(tab, { type: "pawn-teams-capture-selection" });
    if (teams?.ok && teams.text) {
      result = {
        text: teams.text,
        url: teams.url || tab.url || "",
        title: teams.title || tab.title || "",
        anchor: teams.anchor,
      };
    }
  }
  if (!result) {
    const [{ result: plain }] = await chrome.scripting.executeScript({
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
    result = plain;
  }
  if (!result?.text) {
    await pushPending({ error: "No text selected" });
    return;
  }
  await enqueueSnippet({
    kind: "text",
    text: result.text,
    source_url: result.url || tab.url || "",
    page_title: result.title || tab.title || "",
    anchor: result.anchor || undefined,
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
    let anchor;
    let sourceUrl = tab.url || "";
    if (
      isTeamsUrl(tab.url || "") &&
      Number.isFinite(response.centerX) &&
      Number.isFinite(response.centerY)
    ) {
      // Let the region overlay finish unmounting before hit-testing the DOM.
      await new Promise((r) => setTimeout(r, 50));
      await ensureTeamsScripts(tab);
      const hit = await teamsTabMessage(tab, {
        type: "pawn-teams-capture-point",
        clientX: response.centerX,
        clientY: response.centerY,
      });
      if (hit?.ok && hit.anchor) {
        anchor = hit.anchor;
        if (hit.url) sourceUrl = hit.url;
      }
      if (hit && hit.matched === false) {
        await pushPending({
          info:
            "Region clipped — Teams bubble not identified. After Save, look for a Pawn chip (top-right). Selection capture binds the badge to one message more reliably.",
        });
      }
    }
    await enqueueSnippet({
      kind: "image",
      data_base64: response.data_base64,
      media_type: "image/png",
      source_url: sourceUrl,
      page_title: tab.title || "",
      anchor,
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
  if (partial.anchor && typeof partial.anchor === "object") {
    snippet.anchor = partial.anchor;
  }
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

const CALENDAR_HOST_RE =
  /(^|\.)outlook\.(office|office365|live)\.com$|(^|\.)outlook\.cloud\.microsoft$|(^|\.)teams\.(microsoft|live)\.com$|(^|\.)teams\.cloud\.microsoft$/i;

function isCalendarUrl(url) {
  try {
    const host = new URL(url || "").hostname;
    return CALENDAR_HOST_RE.test(host);
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
      const CHROME_TITLE =
        /^(?:\(\d+\)\s*)?(?:calendar|mail|outlook|microsoft\s*teams|tracking|accepted|declined|busy|free|tentative|reminder|categorize|private|details|organizer|attendees|participants|chat|join|copilot|prepare|meeting|inbox|day|week|work week|month)\b/i;
      const isStubTitle = (t) =>
        !t ||
        CHROME_TITLE.test(t) ||
        /calendar\s*[|–—-]\s*microsoft\s*teams/i.test(t) ||
        /-\s*outlook\s*$/i.test(t) ||
        /^calendar\s*-/i.test(t) ||
        /^prepare for this meeting$/i.test(t) ||
        /^microsoft teams meeting$/i.test(t) ||
        /^add a (?:title|room)/i.test(t);
      const junkPerson =
        /^(required|optional|organizer|attendees|participants|accepted|tentative|invite|chat|join|copy|delete|archive|reply|forward|busy|reminder|categorize|private|prepare|copilot|viva|tracking|meeting|calendar|more|close|cancel|save|send|help|you|me)$/i;
      const isJunkPerson = (name) =>
        !name ||
        junkPerson.test(name) ||
        /\[(?:image|object)[^\]]*\]/i.test(name) ||
        /^https?:/i.test(name) ||
        name.length < 2 ||
        name.length > 80 ||
        !/[A-Za-zÀ-ÿ]/.test(name) ||
        // Toolbar singles are lowercase ("chat"); allow "Ada" / emails.
        (!name.includes(" ") && !name.includes("@") && !/^[A-ZÀ-Ÿ]/.test(name));

      const subjectFromChrome = (raw) => {
        const s = norm(raw);
        // "Review Insights Connector - Meeting - Calendar"
        let m = s.match(/^(.+?)\s*[–—|-]\s*Meeting\s*[–—|-]\s*Calendar\b/i);
        if (m) return norm(m[1]);
        m = s.match(/^(.+?)\s*[–—|-]\s*(?:Mail|Outlook|Calendar)\b/i);
        if (m && !isStubTitle(m[1])) return norm(m[1]);
        return "";
      };

      const scored = [];
      const pushTitle = (t, score) => {
        let v = norm(t);
        if (!v || v.length < 2 || v.length > 160) return;
        const fromBreadcrumb = subjectFromChrome(v);
        if (fromBreadcrumb) {
          scored.push({ t: fromBreadcrumb, score: score + 50 });
          return;
        }
        v = v.replace(/\s+[–—|-]\s+Mail.*$/i, "").trim();
        if (isStubTitle(v)) return;
        scored.push({ t: v, score });
      };

      // Strong signals first (breadcrumb / tab / event composition).
      pushTitle(document.title || "", 80);
      const crumb = document.querySelector(
        '[role="navigation"] [aria-current], [data-app-section*="Breadcrumb" i], nav',
      );
      if (crumb) pushTitle(crumb.textContent, 70);
      for (const sel of [
        'input[aria-label*="Add a title" i]',
        'input[aria-label*="Event title" i]',
        'input[aria-label*="title" i]',
        'textarea[aria-label*="title" i]',
        '[data-app-section="CalendarItemComposition"] [role="heading"]',
        '[data-app-section="CalendarItemComposition"] h1',
        '[data-tid="meeting-title"]',
        '[data-tid="event-title"]',
        '[data-tid="calendar-event-title"]',
        '[aria-label="Event"] [role="heading"]',
      ]) {
        document.querySelectorAll(sel).forEach((n) => {
          pushTitle(n.value || n.textContent, 60);
        });
      }

      const bodyRaw = String(document.body?.innerText || "").slice(0, 16000);
      const subjectLine = bodyRaw.match(/(?:^|\n)\s*Subject\s*:\s*([^\n]+)/i);
      if (subjectLine) pushTitle(subjectLine[1], 55);
      // Title often sits just above the attendee line or Teams block.
      const peopleLine = bodyRaw.match(
        /(?:^|\n)([^\n]{4,120})\n+[^\n]*?(?:Mehrdad|Required|Optional|Organizer|; ).{0,80}\n/i,
      );
      // Prefer explicit "Name; Name" attendee strip's preceding heading via lines.
      const lines = bodyRaw
        .split("\n")
        .map(norm)
        .filter(Boolean);
      for (let i = 0; i < lines.length; i++) {
        if (/^(Mehrdad|Davide|[A-Z][a-z]+\s+[A-Z])/.test(lines[i]) && /;/.test(lines[i])) {
          if (i > 0) pushTitle(lines[i - 1], 45);
        }
        if (/^Microsoft Teams meeting$/i.test(lines[i]) && i > 0) {
          // Walk back past date/location chrome to a subject-like line.
          for (let j = i - 1; j >= Math.max(0, i - 6); j--) {
            if (
              !isStubTitle(lines[j]) &&
              !/^\d{1,2}\/\d{1,2}\/\d{4}/.test(lines[j]) &&
              !/^(Wed|Thu|Fri|Sat|Sun|Mon|Tue)\b/i.test(lines[j]) &&
              !/^https?:/i.test(lines[j]) &&
              !/;/.test(lines[j])
            ) {
              pushTitle(lines[j], 40);
              break;
            }
          }
        }
      }
      void peopleLine;

      scored.sort((a, b) => b.score - a.score || b.t.length - a.t.length);
      let title = scored[0]?.t || "";
      if (!title || isStubTitle(title)) {
        const pageSub = subjectFromChrome(document.title || "");
        title = pageSub || title;
      }

      let start = "";
      let end = "";
      const range = bodyRaw.match(
        /(\d{1,2}\/\d{1,2}\/\d{4}),?\s+(\d{1,2}:\d{2}\s*[AP]M)\s*[–—-]\s*(\d{1,2}:\d{2}\s*[AP]M)/i,
      );
      const range2 = bodyRaw.match(
        /(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\s+(\d{1,2}\/\d{1,2}\/\d{4})\s+(\d{1,2}:\d{2}\s*[AP]M)\s*[–—-]\s*(\d{1,2}:\d{2}\s*[AP]M)/i,
      );
      const hit = range || range2;
      if (hit) {
        start = `${hit[1]}, ${hit[2]}`;
        end = `${hit[1]}, ${hit[3]}`;
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

      const people = [];
      let organizer = "";
      const pushPerson = (raw, role) => {
        const name = norm(
          String(raw || "")
            .replace(/\s*\([^)]*\)\s*$/, "")
            .replace(/\s*\[(?:image|object)[^\]]*\]/gi, ""),
        );
        if (isJunkPerson(name)) return;
        if (role === "organizer" && !organizer) organizer = name;
        if (!people.some((p) => p.toLowerCase() === name.toLowerCase())) {
          people.push(name);
        }
      };
      for (const m of bodyRaw.matchAll(
        /^(.{2,80}?)\s*\((Organizer|Required|Optional|Accepted|Tentative)\)\s*$/gim,
      )) {
        pushPerson(m[1], m[2].toLowerCase());
      }
      // "Mehrdad Kiani; Davide Dal Farra" on its own line.
      for (const line of lines) {
        if (
          /;/.test(line) &&
          line.length < 160 &&
          !/^https?:/i.test(line) &&
          !/meeting id/i.test(line)
        ) {
          const parts = line.split(";").map(norm).filter(Boolean);
          if (
            parts.length >= 2 &&
            parts.length <= 12 &&
            parts.every((p) => p.includes(" ") || p.includes("@"))
          ) {
            parts.forEach((p) => pushPerson(p));
          }
        }
      }
      const attLine = bodyRaw.match(
        /(?:^|\n)\s*(?:Attendees|Participants|Required attendees|Invitees)\s*:\s*([^\n]+)/i,
      );
      if (attLine) {
        attLine[1].split(/[,;|/]/).forEach((p) => pushPerson(p));
      }
      const orgLine = bodyRaw.match(
        /(?:^|\n)\s*(?:Organizer|From|Hosted by)\s*:\s*([^\n]+)/i,
      );
      if (orgLine) pushPerson(orgLine[1], "organizer");
      // Tracking pane often labels the organizer explicitly.
      const trackingOrg = bodyRaw.match(
        /(?:^|\n)\s*([A-Z][^\n]{1,60})\n+Invitation sent\b/i,
      );
      if (trackingOrg) pushPerson(trackingOrg[1], "organizer");

      let location = "";
      const locLine = bodyRaw.match(
        /(?:^|\n)\s*(?:Location|Where|Room)\s*:\s*([^\n]+)/i,
      );
      if (locLine && !/^add a room/i.test(locLine[1])) {
        location = norm(locLine[1]).slice(0, 160);
      }

      let meeting_url = "";
      const meet = bodyRaw.match(
        /https?:\/\/(?:teams\.microsoft\.com|teams\.live\.com|meet\.google\.com|[\w.-]*zoom\.us)\/\S+/i,
      );
      if (meet) meeting_url = meet[0].replace(/[).,;]+$/, "");
      else {
        const a = document.querySelector(
          'a[href*="teams.microsoft.com"], a[href*="teams.live.com"], a[href*="meet.google.com"], a[href*="zoom.us"]',
        );
        if (a?.href) meeting_url = a.href;
      }

      let meeting_id = "";
      const mid = bodyRaw.match(/Meeting ID:\s*([0-9\s]+)/i);
      if (mid) meeting_id = norm(mid[1]);
      let passcode = "";
      const pc = bodyRaw.match(/Passcode:\s*(\S+)/i);
      if (pc) passcode = norm(pc[1]);

      const selection = norm(window.getSelection?.()?.toString() || "");
      return {
        title,
        start,
        end,
        attendees: people.slice(0, 40),
        organizer,
        location,
        meeting_url,
        meeting_id,
        passcode,
        selection,
        url: window.location.href,
        page_title: document.title || "",
      };
    },
  });
  return result || null;
}

async function loadSettings() {
  const { settings = {} } = await chrome.storage.local.get("settings");
  return {
    serverUrl: String(settings.serverUrl || "http://127.0.0.1:8000").replace(/\/+$/, ""),
    apiToken: String(settings.apiToken || "").trim(),
  };
}

async function apiFetch(method, path, body) {
  const { serverUrl, apiToken } = await loadSettings();
  if (!serverUrl) throw new Error("Configure server URL in Pawn Capture settings");
  const headers = { Accept: "application/json" };
  if (apiToken) headers.Authorization = `Bearer ${apiToken}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const resp = await fetch(`${serverUrl}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(text || `HTTP ${resp.status}`);
  }
  const ct = resp.headers.get("content-type") || "";
  if (ct.includes("application/json")) return resp.json();
  return {};
}

async function getVaultName() {
  const cached = await chrome.storage.local.get(["vaultName", "metaFetchedAt"]);
  const age = Date.now() - Number(cached.metaFetchedAt || 0);
  if (cached.vaultName && age < 10 * 60 * 1000) {
    return String(cached.vaultName || "").trim();
  }
  try {
    const meta = await apiFetch("GET", "/v1/meta");
    const vaultName = String(meta?.obsidian_vault_name || "").trim();
    await chrome.storage.local.set({ vaultName, metaFetchedAt: Date.now() });
    return vaultName;
  } catch (_) {
    return String(cached.vaultName || "").trim();
  }
}

function obsidianOpenUrl(vaultName, path) {
  const vault = encodeURIComponent(String(vaultName || "").trim());
  const normalized = String(path || "")
    .replace(/^\/+/, "")
    .replace(/\\/g, "/")
    .replace(/\/+/g, "/");
  const file = encodeURIComponent(normalized);
  if (!vault || !file) return "";
  return `obsidian://open?vault=${vault}&file=${file}`;
}

async function cacheAnchorsForPage(pageKey, rows) {
  const key = String(pageKey || "").trim();
  if (!key) return;
  const { pawnAnchorsByPage = {} } = await chrome.storage.local.get("pawnAnchorsByPage");
  pawnAnchorsByPage[key] = {
    fetchedAt: Date.now(),
    data: Array.isArray(rows) ? rows : [],
  };
  await chrome.storage.local.set({ pawnAnchorsByPage });
}

async function fetchAnchors(pageKey, site) {
  const pk = String(pageKey || "").trim();
  const siteName = String(site || "").trim();
  if (!pk && !siteName) {
    return { ok: false, error: "page_key or site required", data: [] };
  }
  const params = new URLSearchParams();
  if (pk) params.set("page_key", pk);
  if (siteName) params.set("site", siteName);
  const result = await apiFetch("GET", `/v1/captures/anchors?${params.toString()}`);
  const data = Array.isArray(result?.data) ? result.data : [];
  await cacheAnchorsForPage(pk || `__site:${siteName || "teams"}`, data);
  return { ok: true, data };
}

async function notifyTeamsTabs(payload) {
  const tabs = await chrome.tabs.query({
    url: [
      "*://teams.microsoft.com/*",
      "*://*.teams.microsoft.com/*",
      "*://teams.live.com/*",
      "*://teams.cloud.microsoft/*",
    ],
  });
  await Promise.all(
    tabs.map(async (tab) => {
      if (tab.id == null) return;
      try {
        await chrome.tabs.sendMessage(tab.id, payload);
      } catch (_) {
        /* tab without content script */
      }
    }),
  );
}

/** Side panel calls after Save so badges appear without waiting for next poll. */
async function mergeSavedAnchors(anchors) {
  const rows = Array.isArray(anchors) ? anchors : [];
  if (!rows.length) return;
  const { pawnAnchorsByPage = {} } = await chrome.storage.local.get("pawnAnchorsByPage");
  const byPage = new Map();
  for (const row of rows) {
    const pk = String(row.page_key || "").trim() || "__site:teams";
    if (!byPage.has(pk)) byPage.set(pk, []);
    byPage.get(pk).push(row);
  }
  for (const [pk, incoming] of byPage) {
    const prev = Array.isArray(pawnAnchorsByPage[pk]?.data)
      ? pawnAnchorsByPage[pk].data
      : [];
    const index = new Map();
    const merged = [];
    for (const row of [...prev, ...incoming]) {
      const sid = String(row.snippet_id || "").trim();
      const key = sid || `${row.message_id || ""}:${row.path || ""}`;
      if (index.has(key)) {
        merged[index.get(key)] = { ...merged[index.get(key)], ...row };
      } else {
        index.set(key, merged.length);
        merged.push(row);
      }
    }
    pawnAnchorsByPage[pk] = { fetchedAt: Date.now(), data: merged };
  }
  // Also keep a site-wide bucket for when the chat id is not in the URL.
  const siteKey = "__site:teams";
  const prevSite = Array.isArray(pawnAnchorsByPage[siteKey]?.data)
    ? pawnAnchorsByPage[siteKey].data
    : [];
  const siteIndex = new Map(prevSite.map((r, i) => [String(r.snippet_id || ""), i]));
  const siteMerged = [...prevSite];
  for (const row of rows) {
    const sid = String(row.snippet_id || "");
    if (!sid) continue;
    if (siteIndex.has(sid)) siteMerged[siteIndex.get(sid)] = { ...siteMerged[siteIndex.get(sid)], ...row };
    else {
      siteIndex.set(sid, siteMerged.length);
      siteMerged.push(row);
    }
  }
  pawnAnchorsByPage[siteKey] = { fetchedAt: Date.now(), data: siteMerged };
  await chrome.storage.local.set({ pawnAnchorsByPage });
  await notifyTeamsTabs({ type: "pawn-overlays-refresh", anchors: rows });
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "pawn-scrape-outlook") {
    void (async () => {
      try {
        const tab = await activeContentTab();
        if (!tab) {
          sendResponse({ ok: false, error: "No active tab" });
          return;
        }
        if (!isCalendarUrl(tab.url || "")) {
          sendResponse({
            ok: false,
            error: "Open an Outlook or Teams calendar event first",
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
  }

  if (msg?.type === "pawn-anchors-get") {
    void (async () => {
      const pk = String(msg.page_key || "").trim();
      const site = String(msg.site || "teams").trim() || "teams";
      const { pawnAnchorsByPage = {} } = await chrome.storage.local.get("pawnAnchorsByPage");
      const cached =
        pawnAnchorsByPage[pk]?.data ||
        pawnAnchorsByPage[`__site:${site}`]?.data ||
        [];
      try {
        const result = await fetchAnchors(pk, site);
        // Prefer conversation matches; fall back to site-wide cache/API rows.
        let data = result.data || [];
        if (pk && !data.length && Array.isArray(cached) && cached.length) {
          data = cached;
        }
        if (!pk) {
          const siteCached = pawnAnchorsByPage[`__site:${site}`]?.data;
          if (!data.length && Array.isArray(siteCached)) data = siteCached;
        }
        sendResponse({ ok: true, data, cached });
      } catch (err) {
        sendResponse({
          ok: false,
          error: String(err?.message || err),
          cached: Array.isArray(cached) ? cached : [],
          data: Array.isArray(cached) ? cached : [],
        });
      }
    })();
    return true;
  }

  if (msg?.type === "pawn-anchors-refresh") {
    void (async () => {
      try {
        await mergeSavedAnchors(msg.anchors || []);
        sendResponse({ ok: true });
      } catch (err) {
        sendResponse({ ok: false, error: String(err?.message || err) });
      }
    })();
    return true;
  }

  if (msg?.type === "pawn-overlay-open") {
    void (async () => {
      try {
        const vault = await getVaultName();
        if (!vault) {
          sendResponse({
            ok: false,
            error: "Set vault.obsidian_vault_name on the server for obsidian:// links",
          });
          return;
        }
        const url = obsidianOpenUrl(vault, msg.path || "");
        await chrome.tabs.create({ url, active: true });
        sendResponse({ ok: true, url });
      } catch (err) {
        sendResponse({ ok: false, error: String(err?.message || err) });
      }
    })();
    return true;
  }

  return false;
});
