/** Side panel: target picker, ordered tray, save to /v1/captures. */

const DEFAULTS = {
  serverUrl: "http://127.0.0.1:8000",
  apiToken: "",
};

/** Lucide-style outlines (viewBox 0 0 24 24), matching Obsidian clickable-icon usage. */
const ICONS = {
  settings:
    '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
  "text-select":
    '<path d="M7 8h10"/><path d="M7 12h8"/><path d="M7 16h6"/><path d="M3 4h1"/><path d="M3 20h1"/><path d="M20 4h1"/><path d="M20 20h1"/><path d="M4 3v1"/><path d="M4 20v1"/><path d="M20 3v1"/><path d="M20 20v1"/>',
  crop: '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
  save: '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/>',
  "file-plus":
    '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M9 15h6"/><path d="M12 12v6"/>',
  type: '<polyline points="4 7 4 4 20 4 20 7"/><line x1="9" x2="15" y1="20" y2="20"/><line x1="12" x2="12" y1="4" y2="20"/>',
  image:
    '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
  "chevron-up": '<path d="m18 15-6-6-6 6"/>',
  "chevron-down": '<path d="m6 9 6 6 6-6"/>',
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
};

function setIcon(node, name) {
  const body = ICONS[name];
  if (!body) return;
  node.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" aria-hidden="true">${body}</svg>`;
}

function iconButton(icon, label, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "clickable-icon";
  b.setAttribute("aria-label", label);
  b.title = label;
  setIcon(b, icon);
  b.onclick = onClick;
  return b;
}

const state = {
  settings: { ...DEFAULTS },
  targetKind: "new",
  stickyPath: "",
  sessionId: "",
  title: "",
  sourceUrl: "",
  captures: [],
  sessions: [],
  tray: [],
  settingsOpen: false,
};

const el = {
  targetKind: document.getElementById("target-kind"),
  targetExtra: document.getElementById("target-extra"),
  title: document.getElementById("title-input"),
  titleField: document.getElementById("title-field"),
  status: document.getElementById("status-dot"),
  settingsPanel: document.getElementById("settings-panel"),
  body: document.querySelector(".pawn-body"),
  serverUrl: document.getElementById("server-url"),
  apiToken: document.getElementById("api-token"),
  btnSaveSettings: document.getElementById("btn-save-settings"),
  btnSelection: document.getElementById("btn-selection"),
  btnRegion: document.getElementById("btn-region"),
  btnSettings: document.getElementById("btn-settings"),
  btnSave: document.getElementById("btn-save"),
  btnNewPage: document.getElementById("btn-new-page"),
  pathLabel: document.getElementById("path-label"),
  notice: document.getElementById("notice"),
  tray: document.getElementById("tray"),
  empty: document.getElementById("empty"),
};

init();

async function init() {
  mountChromeIcons();
  await loadSettings();
  await loadTray();
  bind();
  syncSettingsUi();
  await refreshConnection();
  await drainPending();
  render();
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;
    if (changes.pendingSnippets || changes.pendingNotice) {
      void drainPending();
    }
  });
}

function mountChromeIcons() {
  setIcon(el.btnSettings, "settings");
  setIcon(el.btnSelection, "text-select");
  setIcon(el.btnRegion, "crop");
  setIcon(el.btnNewPage, "file-plus");
}

function syncSettingsUi() {
  el.settingsPanel.hidden = !state.settingsOpen;
  el.btnSettings.classList.toggle("is-active", state.settingsOpen);
  el.btnSettings.setAttribute("aria-pressed", state.settingsOpen ? "true" : "false");
  if (el.titleField) el.titleField.hidden = state.settingsOpen;
  if (el.targetExtra) el.targetExtra.hidden = state.settingsOpen;
  if (el.body) el.body.hidden = state.settingsOpen;
}

function bind() {
  el.targetKind.addEventListener("change", async () => {
    state.targetKind = el.targetKind.value;
    if (state.targetKind === "new") {
      state.stickyPath = "";
      state.sessionId = "";
    }
    await persistTarget();
    await loadPickerData();
    renderTargetExtra();
    render();
    if (state.targetKind !== "new") {
      void openTargetPicker();
    }
  });
  el.title.addEventListener("input", () => {
    state.title = el.title.value;
    void persistTarget();
  });
  el.btnSaveSettings.addEventListener("click", () => void saveSettings());
  el.btnSettings.addEventListener("click", () => {
    state.settingsOpen = !state.settingsOpen;
    syncSettingsUi();
  });
  el.btnSelection.addEventListener("click", () => {
    void requestSnip("pawn-snip-selection");
  });
  el.btnRegion.addEventListener("click", () => {
    void requestSnip("pawn-snip-region");
  });
  el.btnSave.addEventListener("click", () => void saveTray());
  el.btnNewPage.addEventListener("click", () => void startNewPage());
}

async function saveSettings() {
  const btn = el.btnSaveSettings;
  const prev = btn.textContent;
  btn.disabled = true;
  btn.classList.add("is-busy");
  btn.textContent = "Saving…";
  showNotice("Saving settings…", false, true);
  try {
    state.settings.serverUrl = el.serverUrl.value.trim().replace(/\/+$/, "");
    state.settings.apiToken = el.apiToken.value.trim();
    await chrome.storage.local.set({ settings: state.settings });
    await refreshConnection();
    showNotice("Settings saved");
  } catch (err) {
    showNotice(String(err.message || err), true);
  } finally {
    btn.disabled = false;
    btn.classList.remove("is-busy");
    btn.textContent = prev;
  }
}

async function loadSettings() {
  const stored = await chrome.storage.local.get(["settings", "target"]);
  state.settings = { ...DEFAULTS, ...(stored.settings || {}) };
  el.serverUrl.value = state.settings.serverUrl;
  el.apiToken.value = state.settings.apiToken;
  const t = stored.target || {};
  state.targetKind = t.kind || "new";
  state.stickyPath = t.path || "";
  state.sessionId = t.sessionId || "";
  state.title = t.title || "";
  state.sourceUrl = t.sourceUrl || "";
  el.targetKind.value = state.targetKind;
  el.title.value = state.title;
  await loadPickerData();
  renderTargetExtra();
}

async function persistTarget() {
  await chrome.storage.local.set({
    target: {
      kind: state.targetKind,
      path: state.stickyPath,
      sessionId: state.sessionId,
      title: state.title,
      sourceUrl: state.sourceUrl,
    },
  });
}

async function loadTray() {
  const key = trayKey();
  const data = await chrome.storage.local.get(key);
  state.tray = Array.isArray(data[key]) ? data[key] : [];
}

async function persistTray() {
  await chrome.storage.local.set({ [trayKey()]: state.tray });
}

function trayKey() {
  if (state.targetKind === "session" && state.sessionId) {
    return `tray:session:${state.sessionId}`;
  }
  if (state.stickyPath) return `tray:path:${state.stickyPath}`;
  return "tray:new";
}

async function drainPending() {
  const data = await chrome.storage.local.get([
    "pendingSnippets",
    "pendingNotice",
    "lastPageTitle",
    "lastPageUrl",
  ]);
  if (data.pendingNotice?.error) {
    showNotice(data.pendingNotice.error, true);
    await chrome.storage.local.remove("pendingNotice");
  }
  const pending = data.pendingSnippets || [];
  if (!pending.length) {
    render();
    return;
  }
  if (!state.title && data.lastPageTitle) {
    state.title = data.lastPageTitle;
    el.title.value = state.title;
  }
  if (data.lastPageUrl) state.sourceUrl = data.lastPageUrl;
  for (const snip of pending) {
    if (state.tray.some((s) => s.id === snip.id)) continue;
    state.tray.push({ ...snip, saved: false });
  }
  await chrome.storage.local.set({ pendingSnippets: [] });
  await persistTray();
  await persistTarget();
  render();
}

function requestSnip(type) {
  chrome.runtime.sendMessage({ type }, (resp) => {
    if (chrome.runtime.lastError) {
      showNotice(chrome.runtime.lastError.message, true);
      return;
    }
    if (resp && resp.ok === false) showNotice(resp.error || "Capture failed", true);
  });
}

async function api(method, path, body) {
  const base = state.settings.serverUrl.replace(/\/+$/, "");
  const headers = { Accept: "application/json" };
  if (state.settings.apiToken) {
    headers.Authorization = `Bearer ${state.settings.apiToken}`;
  }
  const opts = { method, headers };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(`${base}${path}`, opts);
  const text = await resp.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (_) {
    data = { detail: text };
  }
  if (!resp.ok) {
    const detail = data?.detail || data?.message || text || resp.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

async function refreshConnection() {
  try {
    await api("GET", "/health");
    el.status.classList.add("is-online");
    el.status.classList.remove("is-offline");
    el.status.title = "Online";
  } catch (_) {
    try {
      // /health may be unauthenticated; try captures list with token
      await api("GET", "/v1/captures?limit=1");
      el.status.classList.add("is-online");
      el.status.classList.remove("is-offline");
      el.status.title = "Online";
    } catch (err) {
      el.status.classList.remove("is-online");
      el.status.classList.add("is-offline");
      el.status.title = String(err.message || "Offline");
    }
  }
}

async function loadPickerData() {
  if (state.targetKind === "capture" || state.targetKind === "note") {
    try {
      const data = await api("GET", "/v1/captures?limit=40");
      state.captures = data.data || [];
    } catch (err) {
      state.captures = [];
      if (state.targetKind === "capture") {
        showNotice(String(err.message || err), true);
      }
    }
  }
  if (state.targetKind === "session") {
    try {
      const data = await api("GET", "/v1/sessions?limit=30");
      state.sessions = data.data || [];
    } catch (err) {
      state.sessions = [];
      showNotice(String(err.message || err), true);
    }
  }
}

function renderTargetExtra() {
  el.targetExtra.innerHTML = "";
  if (state.targetKind === "new") return;

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "pawn-pick-trigger";
  const label = document.createElement("span");
  label.className = "pawn-pick-label";
  const current = currentPickLabel();
  if (current) {
    label.textContent = current;
  } else {
    label.textContent = pickPlaceholder();
    label.classList.add("is-placeholder");
  }
  const icon = document.createElement("span");
  icon.className = "clickable-icon";
  icon.setAttribute("aria-hidden", "true");
  setIcon(icon, "search");
  trigger.append(label, icon);
  trigger.addEventListener("click", () => void openTargetPicker());
  el.targetExtra.appendChild(trigger);
}

function pickPlaceholder() {
  if (state.targetKind === "capture") return "Choose a capture…";
  if (state.targetKind === "session") return "Choose a session…";
  return "Choose a note path…";
}

function currentPickLabel() {
  if (state.targetKind === "capture") {
    if (!state.stickyPath) return "";
    const hit = state.captures.find((c) => c.path === state.stickyPath);
    return hit?.title || state.stickyPath;
  }
  if (state.targetKind === "session") {
    if (!state.sessionId) return "";
    const hit = state.sessions.find((s) => s.session_id === state.sessionId);
    return hit?.title || state.sessionId;
  }
  if (state.targetKind === "note") return state.stickyPath || "";
  return "";
}

async function openTargetPicker() {
  if (state.targetKind === "new") return;
  await loadPickerData();
  if (state.targetKind === "capture") {
    const items = state.captures.map((row) => ({
      id: row.path,
      title: row.title || row.path,
      detail: row.path,
      value: row.path,
    }));
    openSuggestModal({
      placeholder: "Find a capture…",
      items,
      emptyText: "No captures yet",
      onChoose: async (item) => {
        state.stickyPath = item.value;
        const hit = state.captures.find((c) => c.path === item.value);
        if (hit?.title) {
          state.title = hit.title;
          el.title.value = state.title;
        }
        await persistTarget();
        await loadTray();
        renderTargetExtra();
        render();
      },
    });
    return;
  }
  if (state.targetKind === "session") {
    const items = state.sessions.map((row) => ({
      id: row.session_id,
      title: row.title || row.session_id,
      detail: row.transcript_path || "no transcript yet",
      value: row.session_id,
    }));
    openSuggestModal({
      placeholder: "Find a session…",
      items,
      emptyText: "No sessions",
      onChoose: async (item) => {
        state.sessionId = item.value;
        const hit = state.sessions.find((s) => s.session_id === item.value);
        if (hit) {
          state.title = hit.title || hit.session_id;
          el.title.value = state.title;
          state.stickyPath = hit.transcript_path || "";
        }
        await persistTarget();
        await loadTray();
        renderTargetExtra();
        render();
      },
    });
    return;
  }
  if (state.targetKind === "note") {
    // No vault-wide note index on the capture API — suggest recent captures + free path.
    const items = state.captures.map((row) => ({
      id: row.path,
      title: row.title || row.path,
      detail: row.path,
      value: row.path,
    }));
    openSuggestModal({
      placeholder: "Note path (type to filter or enter a vault path)…",
      items,
      emptyText: "Type a vault path and press Enter",
      allowCustom: true,
      initialQuery: state.stickyPath || "",
      onChoose: async (item) => {
        state.stickyPath = item.value.trim();
        await persistTarget();
        await loadTray();
        renderTargetExtra();
        render();
      },
    });
  }
}

/** Obsidian-style fuzzy suggest overlay. */
function openSuggestModal(opts) {
  const existing = document.querySelector(".pawn-suggest");
  if (existing) existing.remove();

  const overlay = document.createElement("div");
  overlay.className = "pawn-suggest";
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");

  const card = document.createElement("div");
  card.className = "pawn-suggest-card";
  const inputRow = document.createElement("div");
  inputRow.className = "pawn-suggest-input-row";
  const input = document.createElement("input");
  input.type = "search";
  input.placeholder = opts.placeholder || "Filter…";
  input.autocomplete = "off";
  input.spellcheck = false;
  if (opts.initialQuery) input.value = opts.initialQuery;
  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "clickable-icon";
  clearBtn.setAttribute("aria-label", "Close");
  setIcon(clearBtn, "x");
  inputRow.append(input, clearBtn);

  const list = document.createElement("div");
  list.className = "pawn-suggest-list";
  card.append(inputRow, list);
  overlay.appendChild(card);
  document.body.appendChild(overlay);

  let active = 0;
  let filtered = [];

  const close = () => {
    overlay.remove();
    document.removeEventListener("keydown", onDocKey, true);
  };

  const choose = async (item) => {
    close();
    await opts.onChoose(item);
  };

  const renderList = () => {
    const q = input.value.trim();
    filtered = rankSuggestItems(opts.items || [], q);
    if (!filtered.length && opts.allowCustom && q) {
      filtered = [{ id: `custom:${q}`, title: q, detail: "Use this path", value: q, custom: true }];
    }
    list.innerHTML = "";
    if (!filtered.length) {
      const empty = document.createElement("div");
      empty.className = "pawn-suggest-empty";
      empty.textContent = opts.emptyText || "No matches";
      list.appendChild(empty);
      return;
    }
    if (active >= filtered.length) active = filtered.length - 1;
    if (active < 0) active = 0;
    filtered.forEach((item, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pawn-suggest-item" + (i === active ? " is-active" : "");
      const title = document.createElement("div");
      title.className = "pawn-suggest-item-title";
      title.textContent = item.title;
      btn.appendChild(title);
      if (item.detail) {
        const detail = document.createElement("div");
        detail.className = "pawn-suggest-item-detail";
        detail.textContent = item.detail;
        btn.appendChild(detail);
      }
      btn.addEventListener("mouseenter", () => {
        active = i;
        renderList();
      });
      btn.addEventListener("click", () => void choose(item));
      list.appendChild(btn);
    });
    const activeEl = list.querySelector(".is-active");
    if (activeEl) activeEl.scrollIntoView({ block: "nearest" });
  };

  const onDocKey = (ev) => {
    if (ev.key === "Escape") {
      ev.preventDefault();
      close();
    }
  };

  clearBtn.addEventListener("click", close);
  overlay.addEventListener("click", (ev) => {
    if (ev.target === overlay) close();
  });
  input.addEventListener("input", () => {
    active = 0;
    renderList();
  });
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowDown") {
      ev.preventDefault();
      active = Math.min(active + 1, Math.max(filtered.length - 1, 0));
      renderList();
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      active = Math.max(active - 1, 0);
      renderList();
    } else if (ev.key === "Enter") {
      ev.preventDefault();
      if (filtered[active]) void choose(filtered[active]);
      else if (opts.allowCustom && input.value.trim()) {
        void choose({
          id: `custom:${input.value.trim()}`,
          title: input.value.trim(),
          value: input.value.trim(),
          custom: true,
        });
      }
    } else if (ev.key === "Escape") {
      ev.preventDefault();
      close();
    }
  });
  document.addEventListener("keydown", onDocKey, true);
  renderList();
  requestAnimationFrame(() => input.focus());
}

function rankSuggestItems(items, query) {
  if (!query) return items.slice();
  const q = query.toLowerCase();
  const scored = [];
  for (const item of items) {
    const title = String(item.title || "");
    const detail = String(item.detail || "");
    const hay = `${title} ${detail}`.toLowerCase();
    let score = 0;
    if (hay.includes(q)) score += 10;
    if (title.toLowerCase().startsWith(q)) score += 5;
    if (detail.toLowerCase().includes(q)) score += 2;
    if (!score && fuzzySubsequence(q, hay)) score += 1;
    if (score) scored.push({ item, score, title });
  }
  scored.sort((a, b) => b.score - a.score || a.title.localeCompare(b.title));
  return scored.map((s) => s.item);
}

function fuzzySubsequence(query, text) {
  let i = 0;
  for (const ch of text) {
    if (ch === query[i]) i += 1;
    if (i >= query.length) return true;
  }
  return false;
}

function render() {
  el.pathLabel.textContent = state.stickyPath
    ? state.stickyPath
    : state.targetKind === "session" && state.sessionId
      ? `session:${state.sessionId}`
      : "New capture on save";
  el.tray.innerHTML = "";
  el.empty.hidden = state.tray.length > 0;
  state.tray.forEach((snip, index) => {
    el.tray.appendChild(renderCard(snip, index));
  });
  const unsaved = state.tray.some((s) => !s.saved);
  el.btnSave.disabled = !unsaved;
}

function renderCard(snip, index) {
  const card = document.createElement("article");
  card.className = "pawn-item" + (snip.saved ? " is-saved" : "");
  const top = document.createElement("div");
  top.className = "pawn-item-top";

  const kind = document.createElement("span");
  kind.className = `pawn-item-kind is-kind-${snip.kind}`;
  kind.title = snip.kind === "image" ? "Image" : "Text";
  setIcon(kind, snip.kind === "image" ? "image" : "type");
  top.appendChild(kind);

  const body = document.createElement("div");
  body.className = "pawn-item-body";
  if (snip.kind === "image" && snip.data_base64) {
    const img = document.createElement("img");
    img.src = `data:${snip.media_type || "image/png"};base64,${snip.data_base64}`;
    img.alt = "Snippet";
    body.appendChild(img);
  } else {
    const text = document.createElement("div");
    text.className = "pawn-item-text";
    text.textContent = (snip.text || "").slice(0, 400) || "(empty)";
    body.appendChild(text);
  }
  top.appendChild(body);

  const metaRow = document.createElement("div");
  metaRow.className = "pawn-item-meta-row";
  const meta = document.createElement("div");
  meta.className = "pawn-item-meta";
  const when = snip.captured_at ? new Date(snip.captured_at).toLocaleString() : "";
  meta.textContent = [snip.saved ? "saved" : "unsaved", when, snip.source_url || ""]
    .filter(Boolean)
    .join(" · ");
  const actions = document.createElement("div");
  actions.className = "pawn-item-actions";
  const up = iconButton("chevron-up", "Move up", () => void moveSnippet(index, -1));
  up.disabled = index === 0 || snip.saved;
  const down = iconButton("chevron-down", "Move down", () => void moveSnippet(index, 1));
  down.disabled = index >= state.tray.length - 1 || snip.saved;
  const del = iconButton("x", "Remove", () => void removeSnippet(index));
  actions.append(up, down, del);
  metaRow.append(meta, actions);

  card.append(top, metaRow);
  return card;
}

async function moveSnippet(index, delta) {
  const next = index + delta;
  if (next < 0 || next >= state.tray.length) return;
  if (state.tray[index].saved || state.tray[next].saved) return;
  const tmp = state.tray[index];
  state.tray[index] = state.tray[next];
  state.tray[next] = tmp;
  await persistTray();
  render();
}

async function removeSnippet(index) {
  const snip = state.tray[index];
  if (snip.saved && state.stickyPath) {
    try {
      await api(
        "DELETE",
        `/v1/captures/snippets/${encodeURIComponent(snip.id)}?path=${encodeURIComponent(state.stickyPath)}`,
      );
    } catch (err) {
      showNotice(String(err.message || err), true);
      return;
    }
  }
  state.tray.splice(index, 1);
  await persistTray();
  render();
}

function buildTarget() {
  if (state.targetKind === "session") {
    return {
      kind: "session",
      session_id: state.sessionId,
      title: state.title,
      source_url: state.sourceUrl,
      path: state.stickyPath || undefined,
    };
  }
  if (state.targetKind === "note") {
    return {
      kind: "note",
      path: state.stickyPath,
      title: state.title,
      source_url: state.sourceUrl,
    };
  }
  if (state.targetKind === "capture") {
    return {
      kind: "capture",
      path: state.stickyPath,
      title: state.title,
      source_url: state.sourceUrl,
    };
  }
  // new — sticky path after first save continues the same page
  return {
    kind: "new",
    path: state.stickyPath || undefined,
    title: state.title || "capture",
    source_url: state.sourceUrl,
  };
}

async function saveTray() {
  const unsaved = state.tray.filter((s) => !s.saved);
  if (!unsaved.length) return;
  if (state.targetKind === "session" && !state.sessionId) {
    showNotice("Pick a session first", true);
    void openTargetPicker();
    return;
  }
  if ((state.targetKind === "capture" || state.targetKind === "note") && !state.stickyPath) {
    showNotice("Pick or enter a note path first", true);
    void openTargetPicker();
    return;
  }
  const btn = el.btnSave;
  const prev = btn.textContent;
  btn.disabled = true;
  btn.classList.add("is-busy");
  btn.textContent = "Saving…";
  showNotice(`Saving ${unsaved.length} snippet(s)…`, false, true);
  try {
    const payload = {
      target: buildTarget(),
      snippets: unsaved.map((s) => ({
        id: s.id,
        kind: s.kind,
        text: s.text || "",
        data_base64: s.data_base64 || undefined,
        media_type: s.media_type || undefined,
        source_url: s.source_url || state.sourceUrl,
        captured_at: s.captured_at,
      })),
    };
    const result = await api("POST", "/v1/captures", payload);
    state.stickyPath = result.path;
    const written = new Set(result.written || []);
    const skipped = new Set(result.skipped || []);
    for (const s of state.tray) {
      if (written.has(s.id) || skipped.has(s.id)) s.saved = true;
    }
    await persistTarget();
    await persistTray();
    const mode = result.mode === "annotations" ? "annotations" : "note";
    showNotice(`Saved ${written.size} → ${result.path} (${mode})`);
    render();
  } catch (err) {
    showNotice(String(err.message || err), true);
  } finally {
    btn.classList.remove("is-busy");
    btn.textContent = prev;
    const still = state.tray.some((s) => !s.saved);
    btn.disabled = !still;
  }
}

async function startNewPage() {
  state.targetKind = "new";
  state.stickyPath = "";
  state.sessionId = "";
  state.title = "";
  el.targetKind.value = "new";
  el.title.value = "";
  state.tray = [];
  await persistTarget();
  await persistTray();
  renderTargetExtra();
  render();
  showNotice("New page — next Save creates a fresh capture note");
}

function showNotice(text, isError = false, isBusy = false) {
  el.notice.hidden = !text;
  el.notice.textContent = text || "";
  el.notice.classList.toggle("is-error", !!isError);
  el.notice.classList.toggle("is-busy", !!isBusy && !isError);
}
