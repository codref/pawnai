/** Side panel: target picker, ordered tray, save to /v1/captures. */

const DEFAULTS = {
  serverUrl: "http://127.0.0.1:8000",
  apiToken: "",
};

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
  status: document.getElementById("status-dot"),
  settingsPanel: document.getElementById("settings-panel"),
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
  await loadSettings();
  await loadTray();
  bind();
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
  });
  el.title.addEventListener("input", () => {
    state.title = el.title.value;
    void persistTarget();
  });
  el.btnSaveSettings.addEventListener("click", async () => {
    state.settings.serverUrl = el.serverUrl.value.trim().replace(/\/+$/, "");
    state.settings.apiToken = el.apiToken.value.trim();
    await chrome.storage.local.set({ settings: state.settings });
    showNotice("Settings saved");
    await refreshConnection();
  });
  el.btnSettings.addEventListener("click", () => {
    state.settingsOpen = !state.settingsOpen;
    el.settingsPanel.hidden = !state.settingsOpen;
    el.btnSettings.classList.toggle("is-active", state.settingsOpen);
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
  if (state.targetKind === "capture") {
    try {
      const data = await api("GET", "/v1/captures?limit=40");
      state.captures = data.data || [];
    } catch (err) {
      state.captures = [];
      showNotice(String(err.message || err), true);
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
  if (state.targetKind === "capture") {
    const sel = document.createElement("select");
    sel.className = "pawn-conv-select";
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = state.captures.length ? "Choose a capture…" : "No captures yet";
    sel.appendChild(blank);
    for (const row of state.captures) {
      const opt = document.createElement("option");
      opt.value = row.path;
      opt.textContent = row.title || row.path;
      if (row.path === state.stickyPath) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.addEventListener("change", async () => {
      state.stickyPath = sel.value;
      const hit = state.captures.find((c) => c.path === sel.value);
      if (hit?.title) {
        state.title = hit.title;
        el.title.value = state.title;
      }
      await persistTarget();
      await loadTray();
      render();
    });
    el.targetExtra.appendChild(sel);
  } else if (state.targetKind === "session") {
    const sel = document.createElement("select");
    sel.className = "pawn-conv-select";
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = state.sessions.length ? "Choose a session…" : "No sessions";
    sel.appendChild(blank);
    for (const row of state.sessions) {
      const opt = document.createElement("option");
      opt.value = row.session_id;
      const label = row.title || row.session_id;
      opt.textContent = row.transcript_path ? `${label}` : `${label} (no transcript yet)`;
      if (row.session_id === state.sessionId) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.addEventListener("change", async () => {
      state.sessionId = sel.value;
      const hit = state.sessions.find((s) => s.session_id === sel.value);
      if (hit) {
        state.title = hit.title || hit.session_id;
        el.title.value = state.title;
        state.stickyPath = hit.transcript_path || "";
      }
      await persistTarget();
      await loadTray();
      render();
    });
    el.targetExtra.appendChild(sel);
  } else if (state.targetKind === "note") {
    const input = document.createElement("input");
    input.type = "text";
    input.className = "pawn-conv-select";
    input.placeholder = "Vault path e.g. Projects/Roadmap.md";
    input.value = state.stickyPath;
    input.addEventListener("change", async () => {
      state.stickyPath = input.value.trim();
      await persistTarget();
      await loadTray();
      render();
    });
    el.targetExtra.appendChild(input);
  }
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
  kind.textContent = snip.kind;
  top.appendChild(kind);

  const body = document.createElement("div");
  body.className = "pawn-item-body";
  if (snip.kind === "image" && snip.data_base64) {
    const img = document.createElement("img");
    img.src = `data:${snip.media_type || "image/png"};base64,${snip.data_base64}`;
    img.alt = "Snippet";
    body.appendChild(img);
  } else {
    body.textContent = (snip.text || "").slice(0, 400) || "(empty)";
  }
  const meta = document.createElement("div");
  meta.className = "pawn-item-meta";
  const when = snip.captured_at ? new Date(snip.captured_at).toLocaleString() : "";
  meta.textContent = [snip.saved ? "saved" : "unsaved", when, snip.source_url || ""]
    .filter(Boolean)
    .join(" · ");
  body.appendChild(meta);
  top.appendChild(body);

  const actions = document.createElement("div");
  actions.className = "pawn-item-actions";
  const up = document.createElement("button");
  up.type = "button";
  up.textContent = "↑";
  up.disabled = index === 0 || snip.saved;
  up.onclick = () => void moveSnippet(index, -1);
  const down = document.createElement("button");
  down.type = "button";
  down.textContent = "↓";
  down.disabled = index >= state.tray.length - 1 || snip.saved;
  down.onclick = () => void moveSnippet(index, 1);
  const del = document.createElement("button");
  del.type = "button";
  del.textContent = "✕";
  del.onclick = () => void removeSnippet(index);
  actions.append(up, down, del);
  top.appendChild(actions);

  card.appendChild(top);
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
    return;
  }
  if ((state.targetKind === "capture" || state.targetKind === "note") && !state.stickyPath) {
    showNotice("Pick or enter a note path first", true);
    return;
  }
  el.btnSave.disabled = true;
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
    if (state.targetKind === "new") {
      // Stay on sticky path via kind=new + path (server appends)
    }
    const written = new Set(result.written || []);
    const skipped = new Set(result.skipped || []);
    for (const s of state.tray) {
      if (written.has(s.id) || skipped.has(s.id)) s.saved = true;
    }
    await persistTarget();
    await persistTray();
    showNotice(`Saved ${written.size} snippet(s) → ${result.path}`);
    render();
  } catch (err) {
    showNotice(String(err.message || err), true);
    el.btnSave.disabled = false;
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

function showNotice(text, isError = false) {
  el.notice.hidden = !text;
  el.notice.textContent = text || "";
  el.notice.classList.toggle("is-error", !!isError);
}
