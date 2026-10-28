/** Side panel: Capture tray + Chat over /v1/pawn/chat. */

const DEFAULTS = {
  serverUrl: "http://127.0.0.1:8000",
  apiToken: "",
};

const MAX_CHAT_IMAGES = 4;
const CHAT_MSG_LIMIT = 80;

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
  "chevron-right": '<path d="m9 18 6-6-6-6"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
  monitor:
    '<rect width="20" height="14" x="2" y="3" rx="2" ry="2"/><line x1="8" x2="16" y1="21" y2="21"/><line x1="12" x2="12" y1="17" y2="21"/>',
  "message-square":
    '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
  send: '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
  square: '<rect width="18" height="18" x="3" y="3" rx="2"/>',
  "plus-circle":
    '<circle cx="12" cy="12" r="10"/><path d="M8 12h8"/><path d="M12 8v8"/>',
  plus: '<path d="M5 12h14"/><path d="M12 5v14"/>',
  trash:
    '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>',
  link: '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
  paperclip: '<path d="m16 6-8.5 8.5a2.12 2.12 0 0 0 3 3L19 9"/><path d="m15 3 6 6"/>',
  "file-text":
    '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
  mic: '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>',
  copy: '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
  pencil: '<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/>',
  "external-link":
    '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
  "list-todo":
    '<rect width="6" height="6" x="3" y="5" rx="1"/><path d="m3 17 2 2 4-4"/><path d="M13 6h8"/><path d="M13 12h8"/><path d="M13 18h8"/>',
};

const REASONING_LABEL = { none: "Off", low: "Low", medium: "Medium", high: "High" };
const ROUTE_LABEL = {
  balanced: "Balanced",
  nitro: "Nitro",
  floor: "Floor",
  exacto: "Exacto",
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

const CLIP_MODES = [
  { value: "region", label: "Region", icon: "crop", snip: "pawn-snip-region" },
  { value: "visible", label: "Visible area", icon: "monitor", snip: "pawn-snip-visible" },
];

const state = {
  settings: { ...DEFAULTS },
  activeTab: "capture",
  targetKind: "new",
  stickyPath: "",
  sessionId: "",
  title: "",
  sourceUrl: "",
  collection: "",
  hint: "",
  eventStart: "",
  eventEnd: "",
  eventAttendees: [],
  eventOrganizer: "",
  eventLocation: "",
  eventMeetingUrl: "",
  vaultName: "",
  clipMode: "region",
  clipBusy: false,
  /** "tray" | "chat" — where the next capture lands. */
  captureDest: "tray",
  captures: [],
  sessions: [],
  collections: [],
  tray: [],
  settingsOpen: false,
  chat: {
    conversationId: "",
    /** @type {Record<string, {id:string,title:string,messages:any[],updatedAt:number,model?:string,reasoning?:string,route?:string}>} */
    conversations: {},
    messages: [],
    attachments: [],
    /** Vault notes / sessions attached as chat context (Obsidian @ notes). */
    noteContext: [],
    /** When true, Capture sticky/session is not auto-included. */
    omitBound: false,
    models: [],
    model: "",
    reasoning: "",
    route: "",
    abort: null,
    pageUrl: "",
    pageTitle: "",
  },
};

const MAX_NOTE_CONTEXT = 8;

const MAX_CHAT_CONVERSATIONS = 40;

const TARGET_KIND_OPTIONS = [
  { value: "new", label: "New page" },
  { value: "event", label: "Planner event…" },
  { value: "research", label: "Research…" },
  { value: "capture", label: "Recent capture…" },
  { value: "session", label: "Session…" },
  { value: "note", label: "Other note…" },
];

const el = {
  targetKind: null,
  targetExtra: document.getElementById("target-extra"),
  title: document.getElementById("title-input"),
  titleField: document.getElementById("title-field"),
  status: document.getElementById("status-dot"),
  settingsPanel: document.getElementById("settings-panel"),
  panelCapture: document.getElementById("panel-capture"),
  panelChat: document.getElementById("panel-chat"),
  tabCapture: document.getElementById("tab-capture"),
  tabChat: document.getElementById("tab-chat"),
  body: document.querySelector("#panel-capture .pawn-body"),
  serverUrl: document.getElementById("server-url"),
  apiToken: document.getElementById("api-token"),
  btnSaveSettings: document.getElementById("btn-save-settings"),
  btnSelection: document.getElementById("btn-selection"),
  btnRegion: document.getElementById("btn-region"),
  btnClipMenu: document.getElementById("btn-clip-menu"),
  clipControl: document.getElementById("clip-control"),
  btnSettings: document.getElementById("btn-settings"),
  extVersion: document.getElementById("ext-version"),
  btnSave: document.getElementById("btn-save"),
  btnAddPlanner: document.getElementById("btn-add-planner"),
  destination: document.getElementById("destination"),
  notice: document.getElementById("notice"),
  tray: document.getElementById("tray"),
  empty: document.getElementById("empty"),
  btnChatSelection: document.getElementById("btn-chat-selection"),
  btnChatRegion: document.getElementById("btn-chat-region"),
  btnChatClipMenu: document.getElementById("btn-chat-clip-menu"),
  chatClipControl: document.getElementById("chat-clip-control"),
  btnChatNew: document.getElementById("btn-chat-new"),
  chatConvHost: document.getElementById("chat-conv"),
  chatModelHost: document.getElementById("chat-model"),
  chatContext: document.getElementById("chat-context"),
  chatThread: document.getElementById("chat-thread"),
  chatActions: document.getElementById("chat-actions"),
  chatInput: document.getElementById("chat-input"),
  btnChatSend: document.getElementById("btn-chat-send"),
  btnChatStop: document.getElementById("btn-chat-stop"),
};

/** Fixed-position clip mode popover (Capture or Chat). */
let clipMenuPop = null;

let chatModelSelect = null;
let chatConvPicker = null;

init();

async function init() {
  mountChromeIcons();
  showExtVersion();
  el.targetKind = mountTargetKindSelect(document.getElementById("target-kind"));
  chatConvPicker = mountConversationPicker(el.chatConvHost);
  chatModelSelect = mountModelSelect(el.chatModelHost);
  await loadSettings();
  await loadTray();
  await loadChatState();
  chatConvPicker?.refresh();
  chatModelSelect?.refresh();
  bind();
  syncSettingsUi();
  syncClipUi();
  setActiveTab(state.activeTab, { persist: false });
  await refreshConnection();
  await drainPending();
  render();
  renderChat();
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;
    if (changes.pendingSnippets || changes.pendingNotice) {
      void drainPending();
    }
  });
}

/** Manifest version of the build actually loaded by Chrome/Edge/Firefox. */
function showExtVersion() {
  let ver = "";
  try {
    ver = chrome.runtime.getManifest()?.version || "";
  } catch (_) {
    ver = "";
  }
  if (el.extVersion) {
    el.extVersion.textContent = ver ? `Extension v${ver}` : "";
  }
  if (el.btnSettings && ver) {
    el.btnSettings.title = `Settings · v${ver}`;
  }
  document.title = ver ? `Pawn Capture v${ver}` : "Pawn Capture";
}

/**
 * DOM combobox instead of native <select>.
 * Edge/Chromium side panels often never show the OS select popup (no console error).
 */
function mountTargetKindSelect(host) {
  if (!host) {
    return {
      value: "new",
      addEventListener() {},
    };
  }

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "pawn-select-trigger";
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");
  trigger.title = host.getAttribute("title") || "Target";

  const valueEl = document.createElement("span");
  valueEl.className = "pawn-select-value";
  const chevron = document.createElement("span");
  chevron.className = "clickable-icon";
  chevron.setAttribute("aria-hidden", "true");
  setIcon(chevron, "chevron-down");
  trigger.append(valueEl, chevron);

  const menu = document.createElement("div");
  menu.className = "pawn-select-menu";
  menu.hidden = true;
  menu.setAttribute("role", "listbox");

  for (const opt of TARGET_KIND_OPTIONS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pawn-select-option";
    btn.setAttribute("role", "option");
    btn.dataset.value = opt.value;
    btn.textContent = opt.label;
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      setValue(opt.value, true);
      closeMenu();
    });
    menu.appendChild(btn);
  }

  host.replaceChildren(trigger, menu);

  let current = "new";
  let onChange = null;
  let activeIndex = 0;

  const syncLabel = () => {
    const hit = TARGET_KIND_OPTIONS.find((o) => o.value === current);
    valueEl.textContent = hit?.label || current;
    const options = [...menu.querySelectorAll(".pawn-select-option")];
    options.forEach((btn, i) => {
      const selected = btn.dataset.value === current;
      btn.classList.toggle("is-selected", selected);
      btn.classList.toggle("is-active", i === activeIndex);
      btn.setAttribute("aria-selected", selected ? "true" : "false");
    });
  };

  const closeMenu = () => {
    if (menu.hidden) return;
    menu.hidden = true;
    trigger.setAttribute("aria-expanded", "false");
    setIcon(chevron, "chevron-down");
    document.removeEventListener("pointerdown", onOutside, true);
    document.removeEventListener("keydown", onDocKey, true);
  };

  const openMenu = () => {
    if (!menu.hidden) return;
    activeIndex = Math.max(
      0,
      TARGET_KIND_OPTIONS.findIndex((o) => o.value === current),
    );
    syncLabel();
    menu.hidden = false;
    trigger.setAttribute("aria-expanded", "true");
    setIcon(chevron, "chevron-up");
    document.addEventListener("pointerdown", onOutside, true);
    document.addEventListener("keydown", onDocKey, true);
  };

  const onOutside = (ev) => {
    if (!host.contains(ev.target)) closeMenu();
  };

  const onDocKey = (ev) => {
    if (menu.hidden) return;
    const options = TARGET_KIND_OPTIONS;
    if (ev.key === "Escape") {
      ev.preventDefault();
      closeMenu();
      return;
    }
    if (ev.key === "ArrowDown") {
      ev.preventDefault();
      activeIndex = Math.min(activeIndex + 1, options.length - 1);
      syncLabel();
      return;
    }
    if (ev.key === "ArrowUp") {
      ev.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
      syncLabel();
      return;
    }
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      const opt = options[activeIndex];
      if (opt) {
        setValue(opt.value, true);
        closeMenu();
      }
    }
  };

  const setValue = (v, fire) => {
    const next = TARGET_KIND_OPTIONS.some((o) => o.value === v) ? v : "new";
    const changed = current !== next;
    current = next;
    syncLabel();
    if (fire && changed && typeof onChange === "function") onChange();
  };

  trigger.addEventListener("click", (ev) => {
    ev.preventDefault();
    if (menu.hidden) openMenu();
    else closeMenu();
  });

  syncLabel();

  return {
    get value() {
      return current;
    },
    set value(v) {
      setValue(v, false);
    },
    addEventListener(type, fn) {
      if (type === "change") onChange = fn;
    },
  };
}

function mountChromeIcons() {
  setIcon(el.btnSettings, "settings");
  setIcon(el.btnSelection, "text-select");
  setIcon(el.btnClipMenu, "chevron-down");
  if (el.btnChatSelection) setIcon(el.btnChatSelection, "text-select");
  if (el.btnChatClipMenu) setIcon(el.btnChatClipMenu, "chevron-down");
  if (el.btnChatNew) setIcon(el.btnChatNew, "plus");
  if (el.btnChatStop) setIcon(el.btnChatStop, "square");
  syncClipUi();
}

function clipModeInfo(mode) {
  return CLIP_MODES.find((m) => m.value === mode) || CLIP_MODES[0];
}

function syncClipUi() {
  const info = clipModeInfo(state.clipMode);
  for (const btn of [el.btnRegion, el.btnChatRegion]) {
    if (!btn) continue;
    setIcon(btn, info.icon);
    btn.setAttribute("aria-label", info.label);
    btn.title = info.label;
    btn.disabled = state.clipBusy;
  }
  for (const btn of [el.btnClipMenu, el.btnChatClipMenu]) {
    if (btn) btn.disabled = state.clipBusy;
  }
  if (clipMenuPop) {
    for (const btn of clipMenuPop.querySelectorAll(".pawn-select-option")) {
      btn.classList.toggle("is-selected", btn.dataset.value === state.clipMode);
    }
  }
}

function openClipMenu(which) {
  if (state.clipBusy) return;
  const trigger = which === "chat" ? el.btnChatClipMenu : el.btnClipMenu;
  const control = which === "chat" ? el.chatClipControl : el.clipControl;
  if (!trigger) return;
  if (clipMenuPop) {
    closeClipMenus();
    return;
  }
  const pop = document.createElement("div");
  pop.className = "pawn-fixed-pop pawn-clip-pop";
  pop.setAttribute("role", "menu");
  for (const mode of CLIP_MODES) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className =
      "pawn-select-option" + (mode.value === state.clipMode ? " is-selected" : "");
    btn.setAttribute("role", "menuitem");
    btn.dataset.value = mode.value;
    btn.textContent = mode.label;
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      void setClipMode(mode.value);
      closeClipMenus();
    });
    pop.appendChild(btn);
  }
  document.body.appendChild(pop);
  clipMenuPop = pop;
  placeFixedPop(pop, trigger, { preferUp: which === "chat", align: "end", minWidth: 148 });
  trigger.setAttribute("aria-expanded", "true");
  setIcon(trigger, "chevron-up");
  const onOutside = (ev) => {
    if (pop.contains(ev.target) || control?.contains(ev.target)) return;
    closeClipMenus();
  };
  pop._onOutside = onOutside;
  document.addEventListener("pointerdown", onOutside, true);
}

function closeClipMenus() {
  if (clipMenuPop) {
    if (clipMenuPop._onOutside) {
      document.removeEventListener("pointerdown", clipMenuPop._onOutside, true);
    }
    clipMenuPop.remove();
    clipMenuPop = null;
  }
  for (const trigger of [el.btnClipMenu, el.btnChatClipMenu]) {
    if (!trigger) continue;
    trigger.setAttribute("aria-expanded", "false");
    setIcon(trigger, "chevron-down");
  }
}

async function setClipMode(mode) {
  if (!CLIP_MODES.some((m) => m.value === mode)) return;
  state.clipMode = mode;
  await chrome.storage.local.set({ clipMode: mode });
  syncClipUi();
}

function syncSettingsUi() {
  el.settingsPanel.hidden = !state.settingsOpen;
  el.btnSettings.classList.toggle("is-active", state.settingsOpen);
  el.btnSettings.setAttribute("aria-pressed", state.settingsOpen ? "true" : "false");
  const hidePanels = state.settingsOpen;
  if (el.panelCapture) {
    el.panelCapture.hidden = hidePanels || state.activeTab !== "capture";
  }
  if (el.panelChat) {
    el.panelChat.hidden = hidePanels || state.activeTab !== "chat";
  }
}

function setActiveTab(tab, opts = {}) {
  const next = tab === "chat" ? "chat" : "capture";
  state.activeTab = next;
  state.captureDest = next === "chat" ? "chat" : "tray";
  if (el.tabCapture) {
    el.tabCapture.classList.toggle("is-active", next === "capture");
    el.tabCapture.setAttribute("aria-selected", next === "capture" ? "true" : "false");
  }
  if (el.tabChat) {
    el.tabChat.classList.toggle("is-active", next === "chat");
    el.tabChat.setAttribute("aria-selected", next === "chat" ? "true" : "false");
  }
  syncSettingsUi();
  if (opts.persist !== false) {
    void chrome.storage.local.set({ activeTab: next });
  }
  if (next === "chat") {
    void refreshPageContext();
    void loadChatModels();
    renderChat();
  }
}

function bind() {
  el.tabCapture?.addEventListener("click", () => setActiveTab("capture"));
  el.tabChat?.addEventListener("click", () => setActiveTab("chat"));
  el.targetKind.addEventListener("change", async () => {
    state.targetKind = el.targetKind.value;
    if (
      state.targetKind === "new" ||
      state.targetKind === "research" ||
      state.targetKind === "event"
    ) {
      state.stickyPath = "";
      state.sessionId = "";
    }
    await persistTarget();
    await loadPickerData();
    renderTargetExtra();
    render();
    renderChatContext();
    if (
      state.targetKind !== "new" &&
      state.targetKind !== "research" &&
      state.targetKind !== "event"
    ) {
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
    state.captureDest = "tray";
    void requestSnip("pawn-snip-selection");
  });
  el.btnRegion.addEventListener("click", () => {
    state.captureDest = "tray";
    const info = clipModeInfo(state.clipMode);
    void requestSnip(info.snip, { clip: true });
  });
  el.btnClipMenu.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    openClipMenu("capture");
  });
  el.btnSave.addEventListener("click", () => void saveTray());
  el.btnAddPlanner?.addEventListener("click", () => void addToPlanner());

  el.btnChatSelection?.addEventListener("click", () => {
    state.captureDest = "chat";
    void requestSnip("pawn-snip-selection");
  });
  el.btnChatRegion?.addEventListener("click", () => {
    state.captureDest = "chat";
    const info = clipModeInfo(state.clipMode);
    void requestSnip(info.snip, { clip: true });
  });
  el.btnChatClipMenu?.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    openClipMenu("chat");
  });
  el.btnChatNew?.addEventListener("click", () => void startNewChat());
  el.btnChatSend?.addEventListener("click", () => void sendChat());
  el.btnChatStop?.addEventListener("click", () => stopChat());
  el.chatInput?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      void sendChat();
      return;
    }
    // "@" at start or after whitespace opens the note/session picker (Obsidian-like).
    if (ev.key === "@") {
      const ta = el.chatInput;
      const pos = ta.selectionStart ?? 0;
      const before = ta.value.slice(0, pos);
      if (pos === 0 || /\s$/.test(before)) {
        ev.preventDefault();
        void openNoteContextPicker();
      }
    }
  });
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
    render();
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
  const stored = await chrome.storage.local.get(["settings", "target", "clipMode", "activeTab"]);
  state.settings = { ...DEFAULTS, ...(stored.settings || {}) };
  el.serverUrl.value = state.settings.serverUrl;
  el.apiToken.value = state.settings.apiToken;
  if (stored.clipMode && CLIP_MODES.some((m) => m.value === stored.clipMode)) {
    state.clipMode = stored.clipMode;
  }
  if (stored.activeTab === "chat" || stored.activeTab === "capture") {
    state.activeTab = stored.activeTab;
  }
  const t = stored.target || {};
  state.targetKind = t.kind || "new";
  state.stickyPath = t.path || "";
  state.sessionId = t.sessionId || "";
  state.title = t.title || "";
  state.sourceUrl = t.sourceUrl || "";
  state.collection = t.collection || "";
  state.hint = t.hint || "";
  state.eventStart = t.eventStart || "";
  state.eventEnd = t.eventEnd || "";
  state.eventAttendees = Array.isArray(t.eventAttendees) ? t.eventAttendees : [];
  state.eventOrganizer = t.eventOrganizer || "";
  state.eventLocation = t.eventLocation || "";
  state.eventMeetingUrl = t.eventMeetingUrl || "";
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
      collection: state.collection,
      hint: state.hint,
      eventStart: state.eventStart,
      eventEnd: state.eventEnd,
      eventAttendees: state.eventAttendees,
      eventOrganizer: state.eventOrganizer,
      eventLocation: state.eventLocation,
      eventMeetingUrl: state.eventMeetingUrl,
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
  if (state.targetKind === "research") {
    const coll = (state.collection || "").trim() || "_";
    return `tray:research:${coll}`;
  }
  if (state.targetKind === "event") {
    return `tray:event:${(state.eventStart || "").slice(0, 10) || "_"}`;
  }
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
  if (data.lastPageTitle) {
    state.chat.pageTitle = data.lastPageTitle;
    if (!state.title) {
      state.title = data.lastPageTitle;
      el.title.value = state.title;
    }
  }
  if (data.lastPageUrl) {
    state.sourceUrl = data.lastPageUrl;
    state.chat.pageUrl = data.lastPageUrl;
  }
  const toChat = state.captureDest === "chat";
  for (const snip of pending) {
    if (toChat) {
      attachSnippetToChat(snip);
    } else {
      if (state.tray.some((s) => s.id === snip.id)) continue;
      state.tray.push({ ...snip, saved: false });
    }
  }
  await chrome.storage.local.set({ pendingSnippets: [] });
  if (toChat) {
    await persistChatState();
    if (state.activeTab !== "chat") setActiveTab("chat");
    else renderChat();
    showNotice(`Attached ${pending.length} to chat`);
  } else {
    await persistTray();
    await persistTarget();
    render();
  }
}

async function requestSnip(type, opts = {}) {
  const isClip = !!opts.clip;
  if (isClip) {
    if (state.clipBusy) return;
    state.clipBusy = true;
    syncClipUi();
  }
  try {
    // Visible-area capture runs in the side panel (extension page). Doing it in
    // the MV3 service worker often kills the one-shot reply channel mid-write.
    if (type === "pawn-snip-visible") {
      await captureVisibleInPanel();
      return;
    }
    const resp = await snipOverPort(type);
    await drainPending();
    if (resp && resp.ok === false) {
      const err = resp.error || "Capture failed";
      if (!/already in progress|cancelled/i.test(err)) showNotice(err, true);
    }
  } catch (err) {
    await drainPending();
    const msg = String(err?.message || err || "");
    // Port can drop after a successful enqueue; storage already delivered the clip.
    if (
      /connection closed|message port closed/i.test(msg) &&
      state.tray.some((s) => !s.saved)
    ) {
      return;
    }
    if (!/already in progress|cancelled/i.test(msg)) {
      showNotice(msg || "Capture failed", true);
    }
  } finally {
    if (isClip) {
      state.clipBusy = false;
      syncClipUi();
    }
  }
}

function newSnippetId() {
  if (crypto?.randomUUID) return crypto.randomUUID().replace(/-/g, "").slice(0, 16);
  return `s${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

async function findContentTab() {
  const focused = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  let tab = focused.find((t) => t.id != null && /^https?:/i.test(t.url || ""));
  if (tab) return tab;
  const active = await chrome.tabs.query({ active: true });
  tab = active.find((t) => t.id != null && /^https?:/i.test(t.url || ""));
  return tab || focused[0] || active[0] || null;
}

async function captureVisibleInPanel() {
  const tab = await findContentTab();
  if (!tab?.id) throw new Error("No active tab");
  if (!/^https?:|^file:/i.test(tab.url || "")) {
    throw new Error("Cannot capture this page (open an http(s) tab).");
  }
  const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
    format: "jpeg",
    quality: 90,
  });
  const b64 = String(dataUrl || "").split(",", 2)[1] || "";
  if (!b64) throw new Error("Visible capture failed");
  if (tab.title) {
    state.chat.pageTitle = tab.title;
    if (!state.title) {
      state.title = tab.title;
      el.title.value = state.title;
    }
  }
  if (tab.url) {
    state.sourceUrl = tab.url;
    state.chat.pageUrl = tab.url;
  }
  const snip = {
    id: newSnippetId(),
    kind: "image",
    text: "",
    data_base64: b64,
    media_type: "image/jpeg",
    source_url: tab.url || "",
    page_title: tab.title || "",
    captured_at: new Date().toISOString(),
    saved: false,
  };
  if (state.captureDest === "chat") {
    attachSnippetToChat(snip);
    await persistChatState();
    if (state.activeTab !== "chat") setActiveTab("chat");
    else renderChat();
    showNotice("Attached visible area to chat");
    return;
  }
  state.tray.push(snip);
  await persistTarget();
  await persistTray();
  render();
}

/** Long-lived port handoff — survives full-viewport capture in MV3. */
function snipOverPort(type) {
  return new Promise((resolve, reject) => {
    let settled = false;
    let port;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      try {
        port?.disconnect();
      } catch (_) {
        /* ignore */
      }
      fn(value);
    };
    try {
      port = chrome.runtime.connect({ name: "pawn-snip" });
    } catch (err) {
      reject(err);
      return;
    }
    port.onMessage.addListener((msg) => {
      finish(resolve, msg || { ok: true });
    });
    port.onDisconnect.addListener(() => {
      if (settled) return;
      const err = chrome.runtime.lastError;
      finish(
        reject,
        new Error(err?.message || "Capture connection closed"),
      );
    });
    try {
      port.postMessage({ type });
    } catch (err) {
      finish(reject, err);
    }
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
      state.vaultName = "";
      return;
    }
  }
  await loadMeta();
}

async function loadMeta() {
  try {
    const data = await api("GET", "/v1/meta");
    state.vaultName = String(data?.obsidian_vault_name || "").trim();
  } catch (_) {
    state.vaultName = "";
  }
}

/** Mirror pawn_server.core.vault_protocol.build_obsidian_open_url */
function buildObsidianOpenUrl(vaultName, filePath) {
  const vault = encodeURIComponent(vaultName || "");
  const normalized = String(filePath || "")
    .replace(/^\/+/, "")
    .replace(/\\/g, "/")
    .replace(/\/+/g, "/");
  // encodeURIComponent turns `/` into `%2F` (same effect as server replace).
  const file = encodeURIComponent(normalized);
  if (!vault || !file) return "";
  return `obsidian://open?vault=${vault}&file=${file}`;
}

function pathBasename(path) {
  const parts = String(path || "").replace(/\\/g, "/").split("/").filter(Boolean);
  return parts[parts.length - 1] || path || "";
}

function destinationTitleForPath(path) {
  if (!path) return "";
  const hit = state.captures.find((c) => c.path === path);
  if (hit?.title) return hit.title;
  return pathBasename(path).replace(/\.md$/i, "");
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
  if (state.targetKind === "research") {
    try {
      const data = await api("GET", "/v1/captures/collections");
      state.collections = data.data || [];
    } catch (_) {
      state.collections = [];
    }
  }
}

function renderTargetExtra() {
  el.targetExtra.innerHTML = "";
  if (state.targetKind === "new") return;

  if (state.targetKind === "event") {
    const startField = document.createElement("label");
    startField.className = "pawn-field";
    startField.append(
      Object.assign(document.createElement("span"), { textContent: "Start (local)" }),
    );
    const startInput = document.createElement("input");
    startInput.type = "text";
    startInput.placeholder = "2026-10-28T10:00";
    startInput.value = state.eventStart;
    startInput.addEventListener("input", () => {
      state.eventStart = startInput.value.trim();
      void persistTarget();
    });
    startField.append(startInput);

    const endField = document.createElement("label");
    endField.className = "pawn-field";
    endField.append(
      Object.assign(document.createElement("span"), { textContent: "End (optional)" }),
    );
    const endInput = document.createElement("input");
    endInput.type = "text";
    endInput.placeholder = "2026-10-28T10:30";
    endInput.value = state.eventEnd;
    endInput.addEventListener("input", () => {
      state.eventEnd = endInput.value.trim();
      void persistTarget();
    });
    endField.append(endInput);

    el.targetExtra.append(startField, endField);
    return;
  }

  if (state.targetKind === "research") {
    const collField = document.createElement("label");
    collField.className = "pawn-field";
    const collSpan = document.createElement("span");
    collSpan.textContent = "Collection (optional)";
    const collRow = document.createElement("div");
    collRow.className = "pawn-research-row";
    const collInput = document.createElement("input");
    collInput.type = "text";
    collInput.placeholder = "Leave blank — Pawn proposes";
    collInput.value = state.collection;
    collInput.addEventListener("input", () => {
      state.collection = collInput.value.trim();
      void persistTarget();
      render();
    });
    const pickBtn = document.createElement("button");
    pickBtn.type = "button";
    pickBtn.className = "clickable-icon";
    pickBtn.title = "Pick existing collection";
    pickBtn.setAttribute("aria-label", "Pick existing collection");
    setIcon(pickBtn, "search");
    pickBtn.addEventListener("click", () => void openCollectionPicker());
    collRow.append(collInput, pickBtn);
    collField.append(collSpan, collRow);

    const hintField = document.createElement("label");
    hintField.className = "pawn-field";
    const hintSpan = document.createElement("span");
    hintSpan.textContent = "Hint (optional)";
    const hintInput = document.createElement("input");
    hintInput.type = "text";
    hintInput.placeholder = "e.g. Blade Runner";
    hintInput.value = state.hint;
    hintInput.addEventListener("input", () => {
      state.hint = hintInput.value;
      void persistTarget();
    });
    hintField.append(hintSpan, hintInput);

    el.targetExtra.append(collField, hintField);
    return;
  }

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

async function openCollectionPicker() {
  await loadPickerData();
  const items = state.collections.map((row) => ({
    id: row.id,
    title: row.title || row.id,
    detail: row.id,
    value: row.id,
  }));
  openSuggestModal({
    placeholder: "Find a collection…",
    items,
    emptyText: "No collections yet — type a new name or leave blank",
    onChoose: async (item) => {
      state.collection = item.value;
      await persistTarget();
      renderTargetExtra();
      render();
    },
  });
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
  if (state.targetKind === "new" || state.targetKind === "research") return;
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

  const paintActive = () => {
    const buttons = [...list.querySelectorAll(".pawn-suggest-item")];
    buttons.forEach((btn, i) => {
      btn.classList.toggle("is-active", i === active);
    });
    const activeEl = list.querySelector(".is-active");
    if (activeEl) activeEl.scrollIntoView({ block: "nearest" });
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
      // Highlight only — do not rebuild the list (that ate click events).
      btn.addEventListener("mouseenter", () => {
        if (active === i) return;
        active = i;
        paintActive();
      });
      // pointerdown is reliable; click was lost when the node was recreated.
      btn.addEventListener("pointerdown", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        void choose(item);
      });
      list.appendChild(btn);
    });
    paintActive();
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
      paintActive();
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      active = Math.max(active - 1, 0);
      paintActive();
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
  renderDestination();
  el.tray.innerHTML = "";
  el.empty.hidden = state.tray.length > 0;
  if (el.empty && state.tray.length === 0 && state.targetKind === "research") {
    el.empty.textContent =
      "Select text or capture a region. Save writes one inbox note per snippet and queues enrich.";
  } else if (el.empty && state.tray.length === 0) {
    el.empty.textContent =
      "Select text or capture a region. Snippets stack here in order, then Save appends them to the target note.";
  }
  state.tray.forEach((snip, index) => {
    el.tray.appendChild(renderCard(snip, index));
  });
  const unsaved = state.tray.some((s) => !s.saved);
  el.btnSave.disabled = !unsaved;
}

function showStartNewNoteButton() {
  if (state.targetKind === "research") return state.tray.length > 0;
  if (state.stickyPath) return true;
  if (state.targetKind === "session" && state.sessionId) return true;
  if (state.targetKind !== "new") return true;
  return state.tray.length > 0;
}

function renderDestination() {
  if (!el.destination) return;
  el.destination.replaceChildren();

  const body = document.createElement("div");
  body.className = "pawn-destination-body";

  let titleText = "";
  let pathText = "";
  let openUrl = "";
  let muted = false;

  if (state.targetKind === "research") {
    const coll = (state.collection || "").trim();
    titleText = coll ? `Research · ${coll}` : "Research · Inbox";
    pathText = coll
      ? "One inbox note per snippet; enrich files into this collection"
      : "Pawn proposes a collection on enrich";
    muted = true;
  } else if (state.targetKind === "event") {
    titleText = state.title || "Planner event";
    pathText = [state.eventStart, state.eventEnd].filter(Boolean).join(" → ") || "Set start time";
    muted = true;
  } else if (state.stickyPath) {
    titleText = destinationTitleForPath(state.stickyPath);
    pathText = state.stickyPath;
    openUrl = buildObsidianOpenUrl(state.vaultName, state.stickyPath);
  } else if (state.targetKind === "session" && state.sessionId) {
    const hit = state.sessions.find((s) => s.session_id === state.sessionId);
    titleText = hit?.title || `Session ${state.sessionId}`;
    pathText = hit?.transcript_path || `session:${state.sessionId}`;
    if (hit?.transcript_path) {
      openUrl = buildObsidianOpenUrl(state.vaultName, hit.transcript_path);
    } else {
      muted = true;
    }
  } else {
    titleText = "New capture on save";
    pathText = "Creates a note under Pawn/Captures/ on Save";
    muted = true;
  }

  if (openUrl) {
    const link = document.createElement("a");
    link.className = "pawn-destination-title";
    link.href = openUrl;
    link.textContent = titleText;
    link.title = "Open in Obsidian";
    link.addEventListener("click", (ev) => {
      ev.preventDefault();
      void chrome.tabs.create({ url: openUrl });
    });
    body.appendChild(link);
  } else {
    const title = document.createElement("div");
    title.className = "pawn-destination-title" + (muted ? " is-muted" : "");
    title.textContent = titleText;
    body.appendChild(title);
  }

  if (pathText && pathText !== titleText) {
    const pathEl = document.createElement("div");
    pathEl.className = "pawn-destination-path";
    pathEl.textContent = pathText;
    body.appendChild(pathEl);
  }

  el.destination.appendChild(body);

  if (showStartNewNoteButton()) {
    const btn = iconButton("file-plus", "Start new note", () => void startNewPage());
    el.destination.appendChild(btn);
  }
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
  const attach = iconButton("message-square", "Attach to chat", () => void attachTrayToChat(index));
  const up = iconButton("chevron-up", "Move up", () => void moveSnippet(index, -1));
  up.disabled = index === 0 || snip.saved;
  const down = iconButton("chevron-down", "Move down", () => void moveSnippet(index, 1));
  down.disabled = index >= state.tray.length - 1 || snip.saved;
  const del = iconButton("x", "Remove", () => void removeSnippet(index));
  actions.append(attach, up, down, del);
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
  if (state.targetKind === "event") {
    return {
      kind: "event",
      title: state.title || "event",
      source_url: state.sourceUrl,
      start: state.eventStart || undefined,
      end: state.eventEnd || undefined,
      attendees: state.eventAttendees?.length ? state.eventAttendees : undefined,
      organizer: state.eventOrganizer || undefined,
      location: state.eventLocation || undefined,
      meeting_url: state.eventMeetingUrl || undefined,
      // Sticky path after first Save so later clips (screenshots) append.
      path: state.stickyPath || undefined,
    };
  }
  if (state.targetKind === "research") {
    return {
      kind: "research",
      title: state.title || "capture",
      source_url: state.sourceUrl,
      collection: state.collection || undefined,
      hint: state.hint || undefined,
    };
  }
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
      snippets: unsaved.map((s) => {
        const row = {
          id: s.id,
          kind: s.kind,
          text: s.text || "",
          data_base64: s.data_base64 || undefined,
          media_type: s.media_type || undefined,
          source_url: s.source_url || state.sourceUrl,
          captured_at: s.captured_at,
        };
        if (s.anchor && typeof s.anchor === "object") row.anchor = s.anchor;
        return row;
      }),
    };
    const result = await api("POST", "/v1/captures", payload);
    if (state.targetKind !== "research") {
      state.stickyPath = result.path;
    }
    const written = new Set(result.written || []);
    const skipped = new Set(result.skipped || []);
    for (const s of state.tray) {
      if (written.has(s.id) || skipped.has(s.id)) s.saved = true;
    }
    await persistTarget();
    await persistTray();
    if (Array.isArray(result.anchors) && result.anchors.length) {
      try {
        await chrome.runtime.sendMessage({
          type: "pawn-anchors-refresh",
          anchors: result.anchors,
        });
      } catch (_) {
        /* background may be asleep — painter will refetch later */
      }
    }
    if (result.mode === "research") {
      const n = written.size;
      const jobs = (result.jobs || []).length;
      const where = (result.paths && result.paths[0]) || result.path || "Inbox";
      showNotice(
        jobs
          ? `Saved ${n} → Inbox (enriching ${jobs}…)`
          : `Saved ${n} → ${where}`,
      );
    } else if (result.mode === "event") {
      showNotice(`Planner event → ${result.path}`);
    } else {
      const mode = result.mode === "annotations" ? "annotations" : "note";
      showNotice(`Saved ${written.size} → ${result.path} (${mode})`);
    }
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
  if (state.targetKind === "research") {
    state.tray = [];
    await persistTray();
    render();
    showNotice("Tray cleared — next Save creates fresh inbox atoms");
    return;
  }
  state.targetKind = "new";
  state.stickyPath = "";
  state.sessionId = "";
  state.title = "";
  state.eventStart = "";
  state.eventEnd = "";
  state.eventAttendees = [];
  state.eventOrganizer = "";
  state.eventLocation = "";
  state.eventMeetingUrl = "";
  el.targetKind.value = "new";
  el.title.value = "";
  state.tray = [];
  await persistTarget();
  await persistTray();
  renderTargetExtra();
  render();
  showNotice("New page — next Save creates a fresh capture note");
}

/** Normalize scraped Outlook times toward YYYY-MM-DDTHH:mm when possible. */
function normalizeScrapedTime(raw) {
  const s = String(raw || "").trim();
  if (!s) return "";
  const iso = s.match(/(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})/);
  if (iso) return `${iso[1]}T${iso[2]}`;
  // Outlook reading pane: "10/28/2026, 11:00 AM"
  const us = s.match(
    /(\d{1,2})\/(\d{1,2})\/(\d{4}),?\s+(\d{1,2}):(\d{2})\s*([AP]M)/i,
  );
  if (us) {
    const pad = (n) => String(n).padStart(2, "0");
    let hour = parseInt(us[4], 10);
    const min = pad(us[5]);
    const ap = us[6].toUpperCase();
    if (ap === "PM" && hour < 12) hour += 12;
    if (ap === "AM" && hour === 12) hour = 0;
    return `${us[3]}-${pad(us[1])}-${pad(us[2])}T${pad(hour)}:${min}`;
  }
  const day = s.match(/(\d{4}-\d{2}-\d{2})/);
  if (day && !/\d{1,2}:\d{2}/.test(s)) return day[1];
  const m = Date.parse(s);
  if (!Number.isNaN(m)) {
    const d = new Date(m);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }
  return s.slice(0, 32);
}

function isStubEventTitle(t) {
  const s = String(t || "").trim();
  return (
    !s ||
    /^(?:\(\d+\)\s*)?(calendar|mail|outlook|microsoft\s*teams|tracking|accepted|declined|busy|free|tentative|reminder|details|chat|join|meeting|inbox)\b/i.test(
      s,
    ) ||
    /calendar\s*[|–—-]\s*microsoft\s*teams/i.test(s) ||
    /^calendar\s*-/i.test(s) ||
    /-\s*outlook\s*$/i.test(s) ||
    /^prepare for this meeting$/i.test(s) ||
    /^microsoft teams meeting$/i.test(s)
  );
}

function subjectFromPageTitle(raw) {
  const s = String(raw || "").trim();
  let m = s.match(/^(.+?)\s*[–—|-]\s*Meeting\s*[–—|-]\s*Calendar\b/i);
  if (m) return m[1].trim();
  m = s.match(/^(.+?)\s*[–—|-]\s*(?:Mail|Outlook|Calendar)\b/i);
  if (m && !isStubEventTitle(m[1])) return m[1].trim();
  return "";
}

function isJunkAttendee(name) {
  const s = String(name || "")
    .replace(/\s*\[(?:image|object)[^\]]*\]/gi, "")
    .trim();
  if (!s || s.length < 2 || s.length > 80) return true;
  if (/\[(?:image|object)/i.test(name || "")) return true;
  if (
    /^(required|optional|organizer|attendees|participants|accepted|tentative|invite|chat|join|copy|delete|archive|reply|forward|busy|reminder|categorize|private|prepare|copilot|viva|tracking|meeting|calendar|more|close|cancel|save|send|help|you|me)$/i.test(
      s,
    )
  ) {
    return true;
  }
  if (!/[A-Za-zÀ-ÿ]/.test(s)) return true;
  // Toolbar singles are lowercase ("chat"); allow "Ada" / emails.
  if (!s.includes(" ") && !s.includes("@") && !/^[A-ZÀ-Ÿ]/.test(s)) return true;
  return false;
}

function cleanAttendees(list) {
  const out = [];
  const seen = new Set();
  for (const raw of list || []) {
    const name = String(raw || "")
      .replace(/\s*\[(?:image|object)[^\]]*\]/gi, "")
      .trim();
    if (isJunkAttendee(name)) continue;
    const key = name.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(name);
  }
  return out;
}

/** Prefer a real meeting subject from tray text over tab-title stubs. */
function titleFromTrayContent(fallback) {
  const skip =
    /^(source:|attendees:|participants:|organizer:|location:|where:|meeting:|meeting id:|passcode:|join:|start:|end:|when:|topics:|tags:|https?:|### snippet|microsoft teams meeting)/i;
  const candidates = [];
  for (const s of state.tray) {
    if (s.kind !== "text" || !s.text) continue;
    for (const raw of String(s.text).split(/\n+/)) {
      let line = raw.trim().replace(/^#+\s*/, "");
      if (/^outlook event:/i.test(line)) line = line.split(":").slice(1).join(":").trim();
      if (/^subject:/i.test(line)) line = line.split(":").slice(1).join(":").trim();
      if (!line || skip.test(line) || isStubEventTitle(line)) continue;
      if (/\((Organizer|Required|Optional|Accepted|Tentative)\)\s*$/i.test(line)) continue;
      if (line.length >= 2 && line.length <= 120) candidates.push(line);
    }
  }
  if (!candidates.length) return fallback;
  // Prefer longer subject-like lines (chrome words are short).
  candidates.sort((a, b) => b.length - a.length);
  return candidates[0];
}

/**
 * Outlook → planner: scrape obvious fields into the Capture target + tray.
 * Does **not** Save — user reviews and clicks Save.
 */
async function addToPlanner() {
  showNotice("Reading Outlook event…", false, true);
  let scraped;
  try {
    scraped = await chrome.runtime.sendMessage({ type: "pawn-scrape-outlook" });
  } catch (err) {
    showNotice(String(err?.message || err), true);
    return;
  }
  const url = scraped?.url || state.sourceUrl || "";
  let title = (scraped?.title || "").trim();
  const fromPage =
    subjectFromPageTitle(scraped?.page_title) || subjectFromPageTitle(scraped?.title);
  if (fromPage) title = fromPage;
  else if (isStubEventTitle(title)) {
    const page = (scraped?.page_title || "").trim();
    title = !isStubEventTitle(page) ? page : title;
  }
  const start = normalizeScrapedTime(scraped?.start);
  const end = normalizeScrapedTime(scraped?.end);
  const attendees = cleanAttendees(
    Array.isArray(scraped?.attendees) ? scraped.attendees : [],
  );
  let organizer = String(scraped?.organizer || "").trim();
  if (isJunkAttendee(organizer)) organizer = "";
  const location = String(scraped?.location || "").trim();
  const meetingUrl = String(scraped?.meeting_url || "").trim();
  const meetingId = String(scraped?.meeting_id || "").trim();
  const passcode = String(scraped?.passcode || "").trim();
  // Lean tray note: extras only — title/people/times live on the target.
  const bodyParts = [];
  if (meetingUrl) bodyParts.push(`Meeting: ${meetingUrl}`);
  if (meetingId) bodyParts.push(`Meeting ID: ${meetingId}`);
  if (passcode) bodyParts.push(`Passcode: ${passcode}`);
  if (location && !/^add a room/i.test(location)) bodyParts.push(`Location: ${location}`);
  if (scraped?.selection) bodyParts.push(String(scraped.selection).trim());
  if (!scraped?.ok && scraped?.error && !/calendar event first/i.test(scraped.error || "")) {
    showNotice(scraped.error, true);
  }

  state.targetKind = "event";
  el.targetKind.value = "event";
  state.sourceUrl = url;
  if (start) state.eventStart = start;
  if (end) state.eventEnd = end;
  state.eventAttendees = attendees;
  state.eventOrganizer = organizer;
  state.eventLocation = /^add a room/i.test(location) ? "" : location;
  state.eventMeetingUrl = meetingUrl;
  state.sessionId = "";

  // Drop prior auto-extract text snips; keep user clips (shots / selection).
  const kept = state.tray.filter(
    (s) => !s.saved && s.kind === "image",
  );
  const selectionKept = state.tray.filter(
    (s) =>
      !s.saved &&
      s.kind === "text" &&
      s.text &&
      !/^Meeting:|^Meeting ID:|^Passcode:|^Location:|^Outlook event:|^Start:|^End:|^Attendees:/m.test(
        s.text,
      ),
  );
  if (bodyParts.length) {
    const snip = {
      id: `evt${Date.now().toString(36)}`,
      kind: "text",
      text: bodyParts.join("\n"),
      source_url: url,
      captured_at: new Date().toISOString(),
      saved: false,
    };
    state.tray = [snip, ...kept, ...selectionKept.filter((s) => s.id !== snip.id)];
  } else {
    state.tray = [...kept, ...selectionKept];
  }

  title = titleFromTrayContent(title);
  if (isStubEventTitle(title)) {
    title =
      subjectFromPageTitle(scraped?.page_title) ||
      titleFromTrayContent("") ||
      (state.title || "").trim() ||
      "Calendar event";
  }
  state.title = title;
  el.title.value = title;
  await persistTarget();
  await persistTray();
  renderTargetExtra();
  render();
  const bits = [
    title && !isStubEventTitle(title) ? title : null,
    attendees.length ? `${attendees.length} people` : null,
    start ? start.slice(11, 16) || start : null,
  ].filter(Boolean);
  showNotice(
    bits.length
      ? `Staged ${bits.join(" · ")} — review tray, then Save`
      : "Staged Outlook fields — review tray, then Save",
  );
}

function showNotice(text, isError = false, isBusy = false) {
  el.notice.hidden = !text;
  el.notice.textContent = text || "";
  el.notice.classList.toggle("is-error", !!isError);
  el.notice.classList.toggle("is-busy", !!isBusy && !isError);
}

/* ── Chat ───────────────────────────────────────────────────────────────── */

function newChatId() {
  if (crypto?.randomUUID) return `chat:${crypto.randomUUID()}`;
  return `chat:${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

function conversationLabel(id, title) {
  if (String(id || "").startsWith("note:")) {
    const base = id.slice(5).replace(/\.md$/i, "");
    return base.split("/").pop() || base;
  }
  return title || "Chat";
}

function formatUpdatedAt(ms) {
  if (!ms) return "";
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return "";
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startDay = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const diff = Math.round((startToday - startDay) / 86400000);
  const hm = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (diff === 0) return `Today ${hm}`;
  if (diff === 1) return `Yesterday ${hm}`;
  if (diff < 7) {
    return `${d.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" })} ${hm}`;
  }
  if (d.getFullYear() === now.getFullYear()) {
    return d.toLocaleDateString([], { day: "numeric", month: "short" });
  }
  return d.toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });
}

function ensureConversationRecord(id) {
  const key = id || state.chat.conversationId || newChatId();
  if (!state.chat.conversations[key]) {
    state.chat.conversations[key] = {
      id: key,
      title: conversationLabel(key),
      messages: [],
      updatedAt: Date.now(),
      model: state.chat.model || "",
      reasoning: state.chat.reasoning || "",
      route: state.chat.route || "",
    };
  }
  return state.chat.conversations[key];
}

function snapshotCurrentConversation() {
  const id = state.chat.conversationId || newChatId();
  state.chat.conversationId = id;
  const rec = ensureConversationRecord(id);
  rec.messages = state.chat.messages.slice(-CHAT_MSG_LIMIT);
  rec.model = state.chat.model || "";
  rec.reasoning = state.chat.reasoning || "";
  rec.route = state.chat.route || "";
  if (rec.messages.length) {
    const firstUser = rec.messages.find((m) => m.role === "user");
    if (firstUser?.content && (!rec.title || rec.title === "Chat")) {
      rec.title = String(firstUser.content).split(/\r?\n/)[0].slice(0, 48) || "Chat";
    }
    rec.updatedAt = Date.now();
  }
  pruneConversations();
}

function pruneConversations() {
  const entries = Object.values(state.chat.conversations)
    .filter((c) => (c.messages || []).length > 0 || c.id === state.chat.conversationId)
    .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  const keep = new Set(entries.slice(0, MAX_CHAT_CONVERSATIONS).map((c) => c.id));
  keep.add(state.chat.conversationId);
  for (const id of Object.keys(state.chat.conversations)) {
    if (!keep.has(id)) delete state.chat.conversations[id];
  }
}

function listConversationOptions(query) {
  const q = String(query || "")
    .trim()
    .toLowerCase();
  const options = [];
  const seen = new Set();
  const push = (id, label, deletable, updatedAt) => {
    if (seen.has(id)) return;
    seen.add(id);
    options.push({ id, label, deletable, updatedAt });
  };

  const currentId = state.chat.conversationId;
  const current = state.chat.conversations[currentId];
  push(
    currentId,
    conversationLabel(currentId, current?.title),
    !!(current && (current.messages || []).length),
    current?.updatedAt,
  );

  const recent = Object.values(state.chat.conversations)
    .filter((c) => (c.messages || []).length > 0)
    .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  for (const conv of recent) {
    push(conv.id, conversationLabel(conv.id, conv.title), true, conv.updatedAt);
  }

  if (!q) return options;
  return options.filter(
    (o) => o.label.toLowerCase().includes(q) || o.id.toLowerCase().includes(q),
  );
}

async function switchConversation(id) {
  if (!id || id === state.chat.conversationId) return;
  if (state.chat.abort) stopChat();
  snapshotCurrentConversation();
  state.chat.conversationId = id;
  const rec = ensureConversationRecord(id);
  state.chat.messages = Array.isArray(rec.messages) ? rec.messages.slice() : [];
  state.chat.model = rec.model || state.chat.model || "";
  state.chat.reasoning = rec.reasoning || "";
  state.chat.route = rec.route || "";
  state.chat.attachments = [];
  chatModelSelect?.refresh();
  await persistChatState();
  renderChat();
}

async function deleteConversation(id) {
  if (!id || !state.chat.conversations[id]) return;
  delete state.chat.conversations[id];
  if (state.chat.conversationId === id) {
    if (state.chat.abort) stopChat();
    state.chat.conversationId = newChatId();
    state.chat.messages = [];
    state.chat.attachments = [];
    ensureConversationRecord(state.chat.conversationId);
  }
  await persistChatState();
  renderChat();
  showNotice("Conversation deleted");
}

async function loadChatState() {
  const data = await chrome.storage.local.get(["chatStore", "chatState"]);
  const store = data.chatStore || null;
  if (store && store.conversations) {
    state.chat.conversations = store.conversations || {};
    state.chat.conversationId = store.currentId || newChatId();
    state.chat.attachments = Array.isArray(store.attachments) ? store.attachments : [];
    state.chat.noteContext = Array.isArray(store.noteContext) ? store.noteContext : [];
    state.chat.omitBound = !!store.omitBound;
    state.chat.model = store.model || "";
    state.chat.reasoning = store.reasoning || "";
    state.chat.route = store.route || "";
    state.chat.pageUrl = store.pageUrl || "";
    state.chat.pageTitle = store.pageTitle || "";
  } else {
    // Migrate single-thread chatState from v0.2.0.
    const c = data.chatState || {};
    const id = c.conversationId || newChatId();
    state.chat.conversationId = id;
    state.chat.conversations = {};
    if (Array.isArray(c.messages) && c.messages.length) {
      state.chat.conversations[id] = {
        id,
        title: conversationLabel(id),
        messages: c.messages.slice(-CHAT_MSG_LIMIT),
        updatedAt: Date.now(),
        model: c.model || "",
      };
    }
    state.chat.attachments = Array.isArray(c.attachments) ? c.attachments : [];
    state.chat.noteContext = [];
    state.chat.omitBound = false;
    state.chat.model = c.model || "";
    state.chat.pageUrl = c.pageUrl || "";
    state.chat.pageTitle = c.pageTitle || "";
  }
  const rec = ensureConversationRecord(state.chat.conversationId);
  state.chat.messages = Array.isArray(rec.messages) ? rec.messages.slice() : [];
  if (!state.chat.model && rec.model) state.chat.model = rec.model;
  if (!state.chat.reasoning && rec.reasoning) state.chat.reasoning = rec.reasoning;
  if (!state.chat.route && rec.route) state.chat.route = rec.route;
}

async function persistChatState() {
  snapshotCurrentConversation();
  await chrome.storage.local.set({
    chatStore: {
      currentId: state.chat.conversationId,
      conversations: state.chat.conversations,
      attachments: state.chat.attachments,
      noteContext: state.chat.noteContext,
      omitBound: state.chat.omitBound,
      model: state.chat.model,
      reasoning: state.chat.reasoning,
      route: state.chat.route,
      pageUrl: state.chat.pageUrl,
      pageTitle: state.chat.pageTitle,
    },
  });
  // Drop legacy single-thread blob after migration.
  await chrome.storage.local.remove("chatState");
  chatConvPicker?.refresh();
  chatModelSelect?.refresh();
}

/**
 * Position a fixed popover relative to an anchor.
 * @param {"down"|"up"} [opts.placement="down"] Prefer open direction. Conversation
 *   pickers should stay ``down`` (Obsidian-style) and shrink height instead of flipping.
 * @param {boolean} [opts.flip=true] Allow flipping when the preferred side overflows.
 */
function placeFixedPop(
  pop,
  anchor,
  { placement = "down", preferUp = false, flip = true, minWidth = 220, align = "start" } = {},
) {
  const preferDown = preferUp ? false : placement !== "up";
  const rect = anchor.getBoundingClientRect();
  const maxW = Math.max(160, window.innerWidth - 16);
  const width = Math.min(Math.max(rect.width, minWidth), maxW);
  pop.style.width = `${width}px`;
  let left =
    align === "end"
      ? rect.right - width
      : align === "center"
        ? rect.left + rect.width / 2 - width / 2
        : rect.left;
  left = Math.max(8, Math.min(left, window.innerWidth - width - 8));
  pop.style.left = `${left}px`;
  const gap = 4;
  const spaceBelow = Math.max(80, window.innerHeight - rect.bottom - 12);
  const spaceAbove = Math.max(80, rect.top - 12);
  if (preferDown && !flip) {
    // Stay below the control; cap height so the menu scrolls inside.
    pop.style.maxHeight = `${Math.min(420, spaceBelow)}px`;
    pop.style.top = `${rect.bottom + gap}px`;
    return;
  }
  if (!preferDown && !flip) {
    pop.style.maxHeight = `${Math.min(420, spaceAbove)}px`;
    pop.style.top = "0px";
    requestAnimationFrame(() => {
      pop.style.top = `${Math.max(8, rect.top - pop.offsetHeight - gap)}px`;
    });
    return;
  }
  requestAnimationFrame(() => {
    const box = pop.getBoundingClientRect();
    let top = preferDown ? rect.bottom + gap : rect.top - box.height - gap;
    if (preferDown && flip && box.bottom > window.innerHeight - 8) {
      top = Math.max(8, rect.top - box.height - gap);
    } else if (!preferDown && flip && top < 8) {
      top = Math.min(window.innerHeight - box.height - 8, rect.bottom + gap);
    }
    pop.style.top = `${Math.max(8, top)}px`;
  });
}

function mountConversationPicker(host) {
  if (!host) {
    return { refresh() {} };
  }
  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "pawn-conv-trigger";
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-label", "Choose conversation");
  const labelEl = document.createElement("span");
  labelEl.className = "pawn-conv-trigger-label";
  const chevron = document.createElement("span");
  chevron.className = "clickable-icon";
  chevron.setAttribute("aria-hidden", "true");
  setIcon(chevron, "chevron-down");
  trigger.append(labelEl, chevron);
  host.replaceChildren(trigger);

  let pop = null;
  let query = "";

  const refresh = () => {
    const rec = state.chat.conversations[state.chat.conversationId];
    labelEl.textContent = conversationLabel(state.chat.conversationId, rec?.title);
  };

  const close = () => {
    pop?.remove();
    pop = null;
    document.removeEventListener("pointerdown", onOutside, true);
    document.removeEventListener("keydown", onKey, true);
    setIcon(chevron, "chevron-down");
  };

  const onOutside = (ev) => {
    if (pop?.contains(ev.target) || trigger.contains(ev.target)) return;
    close();
  };

  const onKey = (ev) => {
    if (ev.key === "Escape") close();
  };

  const paintList = (list) => {
    list.replaceChildren();
    const options = listConversationOptions(query);
    if (!options.length) {
      const empty = document.createElement("div");
      empty.className = "pawn-conv-empty";
      empty.textContent = "No conversations match.";
      list.appendChild(empty);
      return;
    }
    for (const opt of options) {
      const row = document.createElement("div");
      row.className =
        "pawn-conv-option" + (opt.id === state.chat.conversationId ? " is-active" : "");
      const pick = document.createElement("button");
      pick.type = "button";
      pick.className = "pawn-conv-option-pick";
      pick.title = opt.label;
      const title = document.createElement("span");
      title.className = "pawn-conv-option-title";
      title.textContent = opt.label;
      pick.appendChild(title);
      const when = formatUpdatedAt(opt.updatedAt);
      if (when) {
        const date = document.createElement("span");
        date.className = "pawn-conv-option-date";
        date.textContent = when;
        pick.appendChild(date);
      }
      pick.addEventListener("click", (ev) => {
        ev.preventDefault();
        close();
        void switchConversation(opt.id);
      });
      row.appendChild(pick);
      if (opt.deletable) {
        const del = document.createElement("button");
        del.type = "button";
        del.className = "clickable-icon";
        del.setAttribute("aria-label", `Delete ${opt.label}`);
        del.title = "Delete conversation";
        setIcon(del, "trash");
        del.addEventListener("click", (ev) => {
          ev.preventDefault();
          ev.stopPropagation();
          if (window.confirm(`Delete “${opt.label}”?`)) {
            close();
            void deleteConversation(opt.id);
          }
        });
        row.appendChild(del);
      }
      list.appendChild(row);
    }
  };

  const open = () => {
    if (pop) {
      close();
      return;
    }
    query = "";
    pop = document.createElement("div");
    pop.className = "pawn-fixed-pop";
    pop.setAttribute("role", "listbox");
    const search = document.createElement("input");
    search.type = "search";
    search.className = "pawn-conv-search";
    search.placeholder = "Search conversations…";
    search.setAttribute("aria-label", "Search conversations");
    const list = document.createElement("div");
    list.className = "pawn-conv-list";
    pop.append(search, list);
    document.body.appendChild(pop);
    paintList(list);
    // Always open below the trigger (Obsidian); never flip over the toolbar.
    placeFixedPop(pop, trigger, { placement: "down", flip: false, minWidth: 260 });
    setIcon(chevron, "chevron-up");
    search.addEventListener("input", () => {
      query = search.value;
      paintList(list);
    });
    document.addEventListener("pointerdown", onOutside, true);
    document.addEventListener("keydown", onKey, true);
    requestAnimationFrame(() => {
      search.focus();
      search.select();
    });
  };

  trigger.addEventListener("click", (ev) => {
    ev.preventDefault();
    open();
  });

  refresh();
  return { refresh, close };
}

function shortModelLabel(id, allIds) {
  if (!id) return "Default";
  const raw = String(id);
  const at = raw.lastIndexOf("@");
  const name = at >= 0 ? raw.slice(at + 1) : raw;
  const ids = allIds || state.chat.models.map((m) => m.id);
  const collisions = ids.filter((other) => {
    if (!other || other === id) return false;
    const oAt = String(other).lastIndexOf("@");
    const oName = oAt >= 0 ? String(other).slice(oAt + 1) : String(other);
    return oName === name;
  });
  if (collisions.length && at >= 0) return raw;
  return name;
}

function currentModelChoice() {
  const id = state.chat.model;
  return state.chat.models.find((m) => m.id === id);
}

function effectiveReasoning() {
  const choice = currentModelChoice();
  if (!choice?.reasoning?.length) return "";
  const stored = state.chat.reasoning;
  if (stored && choice.reasoning.includes(stored)) return stored;
  return choice.reasoning_default || choice.reasoning[0] || "low";
}

function effectiveRoute() {
  const choice = currentModelChoice();
  if (!choice?.routes?.length) return "";
  const stored = state.chat.route;
  if (stored && choice.routes.includes(stored)) return stored;
  return choice.route_default || choice.routes[0] || "balanced";
}

function tuningFields() {
  const choice = currentModelChoice();
  return {
    reasoning: choice?.reasoning?.length ? effectiveReasoning() : undefined,
    route: choice?.routes?.length ? effectiveRoute() : undefined,
  };
}

function mountModelSelect(host) {
  if (!host) {
    return {
      refresh() {},
      get value() {
        return "";
      },
    };
  }
  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "pawn-model-btn";
  trigger.setAttribute("aria-haspopup", "dialog");
  trigger.title = "Model";
  const valueEl = document.createElement("span");
  valueEl.className = "pawn-model-btn-label";
  valueEl.textContent = "Default";
  const chevron = document.createElement("span");
  chevron.className = "clickable-icon";
  chevron.setAttribute("aria-hidden", "true");
  setIcon(chevron, "chevron-down");
  trigger.append(valueEl, chevron);
  host.replaceChildren(trigger);

  let pop = null;

  const refresh = () => {
    const ids = state.chat.models.map((m) => m.id);
    const current = state.chat.model;
    const choice = currentModelChoice();
    const name = current ? shortModelLabel(current, ids) : "Default";
    const effort = choice?.reasoning?.length
      ? REASONING_LABEL[effectiveReasoning()] || effectiveReasoning()
      : "";
    valueEl.textContent = [name, effort].filter(Boolean).join(" ");
    const route = choice?.routes?.length
      ? ROUTE_LABEL[effectiveRoute()] || effectiveRoute()
      : "";
    const vision = choice?.vision ? "Vision" : "";
    trigger.title = [current || "Default", effort, route, vision].filter(Boolean).join(" · ");
    trigger.setAttribute("aria-label", trigger.title);
  };

  const close = () => {
    pop?.remove();
    pop = null;
    trigger.setAttribute("aria-expanded", "false");
    setIcon(chevron, "chevron-down");
    document.removeEventListener("pointerdown", onOutside, true);
    document.removeEventListener("keydown", onKey, true);
  };

  const onOutside = (ev) => {
    if (pop?.contains(ev.target) || trigger.contains(ev.target)) return;
    close();
  };

  const onKey = (ev) => {
    if (ev.key === "Escape") close();
  };

  /** Keep the model pop above the trigger, fully on-screen, scrollable if tall. */
  const fitModelPop = () => {
    if (!pop) return;
    const rect = trigger.getBoundingClientRect();
    const margin = 8;
    const gap = 6;
    const maxW = Math.max(200, window.innerWidth - margin * 2);
    const width = Math.min(Math.max(240, rect.width + 40), maxW);
    pop.style.width = `${width}px`;
    // Prefer left-align with trigger; never hang off the left edge.
    let left = rect.left;
    if (left + width > window.innerWidth - margin) {
      left = window.innerWidth - margin - width;
    }
    left = Math.max(margin, left);
    pop.style.left = `${left}px`;
    const spaceAbove = Math.max(120, rect.top - margin - gap);
    pop.style.maxHeight = `${spaceAbove}px`;
    pop.style.overflowY = "auto";
    // Anchor bottom of pop just above the trigger (grows upward as content grows).
    pop.style.top = "auto";
    pop.style.bottom = `${Math.max(margin, window.innerHeight - rect.top + gap)}px`;
  };

  const openSubmenu = (parentPop, options, current, apply) => {
    parentPop.querySelector(".pawn-model-sub")?.remove();
    const sub = document.createElement("div");
    // Always stack inside the pop — never fly out left/right in the side panel.
    sub.className = "pawn-model-sub is-stacked";
    for (const option of options) {
      const item = document.createElement("button");
      item.type = "button";
      item.className =
        "pawn-model-sub-item" + (option.value === current ? " is-active" : "");
      if (option.disabled) item.disabled = true;
      const lab = document.createElement("span");
      lab.className = "pawn-model-sub-label";
      lab.textContent = option.label;
      item.appendChild(lab);
      if (option.value === current) {
        const mark = document.createElement("span");
        mark.className = "pawn-model-sub-check";
        setIcon(mark, "check");
        item.appendChild(mark);
      }
      item.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        if (option.disabled) return;
        apply(option.value);
      });
      sub.appendChild(item);
    }
    const openRow = parentPop.querySelector(".pawn-model-row.is-open");
    if (openRow instanceof HTMLElement) openRow.insertAdjacentElement("afterend", sub);
    else parentPop.appendChild(sub);
    fitModelPop();
    // Keep the open row + list visible inside the scrollable pop.
    requestAnimationFrame(() => {
      openRow?.scrollIntoView({ block: "nearest" });
    });
  };

  const addPopRow = (parentPop, key, value, onOpen) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "pawn-model-row";
    const keyEl = document.createElement("span");
    keyEl.className = "pawn-model-row-key";
    keyEl.textContent = key;
    const valEl = document.createElement("span");
    valEl.className = "pawn-model-row-value";
    valEl.textContent = value;
    const ch = document.createElement("span");
    ch.className = "pawn-model-row-chevron";
    setIcon(ch, "chevron-down");
    row.append(keyEl, valEl, ch);
    row.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      const wasOpen = row.classList.contains("is-open");
      parentPop.querySelector(".pawn-model-sub")?.remove();
      parentPop.querySelectorAll(".pawn-model-row.is-open").forEach((el) => {
        el.classList.remove("is-open");
        const c = el.querySelector(".pawn-model-row-chevron");
        if (c) setIcon(c, "chevron-down");
      });
      if (wasOpen) {
        fitModelPop();
        return;
      }
      row.classList.add("is-open");
      setIcon(ch, "chevron-up");
      onOpen();
    });
    parentPop.appendChild(row);
  };

  const open = () => {
    if (pop) {
      close();
      return;
    }
    pop = document.createElement("div");
    pop.className = "pawn-model-pop";
    const ids = state.chat.models.map((m) => m.id);
    const current = state.chat.model;
    const choice = currentModelChoice();
    addPopRow(pop, "Model", current ? shortModelLabel(current, ids) : "Default", () => {
      openSubmenu(
        pop,
        ids.length
          ? ids.map((id) => {
              const m = state.chat.models.find((x) => x.id === id);
              return {
                value: id,
                label: shortModelLabel(id, ids) + (m?.vision ? " · vision" : ""),
              };
            })
          : [{ value: "", label: "No models configured", disabled: true }],
        current,
        (id) => {
          if (!id) return;
          state.chat.model = id;
          const rec = ensureConversationRecord(state.chat.conversationId);
          rec.model = id;
          // Reset tuning when model changes if values not in new catalog.
          const next = state.chat.models.find((m) => m.id === id);
          if (!next?.reasoning?.includes(state.chat.reasoning)) state.chat.reasoning = "";
          if (!next?.routes?.includes(state.chat.route)) state.chat.route = "";
          refresh();
          close();
          void persistChatState();
        },
      );
    });
    if (choice?.reasoning?.length) {
      const effort = effectiveReasoning();
      addPopRow(pop, "Effort", REASONING_LABEL[effort] || effort, () => {
        openSubmenu(
          pop,
          choice.reasoning.map((value) => ({
            value,
            label: REASONING_LABEL[value] || value,
          })),
          effort,
          (value) => {
            state.chat.reasoning = value;
            const rec = ensureConversationRecord(state.chat.conversationId);
            rec.reasoning = value;
            refresh();
            close();
            void persistChatState();
          },
        );
      });
    }
    if (choice?.routes?.length) {
      const route = effectiveRoute();
      addPopRow(pop, "Route", ROUTE_LABEL[route] || route, () => {
        openSubmenu(
          pop,
          choice.routes.map((value) => ({
            value,
            label: ROUTE_LABEL[value] || value,
          })),
          route,
          (value) => {
            state.chat.route = value;
            const rec = ensureConversationRecord(state.chat.conversationId);
            rec.route = value;
            refresh();
            close();
            void persistChatState();
          },
        );
      });
    }
    document.body.appendChild(pop);
    fitModelPop();
    trigger.setAttribute("aria-expanded", "true");
    setIcon(chevron, "chevron-up");
    document.addEventListener("pointerdown", onOutside, true);
    document.addEventListener("keydown", onKey, true);
  };

  trigger.addEventListener("click", (ev) => {
    ev.preventDefault();
    if (pop) close();
    else open();
  });

  refresh();
  return {
    refresh,
    get value() {
      return state.chat.model || "";
    },
  };
}

async function loadChatModels() {
  try {
    const data = await api("GET", "/v1/pawn/models");
    const models = Array.isArray(data?.models) ? data.models : [];
    state.chat.models = models;
    if (state.chat.model && !models.some((m) => m.id === state.chat.model)) {
      state.chat.model = data?.background || data?.default || models[0]?.id || "";
    }
    if (!state.chat.model) {
      state.chat.model = data?.background || data?.default || models[0]?.id || "";
    }
    chatModelSelect?.refresh();
  } catch (_) {
    state.chat.models = [];
    chatModelSelect?.refresh();
  }
}

async function refreshPageContext() {
  try {
    const tab = await findContentTab();
    if (tab?.url && /^https?:/i.test(tab.url)) {
      state.chat.pageUrl = tab.url;
      state.chat.pageTitle = tab.title || "";
    }
  } catch (_) {
    /* ignore */
  }
  renderChatContext();
}

function attachSnippetToChat(snip) {
  if (!snip) return;
  if (state.chat.attachments.some((a) => a.id === snip.id)) return;
  if (snip.kind === "image") {
    const images = state.chat.attachments.filter((a) => a.kind === "image");
    if (images.length >= MAX_CHAT_IMAGES) {
      showNotice(`Chat allows at most ${MAX_CHAT_IMAGES} images`, true);
      return;
    }
  }
  state.chat.attachments.push({
    id: snip.id || newSnippetId(),
    kind: snip.kind === "image" ? "image" : "text",
    text: snip.text || "",
    data_base64: snip.data_base64 || "",
    media_type: snip.media_type || "image/png",
    source_url: snip.source_url || "",
    captured_at: snip.captured_at || new Date().toISOString(),
  });
}

async function attachTrayToChat(index) {
  const snip = state.tray[index];
  if (!snip) return;
  attachSnippetToChat(snip);
  await persistChatState();
  setActiveTab("chat");
  showNotice("Attached to chat");
}

function removeChatAttachment(id) {
  state.chat.attachments = state.chat.attachments.filter((a) => a.id !== id);
  void persistChatState();
  renderChatContext();
  renderChatActions();
}

async function startNewChat() {
  if (state.chat.abort) stopChat();
  snapshotCurrentConversation();
  state.chat.conversationId = newChatId();
  ensureConversationRecord(state.chat.conversationId);
  state.chat.messages = [];
  state.chat.attachments = [];
  await persistChatState();
  renderChat();
  showNotice("New chat");
}

function stopChat() {
  try {
    state.chat.abort?.abort();
  } catch (_) {
    /* ignore */
  }
  state.chat.abort = null;
  syncChatBusy(false);
}

function syncChatBusy(busy) {
  if (el.btnChatSend) {
    el.btnChatSend.disabled = !!busy;
    el.btnChatSend.textContent = busy ? "…" : "Send";
  }
  if (el.btnChatStop) el.btnChatStop.hidden = !busy;
  for (const btn of el.chatActions?.querySelectorAll(".pawn-action-chip") || []) {
    btn.disabled = !!busy;
  }
}

function renderChat() {
  chatConvPicker?.refresh();
  renderChatContext();
  renderChatThread();
  renderChatActions();
}

function renderChatContext() {
  if (!el.chatContext) return;
  el.chatContext.replaceChildren();

  if (state.chat.pageUrl || state.chat.pageTitle) {
    el.chatContext.appendChild(
      makeChip({
        cls: "is-page",
        icon: "link",
        label: state.chat.pageTitle || hostnameOf(state.chat.pageUrl) || "Page",
        title: state.chat.pageUrl || state.chat.pageTitle,
      }),
    );
  }

  // Capture sticky / session (auto) unless the user detached it.
  if (!state.chat.omitBound) {
    const bound = describeBoundContext();
    if (bound && !noteContextHas(bound)) {
      el.chatContext.appendChild(
        makeChip({
          cls: bound.kind === "session" ? "is-session" : "is-note",
          icon: bound.kind === "session" ? "mic" : "file-text",
          label: bound.label,
          title: bound.title,
          onRemove: () => {
            state.chat.omitBound = true;
            void persistChatState();
            renderChatContext();
          },
        }),
      );
    }
  }

  for (const note of state.chat.noteContext) {
    el.chatContext.appendChild(
      makeChip({
        cls: note.kind === "session" ? "is-session" : "is-note",
        icon: note.kind === "session" ? "mic" : "file-text",
        label: note.label || pathBasename(note.path || "").replace(/\.md$/i, "") || "Note",
        title: note.title || note.path || note.sessionId || "",
        onRemove: () => removeNoteContext(note.id),
      }),
    );
  }

  for (const att of state.chat.attachments) {
    const isImg = att.kind === "image";
    el.chatContext.appendChild(
      makeChip({
        cls: isImg ? "is-image" : "is-text",
        icon: isImg ? "image" : "type",
        label: isImg
          ? "Screenshot"
          : (att.text || "Text").replace(/\s+/g, " ").trim().slice(0, 40) || "Text",
        title: isImg ? "Attached image" : att.text || "Attached text",
        onRemove: () => removeChatAttachment(att.id),
      }),
    );
  }

  const add = document.createElement("button");
  add.type = "button";
  add.className = "pawn-chip pawn-chip-add";
  add.setAttribute("aria-label", "Add note or session as context");
  add.title = "Add note or session (@)";
  const addIcon = document.createElement("span");
  addIcon.className = "pawn-chip-icon";
  setIcon(addIcon, "plus");
  add.appendChild(addIcon);
  add.addEventListener("click", () => void openNoteContextPicker());
  el.chatContext.appendChild(add);
}

function makeChip({ cls, icon, label, title, onRemove }) {
  const chip = document.createElement("span");
  chip.className = `pawn-chip ${cls || ""}`.trim();
  if (title) chip.title = title;
  if (icon) {
    const ic = document.createElement("span");
    ic.className = "pawn-chip-icon";
    setIcon(ic, icon);
    chip.appendChild(ic);
  }
  const lab = document.createElement("span");
  lab.className = "pawn-chip-label";
  lab.textContent = label;
  chip.appendChild(lab);
  if (onRemove) {
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "pawn-chip-remove";
    rm.setAttribute("aria-label", "Remove");
    setIcon(rm, "x");
    rm.addEventListener("click", (ev) => {
      ev.preventDefault();
      onRemove();
    });
    chip.appendChild(rm);
  }
  return chip;
}

function describeBoundContext() {
  if (state.stickyPath) {
    return {
      kind: "note",
      path: state.stickyPath,
      label: pathBasename(state.stickyPath).replace(/\.md$/i, "") || state.stickyPath,
      title: `Capture target: ${state.stickyPath}`,
    };
  }
  if (state.targetKind === "session" && state.sessionId) {
    const hit = state.sessions.find((s) => s.session_id === state.sessionId);
    const path = hit?.transcript_path || "";
    return {
      kind: "session",
      path,
      sessionId: state.sessionId,
      label: hit?.title || state.sessionId,
      title: path
        ? `Session ${state.sessionId} · ${path}`
        : `Session ${state.sessionId} (no transcript note yet)`,
    };
  }
  return null;
}

function noteContextHas(ref) {
  return state.chat.noteContext.some((n) => {
    if (ref.path && n.path && n.path === ref.path) return true;
    if (ref.sessionId && n.sessionId && n.sessionId === ref.sessionId) return true;
    return false;
  });
}

function removeNoteContext(id) {
  state.chat.noteContext = state.chat.noteContext.filter((n) => n.id !== id);
  void persistChatState();
  renderChatContext();
}

function addNoteContext(ref) {
  if (!ref) return;
  if (state.chat.noteContext.length >= MAX_NOTE_CONTEXT) {
    showNotice(`At most ${MAX_NOTE_CONTEXT} notes/sessions as context`, true);
    return;
  }
  if (noteContextHas(ref)) {
    showNotice("Already attached");
    return;
  }
  state.chat.noteContext.push({
    id: newSnippetId(),
    kind: ref.kind === "session" ? "session" : "note",
    path: ref.path || "",
    sessionId: ref.sessionId || "",
    label: ref.label || "",
    title: ref.title || ref.path || "",
  });
  // Explicit attach supersedes "omit bound" for overlapping paths.
  if (ref.path && ref.path === state.stickyPath) state.chat.omitBound = false;
  void persistChatState();
  renderChatContext();
  showNotice(`Attached ${ref.label || ref.path || "context"}`);
}

/** Notes/sessions that will be sent on the next chat turn. */
function notesForSend() {
  const out = [];
  const seen = new Set();
  const push = (ref) => {
    if (!ref) return;
    const key = ref.path || (ref.sessionId ? `session:${ref.sessionId}` : "");
    if (!key || seen.has(key)) return;
    seen.add(key);
    out.push(ref);
  };
  for (const n of state.chat.noteContext) push(n);
  if (!state.chat.omitBound) push(describeBoundContext());
  return out;
}

async function openNoteContextPicker() {
  showNotice("Loading notes & sessions…", false, true);
  try {
    const [capData, sessData] = await Promise.all([
      api("GET", "/v1/captures?limit=50").catch(() => ({ data: [] })),
      api("GET", "/v1/sessions?limit=40").catch(() => ({ data: [] })),
    ]);
    state.captures = capData.data || state.captures || [];
    state.sessions = sessData.data || state.sessions || [];
  } catch (_) {
    /* keep cached */
  }
  showNotice("");

  const items = [];
  for (const row of state.sessions || []) {
    const path = row.transcript_path || "";
    const when =
      formatSuggestDate(row.updated_at) || formatSuggestDate(row.created_at) || "";
    const whenLabel = when
      ? row.updated_at
        ? `Updated ${when}`
        : `Created ${when}`
      : "";
    const bits = ["Session", whenLabel, path || "(no transcript yet)"].filter(Boolean);
    items.push({
      id: `session:${row.session_id}`,
      title: row.title || row.session_id,
      detail: bits.join(" · "),
      value: row.session_id,
      kind: "session",
      path,
      sessionId: row.session_id,
      label: row.title || row.session_id,
      updatedAt: row.updated_at || row.created_at || "",
    });
  }
  for (const row of state.captures || []) {
    items.push({
      id: `note:${row.path}`,
      title: row.title || pathBasename(row.path).replace(/\.md$/i, ""),
      detail: row.path,
      value: row.path,
      kind: "note",
      path: row.path,
      label: row.title || pathBasename(row.path).replace(/\.md$/i, ""),
    });
  }

  openSuggestModal({
    placeholder: "Attach a note or session… (type a vault path)",
    items,
    emptyText: "No matches — type a vault path and press Enter",
    allowCustom: true,
    onChoose: async (item) => {
      try {
        if (item.kind === "session") {
          addNoteContext({
            kind: "session",
            path: item.path || "",
            sessionId: item.sessionId || item.value,
            label: item.label || item.title,
            title: item.detail || item.title,
          });
          return;
        }
        if (item.kind === "note" && !item.custom) {
          addNoteContext({
            kind: "note",
            path: item.path || item.value,
            label: item.label || item.title,
            title: item.path || item.value,
          });
          return;
        }
        // Typed vault path (custom) or legacy items without kind.
        let path = String(item.value || item.title || "").trim();
        path = path.replace(/^\[\[|\]\]$/g, "").split("|")[0].trim();
        if (!path) return;
        if (!/\.(md|base|canvas)$/i.test(path)) path = `${path}.md`;
        path = path.replace(/^\/+/, "");
        addNoteContext({
          kind: "note",
          path,
          label: pathBasename(path).replace(/\.md$/i, ""),
          title: path,
        });
      } catch (err) {
        showNotice(String(err?.message || err), true);
      }
    },
  });
}

/** Compact date for suggest rows (Today 14:32, 19 Oct, …). */
function formatSuggestDate(iso) {
  if (!iso) return "";
  const ms = Date.parse(iso);
  if (!Number.isFinite(ms)) return "";
  return formatUpdatedAt(ms);
}

function hostnameOf(url) {
  try {
    return new URL(url).hostname;
  } catch (_) {
    return "";
  }
}

function chatBoundNotePath() {
  if (state.stickyPath) return state.stickyPath;
  if (state.targetKind === "session" && state.sessionId) {
    const hit = state.sessions.find((s) => s.session_id === state.sessionId);
    return hit?.transcript_path || "";
  }
  return "";
}

function renderChatThread() {
  if (!el.chatThread) return;
  el.chatThread.replaceChildren();
  if (!state.chat.messages.length) {
    const empty = document.createElement("p");
    empty.className = "pawn-chat-empty";
    empty.textContent =
      "Attach a selection, screenshot, or note/session (+ or @), then ask Pawn — or use a chip below.";
    el.chatThread.appendChild(empty);
    return;
  }
  for (const msg of state.chat.messages) {
    el.chatThread.appendChild(renderChatMessage(msg));
  }
  el.chatThread.scrollTop = el.chatThread.scrollHeight;
}

function renderChatMessage(msg) {
  const wrap = document.createElement("div");
  wrap.className = `pawn-msg is-${msg.role || "assistant"}`;
  const role = msg.role === "user" ? "user" : msg.error ? "error" : "assistant";

  if (msg.progress && role === "assistant" && !msg.content) {
    const prog = document.createElement("div");
    prog.className = "pawn-msg-progress";
    prog.textContent = msg.progress;
    wrap.appendChild(prog);
  } else if (msg.progress && role === "assistant" && msg.content) {
    const det = document.createElement("details");
    det.className = "pawn-progress-log";
    const sum = document.createElement("summary");
    sum.textContent = "Working…";
    det.appendChild(sum);
    const line = document.createElement("div");
    line.textContent = msg.progress;
    det.appendChild(line);
    wrap.appendChild(det);
  }

  if (msg.error) {
    const err = document.createElement("div");
    err.className = "pawn-msg-error";
    err.textContent = msg.error;
    wrap.appendChild(err);
  }

  if (msg.content) {
    const body = document.createElement("div");
    body.className = "pawn-msg-text" + (role === "assistant" ? " markdown-rendered" : "");
    if (role === "user") {
      body.textContent = msg.content;
    } else {
      body.innerHTML = formatChatHtml(msg.content);
      wireObsidianLinks(body);
    }
    wrap.appendChild(body);
  }

  if (msg.content && !msg.error) {
    if (role === "user") {
      wrap.appendChild(renderUserMsgActions(msg.content));
    } else if (role === "assistant") {
      wrap.appendChild(renderAssistantMsgActions(msg.content));
    }
  }
  return wrap;
}

function renderUserMsgActions(text) {
  const bar = document.createElement("div");
  bar.className = "pawn-msg-actions is-user";
  bar.appendChild(
    iconButton("copy", "Copy", () => {
      void copyText(text);
    }),
  );
  bar.appendChild(
    iconButton("pencil", "Edit in composer", () => {
      injectComposer(text);
    }),
  );
  return bar;
}

function renderAssistantMsgActions(text) {
  const bar = document.createElement("div");
  bar.className = "pawn-msg-actions";
  bar.appendChild(
    iconButton("copy", "Copy", () => {
      void copyText(text);
    }),
  );
  const paths = extractVaultPaths(text);
  if (paths.length && state.vaultName) {
    const first = paths[0];
    bar.appendChild(
      iconButton("external-link", `Open in Obsidian · ${pathBasename(first)}`, () => {
        openVaultPath(first);
      }),
    );
  } else if (paths.length) {
    bar.appendChild(
      iconButton("external-link", "Open in Obsidian (set vault.obsidian_vault_name)", () => {
        showNotice("Set vault.obsidian_vault_name on the server for obsidian:// links", true);
      }),
    );
  }
  bar.appendChild(
    iconButton("list-todo", "Follow-up: TaskNotes TODO", () => {
      injectComposer(
        "Create a TaskNotes TODO from your last answer. Use a clear title and short body.\n\n",
      );
    }),
  );
  bar.appendChild(
    iconButton("pencil", "Follow-up in composer", () => {
      injectComposer("");
      el.chatInput?.focus();
    }),
  );
  return bar;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text || "");
    showNotice("Copied");
  } catch (_) {
    showNotice("Copy failed", true);
  }
}

function injectComposer(text) {
  if (!el.chatInput) return;
  el.chatInput.value = text || "";
  el.chatInput.focus();
  const len = el.chatInput.value.length;
  el.chatInput.setSelectionRange(len, len);
  el.chatInput.scrollIntoView({ block: "nearest" });
}

function openVaultPath(path) {
  const url = buildObsidianOpenUrl(state.vaultName, path);
  if (!url) {
    showNotice("Set vault.obsidian_vault_name on the server", true);
    return;
  }
  void chrome.tabs.create({ url });
}

/** Vault paths and wikilink targets from assistant markdown. */
function extractVaultPaths(md) {
  const found = [];
  const seen = new Set();
  const push = (p) => {
    const path = String(p || "")
      .trim()
      .replace(/\\/g, "/")
      .replace(/^\/+/, "");
    if (!path || seen.has(path)) return;
    seen.add(path);
    found.push(path);
  };
  const wiki = /\[\[([^\]|#]+)(?:\|[^\]]+)?\]\]/g;
  let m;
  while ((m = wiki.exec(md || ""))) push(m[1].trim());
  const pathRe =
    /["']?([A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_ .'-]+)+\.(?:md|base|canvas))["']?/gi;
  while ((m = pathRe.exec(md || ""))) push(m[1]);
  return found;
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatChatHtml(text) {
  const raw = String(text || "");
  const blocks = [];
  let work = raw.replace(/```([\w-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const i = blocks.length;
    blocks.push(
      `<pre class="pawn-md-pre"><code class="language-${escapeHtml(lang || "")}">${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`,
    );
    return `\u0000BLOCK${i}\u0000`;
  });

  work = escapeHtml(work);

  // Wikilinks [[path]] or [[path|alias]]
  work = work.replace(/\[\[([^\]|#\n]+)(?:\|([^\]]+))?\]\]/g, (_, path, alias) => {
    const p = path.trim();
    const label = (alias || p).trim();
    return obsidianAnchorHtml(p, label);
  });

  // Bare vault paths (Pawn/.../file.md)
  work = work.replace(
    /(["']?)([A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_ .'-]+)+\.(?:md|base|canvas))\1/g,
    (full, _q, path, offset, whole) => {
      const before = whole[offset - 1];
      const after = whole[offset + full.length];
      if (before && /[\w./-]/.test(before)) return full;
      if (after && /[\w/-]/.test(after)) return full;
      if (full.includes("data-vault-path")) return full;
      return obsidianAnchorHtml(path, path);
    },
  );

  // Inline code
  work = work.replace(/`([^`\n]+)`/g, (_, code) => `<code class="pawn-md-code">${code}</code>`);

  // Bold / italic
  work = work.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  work = work.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");

  // Task / bullet lines
  work = work.replace(
    /(^|\n)([-*]|\d+\.) \[([ xX])\] (.+?)(?=\n|$)/g,
    (_, br, _m, checked, body) =>
      `${br}<div class="pawn-md-li">${checked === " " ? "☐" : "☑"} ${body}</div>`,
  );
  work = work.replace(
    /(^|\n)([-*]) (.+?)(?=\n|$)/g,
    (_, br, _m, body) => `${br}<div class="pawn-md-li">• ${body}</div>`,
  );

  // External http(s) links (skip ones already in anchors)
  work = work.replace(
    /(?<!["|=])(https?:\/\/[^\s<]+)/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>',
  );

  work = work.replace(/\n\n+/g, "</p><p>");
  work = work.replace(/\n/g, "<br>");
  work = `<p>${work}</p>`;

  work = work.replace(/\u0000BLOCK(\d+)\u0000/g, (_, i) => blocks[Number(i)] || "");
  return work;
}

function obsidianAnchorHtml(path, label) {
  const p = String(path || "")
    .trim()
    .replace(/\\/g, "/")
    .replace(/^\/+/, "");
  const lab = escapeHtml(label || p);
  const url = buildObsidianOpenUrl(state.vaultName, p);
  if (url) {
    return `<a class="pawn-vault-link" href="${escapeHtml(url)}" data-vault-path="${escapeHtml(p)}" title="Open in Obsidian">${lab}</a>`;
  }
  return `<a class="pawn-vault-link is-disabled" href="#" data-vault-path="${escapeHtml(p)}" title="Set vault.obsidian_vault_name to open in Obsidian">${lab}</a>`;
}

function wireObsidianLinks(root) {
  if (!root) return;
  for (const a of root.querySelectorAll("a.pawn-vault-link")) {
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      const path = a.getAttribute("data-vault-path") || "";
      const href = a.getAttribute("href") || "";
      if (href && href.startsWith("obsidian://")) {
        void chrome.tabs.create({ url: href });
        return;
      }
      if (path) openVaultPath(path);
      else showNotice("Set vault.obsidian_vault_name on the server", true);
    });
  }
}

function renderChatActions() {
  if (!el.chatActions) return;
  el.chatActions.replaceChildren();
  const busy = !!state.chat.abort;
  const hasAtt = state.chat.attachments.length > 0;
  const hasPage = !!(state.chat.pageUrl || state.chat.pageTitle);
  const hasSession = state.targetKind === "session" && !!state.sessionId;
  const hasCtx = hasAtt || hasPage;
  // Chips inject into the composer so the user can steer attached context.
  const actions = [
    {
      id: "todo",
      label: "Create TODO",
      enabled: hasCtx,
      inject:
        "Create a TaskNotes TODO from the attached context (and page if present). Title: ",
    },
    {
      id: "idea",
      label: "/idea",
      enabled: hasCtx,
      inject: "/idea ",
    },
    {
      id: "summarize",
      label: "Summarize",
      enabled: hasCtx,
      inject:
        "Summarize the attached clips and page context. Keep it concise and cite the source URL when useful.\n\n",
    },
    {
      id: "goal",
      label: "/goal",
      enabled: true,
      inject: "/goal ",
    },
    {
      id: "session",
      label: "Use session",
      enabled: hasSession && hasCtx,
      inject: `Relate the attached context to diarization session ${state.sessionId}. Extract action items.\n\n`,
    },
  ];
  for (const a of actions) {
    if (!a.enabled) continue;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pawn-action-chip";
    btn.textContent = a.label;
    btn.title = "Inject into composer — edit, then Send";
    btn.disabled = busy;
    btn.addEventListener("click", () => injectComposer(a.inject));
    el.chatActions.appendChild(btn);
  }
}

async function sendChat(overrideMessage) {
  if (state.chat.abort) return;
  const typed = (overrideMessage != null ? overrideMessage : el.chatInput?.value || "").trim();
  const images = state.chat.attachments.filter((a) => a.kind === "image" && a.data_base64);
  const texts = state.chat.attachments.filter((a) => a.kind === "text" && (a.text || "").trim());
  if (!typed && !images.length && !texts.length) {
    showNotice("Type a message or attach a clip first", true);
    return;
  }

  await refreshPageContext();

  let model = state.chat.model || undefined;
  if (images.length) {
    const vision = state.chat.models.find((m) => m.vision);
    const current = state.chat.models.find((m) => m.id === model);
    if (model && current && !current.vision && vision) {
      model = vision.id;
      showNotice(`Using vision model ${model}`);
    } else if (!model && vision) {
      model = vision.id;
    } else if (!vision && images.length) {
      showNotice("No vision model available — images may be ignored", true);
    }
  }

  const selectionParts = [];
  for (const t of texts) selectionParts.push(t.text.trim());
  if (state.chat.pageUrl) {
    selectionParts.push(`Page: ${state.chat.pageTitle || ""} ${state.chat.pageUrl}`.trim());
  }
  let selection = selectionParts.join("\n\n").slice(0, 6000) || undefined;

  let message = typed;
  if (!message && images.length) message = "Look at this image.";
  if (!message && texts.length) message = "Process the attached text.";

  const conversation = chatConversationKey();
  const tuning = tuningFields();
  const notes = notesForSend();
  const pathNotes = notes.filter((n) => n.path);
  const sessionOnly = notes.filter((n) => n.kind === "session" && !n.path && n.sessionId);
  if (sessionOnly.length) {
    const hint = sessionOnly
      .map((n) => `Diarization session ${n.sessionId}${n.label ? ` (${n.label})` : ""}`)
      .join("; ");
    selection = [selection, `Attached sessions (use sessions_list / session_transcript): ${hint}`]
      .filter(Boolean)
      .join("\n\n")
      .slice(0, 6000);
  }
  const body = {
    conversation,
    message,
    selection,
    model,
    images: images.slice(0, MAX_CHAT_IMAGES).map((img, i) => ({
      filename: `clip-${i + 1}.${extForMime(img.media_type)}`,
      media_type: img.media_type || "image/png",
      data_base64: img.data_base64,
      role: "question",
    })),
  };
  if (tuning.reasoning) body.reasoning = tuning.reasoning;
  if (tuning.route) body.route = tuning.route;
  if (pathNotes.length) {
    body.active_note = { path: pathNotes[0].path, inline: false };
    body.context = pathNotes.slice(1).map((n) => ({ path: n.path, inline: false }));
  }

  const userMsg = {
    id: newSnippetId(),
    role: "user",
    content: message,
    at: new Date().toISOString(),
  };
  const assistantMsg = {
    id: newSnippetId(),
    role: "assistant",
    content: "",
    progress: "Thinking…",
    at: new Date().toISOString(),
  };
  state.chat.messages.push(userMsg, assistantMsg);
  if (overrideMessage == null && el.chatInput) el.chatInput.value = "";
  // Clear one-shot attachments after send (like Obsidian pending images).
  state.chat.attachments = [];
  await persistChatState();
  renderChat();
  syncChatBusy(true);

  const abort = new AbortController();
  state.chat.abort = abort;
  try {
    await streamPawnChat(body, abort.signal, {
      onProgress(text) {
        assistantMsg.progress = text || "Working…";
        renderChatThread();
      },
      onAnswer(content) {
        assistantMsg.content = content || "";
        assistantMsg.progress = "";
        renderChatThread();
      },
      onJob(job) {
        const id = job?.id || "";
        assistantMsg.content =
          (assistantMsg.content ? `${assistantMsg.content}\n\n` : "") +
          (id ? `Background job ${id} queued.` : "Background job queued.");
        assistantMsg.progress = "";
        renderChatThread();
      },
      onError(message) {
        assistantMsg.error = message || "Chat failed";
        assistantMsg.progress = "";
        renderChatThread();
      },
    });
    if (!assistantMsg.content && !assistantMsg.error) {
      assistantMsg.progress = "";
      if (!assistantMsg.content) assistantMsg.content = "(no response)";
    }
  } catch (err) {
    if (abort.signal.aborted) {
      assistantMsg.progress = "";
      if (!assistantMsg.content) assistantMsg.content = "(stopped)";
    } else {
      assistantMsg.error = String(err.message || err);
      assistantMsg.progress = "";
    }
  } finally {
    state.chat.abort = null;
    syncChatBusy(false);
    await persistChatState();
    renderChat();
  }
}

function chatConversationKey() {
  // Always use chat:{uuid} so New chat resets memory. Bound vault notes are
  // passed as active_note (inline:false) so the agent can note_read them.
  if (!state.chat.conversationId) state.chat.conversationId = newChatId();
  return state.chat.conversationId;
}

function extForMime(mime) {
  const m = String(mime || "").toLowerCase();
  if (m.includes("jpeg") || m.includes("jpg")) return "jpg";
  if (m.includes("webp")) return "webp";
  if (m.includes("gif")) return "gif";
  return "png";
}

/** Incremental SSE parser (keep-alives / comments dropped). */
class SseParser {
  constructor() {
    this.buffer = "";
  }
  push(chunk) {
    this.buffer += String(chunk || "").replace(/\r\n/g, "\n");
    const out = [];
    let idx;
    while ((idx = this.buffer.indexOf("\n\n")) >= 0) {
      const block = this.buffer.slice(0, idx);
      this.buffer = this.buffer.slice(idx + 2);
      const ev = parseSseBlock(block);
      if (ev) out.push(ev);
    }
    return out;
  }
  flush() {
    const rest = this.buffer;
    this.buffer = "";
    const ev = rest.trim() ? parseSseBlock(rest) : null;
    return ev ? [ev] : [];
  }
}

function parseSseBlock(block) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon >= 0 ? line.slice(0, colon) : line;
    let value = colon >= 0 ? line.slice(colon + 1) : "";
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  if (!data.length) return null;
  return { event, data: data.join("\n") };
}

function dispatchChatEvent(ev, handlers) {
  let data = {};
  try {
    data = JSON.parse(ev.data);
  } catch (_) {
    return;
  }
  switch (ev.event) {
    case "progress":
      handlers.onProgress?.(String(data.text ?? ""));
      break;
    case "answer":
      handlers.onAnswer?.(String(data.content ?? ""));
      break;
    case "job":
      handlers.onJob?.(data);
      break;
    case "error":
      handlers.onError?.(String(data.message ?? "Unknown error"));
      break;
    default:
      break;
  }
}

async function streamPawnChat(body, signal, handlers) {
  const base = state.settings.serverUrl.replace(/\/+$/, "");
  const headers = {
    Accept: "text/event-stream",
    "Content-Type": "application/json",
  };
  if (state.settings.apiToken) {
    headers.Authorization = `Bearer ${state.settings.apiToken}`;
  }
  const resp = await fetch(`${base}/v1/pawn/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok) {
    const text = await resp.text();
    let detail = text;
    try {
      const j = JSON.parse(text);
      detail = j.detail || j.message || text;
    } catch (_) {
      /* keep text */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (!resp.body || typeof resp.body.getReader !== "function") {
    const text = await resp.text();
    const parser = new SseParser();
    for (const ev of [...parser.push(text), ...parser.flush()]) {
      dispatchChatEvent(ev, handlers);
    }
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const ev of parser.push(decoder.decode(value, { stream: true }))) {
      dispatchChatEvent(ev, handlers);
    }
  }
  for (const ev of parser.flush()) dispatchChatEvent(ev, handlers);
}
