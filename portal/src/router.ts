import { reactive } from "vue";
import { createRouter, createWebHashHistory, type RouteLocationRaw } from "vue-router";
import { useTabsStore, tabId, TOOL_REF, type OpenOpts, type Tab, type TabKind } from "@/stores/tabs";

// Hash history: pawn-server serves dist/ with StaticFiles, which has no SPA
// fallback for deep paths. The center area is a tab strip (stores/tabs.ts);
// the URL mirrors the active tab so links and reloads land on it.
const Empty = { render: () => null };

export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", name: "home", component: Empty },
    { path: "/chat/:id", name: "chat", component: Empty },
    { path: "/file", name: "file", component: Empty },
    { path: "/clip/:id", name: "clip", component: Empty },
    { path: "/inbox", name: "inbox", component: Empty },
    { path: "/settings", name: "settings", component: Empty },
    { path: "/sessions", name: "sessions", component: Empty },
    { path: "/session/:id", name: "session", component: Empty },
    { path: "/day", name: "day", component: Empty },
    { path: "/speakers", name: "speakers", component: Empty },
    { path: "/assistants", name: "assistants", component: Empty },
    { path: "/brainmap/:board", name: "brainmap", component: Empty },
    { path: "/:pathMatch(.*)*", redirect: "/" },
  ],
});

export type DayView = "day" | "week";

/** Date, grid and focused event for the singleton Day tab. The URL query mirrors this. */
export const dayRoute = reactive<{ date: string; view: DayView; event: string }>({
  date: "",
  view: "day",
  event: "",
});

/** Selected speaker inside the Speakers tool tab (`?id=`). */
export const speakersRoute = reactive<{ id: string }>({ id: "" });

export type SessionSection = "transcript" | "summary" | "items";

/** Which section the active session tab is showing. `?section=` mirrors items and summary. */
export const sessionRoute = reactive<{ id: string; section: SessionSection | ""; seek: number | null }>({
  id: "",
  section: "",
  seek: null,
});

/** Brainmap board tab: the card to fly to when it opens (`?focus=`). */
export const brainmapRoute = reactive<{ board: string; focus: string }>({ board: "", focus: "" });

export interface DayOpen {
  date?: string;
  view?: DayView;
  eventId?: string;
}

function queryString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function todayStamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

function dayQuery(): Record<string, string> {
  const q: Record<string, string> = {};
  if (dayRoute.date) q.date = dayRoute.date;
  if (dayRoute.view === "week") q.view = "week";
  if (dayRoute.event) q.event = dayRoute.event;
  return q;
}

/** pin: keep the tab (otherwise it is the group's preview tab); edit: files
 * open in the editor; where: current group, a new split, or a floating window. */
export type OpenOptions = OpenOpts;

function locationFor(kind: string, ref: string): RouteLocationRaw {
  if (kind === "chat") return { name: "chat", params: { id: ref } };
  if (kind === "clip") return { name: "clip", params: { id: ref } };
  if (kind === "inbox") return { name: "inbox" };
  if (kind === "settings") return { name: "settings" };
  if (kind === "sessions") return { name: "sessions" };
  if (kind === "session") {
    const query: Record<string, string> = {};
    if (
      ref === sessionRoute.id &&
      (sessionRoute.section === "items" || sessionRoute.section === "summary")
    ) {
      query.section = sessionRoute.section;
    }
    return { name: "session", params: { id: ref }, query };
  }
  if (kind === "day") {
    const date = ref && ref !== TOOL_REF ? ref : dayRoute.date;
    const q = { ...dayQuery() };
    if (date) q.date = date;
    else delete q.date;
    if (ref && ref !== TOOL_REF && ref !== dayRoute.date) delete q.event;
    return { name: "day", query: q };
  }
  if (kind === "event") return { name: "day", query: { ...dayQuery(), event: ref } };
  if (kind === "speakers") {
    const id = ref && ref !== TOOL_REF ? ref : speakersRoute.id;
    return { name: "speakers", query: id ? { id } : {} };
  }
  if (kind === "speaker") return { name: "speakers", query: { id: ref } };
  if (kind === "assistants") return { name: "assistants" };
  if (kind === "brainmap") {
    const board = ref || "items";
    const query: Record<string, string> = {};
    if (brainmapRoute.board === board && brainmapRoute.focus) query.focus = brainmapRoute.focus;
    return { name: "brainmap", params: { board }, query };
  }
  if (kind === "file") return { name: "file", query: { key: ref } };
  return { name: "home" };
}

function routeFor(tab: { kind: TabKind; ref: string } | null): RouteLocationRaw {
  if (!tab) return { name: "home" };
  return locationFor(tab.kind, tab.ref);
}

function sync(tab: Tab | null): void {
  void router.replace(routeFor(tab));
}

// Typed URL / back-forward / reload: open what the URL names (as a preview).
router.afterEach((to) => {
  const tabs = useTabsStore();
  if (to.name === "chat" && to.params.id) {
    const id = String(to.params.id);
    if (tabs.activeId !== tabId("chat", id)) tabs.open("chat", id);
  } else if (to.name === "file" && to.query.key) {
    const key = String(to.query.key);
    if (tabs.activeId !== tabId("file", key)) tabs.open("file", key);
  } else if (to.name === "clip" && to.params.id) {
    const id = String(to.params.id);
    if (tabs.activeId !== tabId("clip", id)) tabs.open("clip", id);
  } else if (to.name === "session" && to.params.id) {
    const id = String(to.params.id);
    const section = queryString(to.query.section);
    sessionRoute.id = id;
    sessionRoute.section = section === "items" || section === "summary" ? section : "";
    if (tabs.activeId !== tabId("session", id)) tabs.open("session", id);
  } else if (to.name === "brainmap" && to.params.board) {
    const board = String(to.params.board);
    brainmapRoute.board = board;
    brainmapRoute.focus = queryString(to.query.focus);
    if (tabs.activeId !== tabId("brainmap", board)) tabs.open("brainmap", board, { pin: true });
  } else if (to.name === "inbox" || to.name === "settings" || to.name === "sessions" || to.name === "day" || to.name === "speakers" || to.name === "assistants") {
    if (to.name === "day") {
      const date = queryString(to.query.date);
      const view = queryString(to.query.view);
      const event = queryString(to.query.event);
      if (date) dayRoute.date = date;
      dayRoute.view = view === "week" ? "week" : "day";
      dayRoute.event = event;
    } else if (to.name === "speakers") {
      speakersRoute.id = queryString(to.query.id);
    }
    const kind = to.name as TabKind;
    if (tabs.activeId !== tabId(kind, TOOL_REF)) tabs.open(kind, TOOL_REF, { pin: true });
  } else if (to.name === "home" && tabs.active) {
    sync(tabs.active);
  }
});

export function openChat(id: string, opts: OpenOptions = {}): void {
  const tab = useTabsStore().open("chat", id, opts);
  void router.push(routeFor(tab));
}

export function openFile(key: string, opts: OpenOptions = {}): void {
  const tab = useTabsStore().open("file", key, opts);
  void router.push(routeFor(tab));
}

export function openClip(id: string, opts: OpenOptions = {}): void {
  const tab = useTabsStore().open("clip", id, opts);
  void router.push(routeFor(tab));
}

/** The Inbox tool tab (always kept open, never a preview). */
export function openInbox(opts: OpenOptions = {}): void {
  const tab = useTabsStore().open("inbox", TOOL_REF, { ...opts, pin: true });
  void router.push(routeFor(tab));
}

export function openSettings(opts: OpenOptions = {}): void {
  const tab = useTabsStore().open("settings", TOOL_REF, { ...opts, pin: true });
  void router.push(routeFor(tab));
}

/** Sessions tool tab (always kept, never a preview). */
export function openSessions(opts: OpenOptions = {}): void {
  const tab = useTabsStore().open("sessions", TOOL_REF, { ...opts, pin: true });
  void router.push(routeFor(tab));
}

export function openSession(id: string, opts: OpenOptions = {}, section?: SessionSection, seek?: number): void {
  sessionRoute.seek = seek ?? null;
  if (seek !== undefined && seek !== null) {
    sessionRoute.id = id;
    sessionRoute.section = "";
  } else if (section) {
    sessionRoute.id = id;
    sessionRoute.section = section;
  } else if (sessionRoute.id !== id) {
    sessionRoute.id = id;
    sessionRoute.section = "";
  }
  const tab = useTabsStore().open("session", id, opts);
  void router.push(routeFor(tab));
}

/** Day tool tab. The date, week/day grid and focused event stay in the URL query. */
export function openDay(query: DayOpen = {}, opts: OpenOptions = {}): void {
  if (query.date !== undefined) dayRoute.date = query.date;
  if (!dayRoute.date) dayRoute.date = todayStamp();
  if (query.view !== undefined) dayRoute.view = query.view;
  if (query.eventId !== undefined) dayRoute.event = query.eventId;
  const tab = useTabsStore().open("day", TOOL_REF, { ...opts, pin: true });
  void router.push(routeFor(tab));
}

/** Speakers tool tab. *id* selects a row; omit it to keep the current selection. */
export function openSpeakers(id?: string, opts: OpenOptions = {}): void {
  if (id !== undefined) speakersRoute.id = id;
  const tab = useTabsStore().open("speakers", TOOL_REF, { ...opts, pin: true });
  void router.push(routeFor(tab));
}

/** Assistants tool tab (voice assistants: phone, laptop). */
export function openAssistants(opts: OpenOptions = {}): void {
  const tab = useTabsStore().open("assistants", TOOL_REF, { ...opts, pin: true });
  void router.push(routeFor(tab));
}

/** A brainmap board (one tab per board). *focus* is a node id to fly to. */
export function openBrainmap(board = "items", focus = "", opts: OpenOptions = {}): void {
  brainmapRoute.board = board;
  brainmapRoute.focus = focus;
  const tab = useTabsStore().open("brainmap", board, { ...opts, pin: true });
  void router.push(routeFor(tab));
}

export function openTarget(kind: TabKind, ref: string, opts: OpenOptions = {}): void {
  if (kind === "chat") openChat(ref, opts);
  else if (kind === "clip") openClip(ref, opts);
  else if (kind === "inbox") openInbox(opts);
  else if (kind === "settings") openSettings(opts);
  else if (kind === "sessions") openSessions(opts);
  else if (kind === "session") openSession(ref, opts);
  else if (kind === "day") openDay(ref && ref !== TOOL_REF ? { date: ref } : {}, opts);
  else if (kind === "speakers") openSpeakers(ref && ref !== TOOL_REF ? ref : undefined, opts);
  else if (kind === "assistants") openAssistants(opts);
  else if (kind === "brainmap") openBrainmap(ref || "items", "", opts);
  else openFile(ref, opts);
}

/** Hash URL for a tab, or for an event / speaker target. */
export function linkFor(kind: string, ref: string): string {
  const href = router.resolve(locationFor(kind, ref)).href;
  return new URL(href, window.location.href).toString();
}

export function activateTab(id: string): void {
  const tabs = useTabsStore();
  if (!tabs.find(id)) return;
  tabs.activate(id);
  sync(tabs.active);
}

/** Focus a group (click inside it): the URL follows its active tab. */
export function focusGroup(groupId: string): void {
  const tabs = useTabsStore();
  if (tabs.focusedId === groupId) return;
  tabs.focusGroup(groupId);
  sync(tabs.active);
}

/** Re-sync the URL after a layout change (split / float / dock / move). */
export function syncRoute(): void {
  sync(useTabsStore().active);
}

export function closeTab(id: string): void {
  sync(useTabsStore().close(id));
}

export function closeTabs(ids: string[]): void {
  sync(useTabsStore().closeMany(ids));
}

/** A chat or file is gone: close its tab if open. */
export function closeRef(kind: TabKind, ref: string): void {
  const tabs = useTabsStore();
  const id = tabId(kind, ref);
  if (tabs.find(id)) closeTab(id);
}

/** A file moved: keep its tab on the new key. */
export function retargetFile(oldKey: string, newKey: string): void {
  const tabs = useTabsStore();
  tabs.retarget("file", oldKey, newKey);
  if (tabs.active?.kind === "file") sync(tabs.active);
}
