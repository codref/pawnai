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
    { path: "/:pathMatch(.*)*", redirect: "/" },
  ],
});

/** pin: keep the tab (otherwise it is the group's preview tab); edit: files
 * open in the editor; where: current group, a new split, or a floating window. */
export type OpenOptions = OpenOpts;

function routeFor(tab: { kind: TabKind; ref: string } | null): RouteLocationRaw {
  if (!tab) return { name: "home" };
  if (tab.kind === "chat") return { name: "chat", params: { id: tab.ref } };
  if (tab.kind === "clip") return { name: "clip", params: { id: tab.ref } };
  if (tab.kind === "inbox") return { name: "inbox" };
  if (tab.kind === "settings") return { name: "settings" };
  return { name: "file", query: { key: tab.ref } };
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
  } else if (to.name === "inbox" || to.name === "settings") {
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

export function openTarget(kind: TabKind, ref: string, opts: OpenOptions = {}): void {
  if (kind === "chat") openChat(ref, opts);
  else if (kind === "clip") openClip(ref, opts);
  else if (kind === "inbox") openInbox(opts);
  else if (kind === "settings") openSettings(opts);
  else openFile(ref, opts);
}

/** Hash URL for a chat / file / clip (Copy link). */
export function linkFor(kind: TabKind, ref: string): string {
  const href = router.resolve(routeFor({ kind, ref })).href;
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
