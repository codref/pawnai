import { defineStore } from "pinia";
import { computed, ref, watch } from "vue";
import { api } from "@/api";
import { loadJson, saveJson } from "@/lib/storage";

// The center workspace: tabs live in groups. Docked groups form a split tree
// (`root`); floating groups sit above it with their own rect. A tab id is
// `${kind}:${ref}`, unique across the whole workspace, so opening something
// that is already open anywhere just focuses it.

export type TabKind =
  | "chat"
  | "file"
  | "clip"
  | "inbox"
  | "settings"
  | "sessions"
  | "session"
  | "day"
  | "speakers"
  | "assistants"
  | "brainmap";
const KINDS: readonly TabKind[] = [
  "chat",
  "file",
  "clip",
  "inbox",
  "settings",
  "sessions",
  "session",
  "day",
  "speakers",
  "assistants",
  "brainmap",
];
/** Singleton tool tabs (one per workspace, ref is always "main"). */
export const TOOL_REF = "main";

export interface Tab {
  id: string;
  kind: TabKind;
  ref: string;
  /** False for the single "preview" tab per group that the next open replaces. */
  pinned: boolean;
  /** Pinned to the start of its group: no close button, survives "close others". */
  sticky: boolean;
  groupId: string;
  /** Open a file straight in the editor. */
  edit?: boolean;
}

export interface TabStatus {
  title?: string;
  dirty?: boolean;
  busy?: boolean;
}

export interface FloatRect {
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
  max?: boolean;
}

export interface Group {
  id: string;
  activeId: string | null;
  /** Set for floating groups (not in the split tree). */
  float: FloatRect | null;
}

export type SplitDir = "row" | "column";
export type LayoutNode =
  | { type: "group"; id: string }
  | { type: "split"; id: string; dir: SplitDir; sizes: number[]; children: LayoutNode[] };

export type Side = "left" | "right" | "top" | "bottom";
export type Where = "tab" | "split" | "float";

export interface OpenOpts {
  pin?: boolean;
  edit?: boolean;
  where?: Where;
  /** Floats: where to put the window (workspace-relative px). */
  at?: { x: number; y: number };
}

interface SavedTab {
  id: string;
  kind: TabKind;
  ref: string;
  pinned: boolean;
  sticky?: boolean;
  groupId?: string;
}

interface SavedV2 {
  version: 2;
  tabs: SavedTab[];
  groups: Group[];
  root: LayoutNode;
  focused: string | null;
}

interface SavedV1 {
  tabs?: SavedTab[];
  active?: string | null;
}

export function tabId(kind: TabKind, ref: string): string {
  return `${kind}:${ref}`;
}

const MAX_TABS = 40;
const FLOAT_W = 520;
const FLOAT_H = 560;

let idCounter = 0;
function newId(prefix: string): string {
  return `${prefix}${Date.now().toString(36)}${(idCounter++).toString(36)}`;
}

function groupIdsIn(node: LayoutNode, out: string[] = []): string[] {
  if (node.type === "group") out.push(node.id);
  else node.children.forEach((c) => groupIdsIn(c, out));
  return out;
}

function normSizes(sizes: number[], n: number): number[] {
  const s = sizes.slice(0, n).map((x) => (Number.isFinite(x) && x > 0 ? x : 1 / n));
  while (s.length < n) s.push(1 / n);
  const sum = s.reduce((a, b) => a + b, 0) || 1;
  return s.map((x) => x / sum);
}

/** Drop unknown/duplicate group leaves and collapse single-child splits. */
function cleanTree(node: LayoutNode, keep: Set<string>, seen: Set<string>): LayoutNode | null {
  if (node.type === "group") {
    if (!keep.has(node.id) || seen.has(node.id)) return null;
    seen.add(node.id);
    return node;
  }
  const kids: LayoutNode[] = [];
  const sizes: number[] = [];
  node.children.forEach((c, i) => {
    const k = cleanTree(c, keep, seen);
    if (k) {
      kids.push(k);
      sizes.push(node.sizes[i] ?? 1);
    }
  });
  if (!kids.length) return null;
  if (kids.length === 1) return kids[0];
  // Flatten nested splits in the same direction.
  const flatKids: LayoutNode[] = [];
  const flatSizes: number[] = [];
  const total = sizes.reduce((a, b) => a + b, 0) || 1;
  kids.forEach((k, i) => {
    const share = sizes[i] / total;
    if (k.type === "split" && k.dir === node.dir) {
      k.children.forEach((kk, j) => {
        flatKids.push(kk);
        flatSizes.push(share * (k.sizes[j] ?? 1 / k.children.length));
      });
    } else {
      flatKids.push(k);
      flatSizes.push(share);
    }
  });
  return { ...node, children: flatKids, sizes: normSizes(flatSizes, flatKids.length) };
}

export const useTabsStore = defineStore("tabs", () => {
  const tabs = ref<Tab[]>([]);
  const groups = ref<Group[]>([]);
  const root = ref<LayoutNode>({ type: "group", id: "g0" });
  const focusedId = ref<string>("g0");
  /** Last docked group that had focus: where floats dock back to. */
  const lastDocked = ref<string>("g0");
  const status = ref<Record<string, TabStatus>>({});
  let zTop = 10;
  let serverTimer: ReturnType<typeof setTimeout> | undefined;
  let hydrated = false;

  // ── derived ─────────────────────────────────────────────────────────────

  const focused = computed(() => groups.value.find((g) => g.id === focusedId.value) ?? null);
  const activeId = computed(() => focused.value?.activeId ?? null);
  const active = computed(() => tabs.value.find((t) => t.id === activeId.value) ?? null);
  const docked = computed(() => groupIdsIn(root.value));
  const floats = computed(() => groups.value.filter((g) => g.float));
  /** Tab ids that are visible right now (active in some group). */
  const visibleIds = computed(() => new Set(groups.value.map((g) => g.activeId).filter(Boolean)));

  function find(id: string): Tab | undefined {
    return tabs.value.find((t) => t.id === id);
  }

  function group(id: string): Group | undefined {
    return groups.value.find((g) => g.id === id);
  }

  /** A group's tabs in display order: sticky first, then the rest. */
  function tabsIn(groupId: string): Tab[] {
    const mine = tabs.value.filter((t) => t.groupId === groupId);
    return [...mine.filter((t) => t.sticky), ...mine.filter((t) => !t.sticky)];
  }

  // ── groups / layout ─────────────────────────────────────────────────────

  function addGroup(float: FloatRect | null = null): Group {
    const g: Group = { id: newId("g"), activeId: null, float };
    groups.value.push(g);
    return g;
  }

  function focusGroup(id: string): void {
    const g = group(id);
    if (!g) return;
    focusedId.value = id;
    if (g.float) raise(id);
    else lastDocked.value = id;
  }

  function raise(id: string): void {
    const g = group(id);
    if (g?.float && g.float.z !== zTop) g.float = { ...g.float, z: ++zTop };
  }

  function setFloat(id: string, patch: Partial<FloatRect>): void {
    const g = group(id);
    if (g?.float) g.float = { ...g.float, ...patch };
  }

  function setSizes(splitId: string, sizes: number[]): void {
    const walk = (n: LayoutNode): boolean => {
      if (n.type !== "split") return false;
      if (n.id === splitId) {
        n.sizes = normSizes(sizes, n.children.length);
        return true;
      }
      return n.children.some(walk);
    };
    walk(root.value);
  }

  /** Insert *newGroupId* next to *targetId* in the split tree. */
  function insertBeside(targetId: string, newGroupId: string, side: Side): void {
    const dir: SplitDir = side === "left" || side === "right" ? "row" : "column";
    const before = side === "left" || side === "top";
    const leaf: LayoutNode = { type: "group", id: newGroupId };
    const place = (n: LayoutNode): LayoutNode => {
      if (n.type === "group") {
        if (n.id !== targetId) return n;
        return {
          type: "split",
          id: newId("s"),
          dir,
          sizes: [0.5, 0.5],
          children: before ? [leaf, n] : [n, leaf],
        };
      }
      const idx = n.children.findIndex((c) => c.type === "group" && c.id === targetId);
      if (idx >= 0 && n.dir === dir) {
        const children = [...n.children];
        const sizes = [...n.sizes];
        const half = (sizes[idx] ?? 1 / children.length) / 2;
        sizes[idx] = half;
        const at = before ? idx : idx + 1;
        children.splice(at, 0, leaf);
        sizes.splice(at, 0, half);
        return { ...n, children, sizes: normSizes(sizes, children.length) };
      }
      return { ...n, children: n.children.map(place) };
    };
    root.value = place(root.value);
  }

  /** Remove empty groups, keep at least one docked group, fix focus. */
  function prune(): void {
    const used = new Set(tabs.value.map((t) => t.groupId));
    const dockedIds = groupIdsIn(root.value);
    let keepDocked = dockedIds.filter((id) => used.has(id));
    if (!keepDocked.length) keepDocked = [dockedIds.includes(lastDocked.value) ? lastDocked.value : dockedIds[0]];
    const keep = new Set<string>([
      ...keepDocked,
      ...groups.value.filter((g) => g.float && used.has(g.id)).map((g) => g.id),
    ]);
    groups.value = groups.value.filter((g) => keep.has(g.id));
    root.value = cleanTree(root.value, new Set(keepDocked), new Set()) ?? {
      type: "group",
      id: keepDocked[0],
    };
    for (const g of groups.value) {
      if (!g.activeId || !tabs.value.some((t) => t.id === g.activeId && t.groupId === g.id)) {
        g.activeId = tabsIn(g.id)[0]?.id ?? null;
      }
    }
    const nowDocked = groupIdsIn(root.value);
    if (!nowDocked.includes(lastDocked.value)) lastDocked.value = nowDocked[0];
    if (!group(focusedId.value)) focusedId.value = lastDocked.value;
  }

  /** Make the stored state self-consistent (after load / hydrate). */
  function normalize(): void {
    const ids = new Set(groups.value.map((g) => g.id));
    const treeIds = groupIdsIn(root.value).filter((id) => ids.has(id));
    for (const id of treeIds) {
      const g = group(id);
      if (g) g.float = null;
    }
    // Groups that are neither floating nor in the tree get docked at the end.
    for (const g of groups.value) {
      if (!g.float && !treeIds.includes(g.id)) {
        root.value =
          root.value.type === "split" && root.value.dir === "row"
            ? {
                ...root.value,
                children: [...root.value.children, { type: "group", id: g.id }],
                sizes: [...root.value.sizes, 1 / (root.value.children.length + 1)],
              }
            : {
                type: "split",
                id: newId("s"),
                dir: "row",
                sizes: [0.5, 0.5],
                children: [root.value, { type: "group", id: g.id }],
              };
        treeIds.push(g.id);
      }
      if (g.float) zTop = Math.max(zTop, g.float.z || 0);
    }
    if (!treeIds.length) {
      const g = addGroup();
      root.value = { type: "group", id: g.id };
      treeIds.push(g.id);
    }
    root.value = cleanTree(root.value, new Set(treeIds), new Set()) ?? { type: "group", id: treeIds[0] };
    const fallback = groupIdsIn(root.value)[0];
    for (const t of tabs.value) if (!group(t.groupId)) t.groupId = fallback;
    if (!group(lastDocked.value) || group(lastDocked.value)?.float) lastDocked.value = fallback;
    if (!group(focusedId.value)) focusedId.value = lastDocked.value;
    prune();
  }

  function load(raw: unknown): void {
    const data = (raw ?? {}) as Partial<SavedV2> & SavedV1;
    const valid = (t: SavedTab | null | undefined): t is SavedTab =>
      Boolean(t && KINDS.includes(t.kind) && t.ref);
    if (data.version === 2 && Array.isArray(data.groups) && data.root) {
      groups.value = data.groups
        .filter((g) => g && g.id)
        .map((g) => ({ id: g.id, activeId: g.activeId ?? null, float: g.float ?? null }));
      root.value = data.root;
      tabs.value = (data.tabs ?? []).filter(valid).map((t) => ({
        id: tabId(t.kind, t.ref),
        kind: t.kind,
        ref: t.ref,
        pinned: t.pinned !== false,
        sticky: Boolean(t.sticky),
        groupId: t.groupId ?? "",
      }));
      focusedId.value = data.focused ?? "";
    } else {
      const g: Group = { id: newId("g"), activeId: data.active ?? null, float: null };
      groups.value = [g];
      root.value = { type: "group", id: g.id };
      tabs.value = (data.tabs ?? []).filter(valid).map((t) => ({
        id: tabId(t.kind, t.ref),
        kind: t.kind,
        ref: t.ref,
        pinned: t.pinned !== false,
        sticky: false,
        groupId: g.id,
      }));
      focusedId.value = g.id;
    }
    lastDocked.value = groupIdsIn(root.value)[0] ?? "";
    normalize();
  }

  function snapshot(): SavedV2 {
    return {
      version: 2,
      tabs: tabs.value.map(({ id, kind, ref, pinned, sticky, groupId }) => ({
        id,
        kind,
        ref,
        pinned,
        sticky,
        groupId,
      })),
      groups: groups.value.map((g) => ({ id: g.id, activeId: g.activeId, float: g.float })),
      root: root.value,
      focused: focusedId.value,
    };
  }

  // ── tabs ────────────────────────────────────────────────────────────────

  function activate(id: string): void {
    const t = find(id);
    if (!t) return;
    const g = group(t.groupId);
    if (!g) return;
    g.activeId = id;
    focusGroup(g.id);
  }

  function defaultFloatRect(at?: { x: number; y: number }): FloatRect {
    const n = floats.value.length;
    return {
      x: Math.max(8, (at?.x ?? 120 + n * 28) - 40),
      y: Math.max(8, (at?.y ?? 60 + n * 28) - 16),
      w: FLOAT_W,
      h: FLOAT_H,
      z: ++zTop,
    };
  }

  /** Where a new tab goes for *where*; creates the group if needed. */
  function targetGroup(where: Where, at?: { x: number; y: number }): Group {
    if (where === "float") return addGroup(defaultFloatRect(at));
    if (where === "split") {
      const g = addGroup();
      const beside = focused.value && !focused.value.float ? focused.value.id : lastDocked.value;
      insertBeside(beside, g.id, "right");
      return g;
    }
    return focused.value ?? group(lastDocked.value) ?? groups.value[0];
  }

  /** Open (or focus) a tab. Unpinned opens reuse the group's preview tab. */
  function open(kind: TabKind, tabRef: string, opts: OpenOpts = {}): Tab {
    const id = tabId(kind, tabRef);
    const where = opts.where ?? "tab";
    let tab = find(id);
    if (tab) {
      if (opts.pin || where !== "tab") tab.pinned = true;
      if (opts.edit) tab.edit = true;
      if (where !== "tab") {
        const g = targetGroup(where, opts.at);
        moveTab(id, g.id);
      } else {
        activate(id);
      }
      return tab;
    }
    const g = targetGroup(where, opts.at);
    tab = {
      id,
      kind,
      ref: tabRef,
      pinned: Boolean(opts.pin) || where !== "tab",
      sticky: false,
      groupId: g.id,
      edit: Boolean(opts.edit),
    };
    const preview = tabs.value.findIndex(
      (t) => t.groupId === g.id && !t.pinned && !t.sticky && !status.value[t.id]?.dirty,
    );
    if (!tab.pinned && preview >= 0) {
      tabs.value.splice(preview, 1, tab);
    } else {
      const at = tabs.value.findIndex((t) => t.id === g.activeId);
      tabs.value.splice(at >= 0 ? at + 1 : tabs.value.length, 0, tab);
    }
    trim();
    g.activeId = id;
    focusGroup(g.id);
    prune();
    return find(id) ?? tab;
  }

  /** Drop the oldest clean, non-visible tabs beyond MAX_TABS. */
  function trim(): void {
    while (tabs.value.length > MAX_TABS) {
      const idx = tabs.value.findIndex(
        (t) => !visibleIds.value.has(t.id) && !t.sticky && !status.value[t.id]?.dirty,
      );
      if (idx < 0) return;
      tabs.value.splice(idx, 1);
    }
  }

  function pin(id: string): void {
    const tab = find(id);
    if (tab) tab.pinned = true;
  }

  function pinRef(kind: TabKind, tabRef: string): void {
    pin(tabId(kind, tabRef));
  }

  function setSticky(id: string, sticky: boolean): void {
    const tab = find(id);
    if (!tab) return;
    tab.sticky = sticky;
    if (sticky) tab.pinned = true;
  }

  /** Move a tab into *groupId*, before *beforeId* (or at the end). */
  function moveTab(id: string, groupId: string, beforeId?: string): void {
    const tab = find(id);
    const dest = group(groupId);
    if (!tab || !dest) return;
    const from = group(tab.groupId);
    const idx = tabs.value.indexOf(tab);
    tabs.value.splice(idx, 1);
    tab.groupId = groupId;
    let at = beforeId ? tabs.value.findIndex((t) => t.id === beforeId) : -1;
    if (at < 0) {
      const last = tabs.value.map((t) => t.groupId).lastIndexOf(groupId);
      at = last >= 0 ? last + 1 : tabs.value.length;
    }
    tabs.value.splice(at, 0, tab);
    if (from && from !== dest && from.activeId === id) {
      const rest = tabsIn(from.id);
      from.activeId = rest[0]?.id ?? null;
    }
    dest.activeId = id;
    focusGroup(groupId);
    prune();
  }

  /** Split *targetGroupId* on *side* and move the tab into the new group. */
  function splitTo(id: string, targetGroupId: string, side: Side): void {
    const target = group(targetGroupId);
    if (!target) return;
    if (target.float) {
      // Splitting inside a float: just dock beside the last docked group.
      targetGroupId = lastDocked.value;
    }
    const g = addGroup();
    insertBeside(targetGroupId, g.id, side);
    const tab = find(id);
    if (tab) tab.pinned = true;
    moveTab(id, g.id);
  }

  function floatTab(id: string, at?: { x: number; y: number }): void {
    const tab = find(id);
    if (!tab) return;
    tab.pinned = true;
    const g = addGroup(defaultFloatRect(at));
    moveTab(id, g.id);
  }

  /** Move a floating group's tabs back into the last docked group. */
  function dockFloat(groupId: string): void {
    const g = group(groupId);
    if (!g?.float) return;
    const dest = group(lastDocked.value) ?? group(docked.value[0]);
    if (!dest) return;
    const keepActive = g.activeId;
    for (const t of tabsIn(groupId)) moveTab(t.id, dest.id);
    if (keepActive) activate(keepActive);
    prune();
  }

  /** Remove a tab; returns the focused group's active tab (or null). */
  function close(id: string): Tab | null {
    const idx = tabs.value.findIndex((t) => t.id === id);
    if (idx < 0) return active.value;
    const tab = tabs.value[idx];
    const g = group(tab.groupId);
    const order = tabsIn(tab.groupId);
    const pos = order.findIndex((t) => t.id === id);
    tabs.value.splice(idx, 1);
    const { [id]: _drop, ...rest } = status.value;
    status.value = rest;
    if (g && g.activeId === id) {
      const left = order.filter((t) => t.id !== id);
      g.activeId = (left[pos] ?? left[pos - 1] ?? null)?.id ?? null;
    }
    prune();
    return active.value;
  }

  function closeMany(ids: string[]): Tab | null {
    for (const id of ids) close(id);
    return active.value;
  }

  /** Close every clean tab of a group (dirty ones stay). */
  function closeGroup(groupId: string): Tab | null {
    return closeMany(tabsIn(groupId).filter((t) => !status.value[t.id]?.dirty).map((t) => t.id));
  }

  /** Drag-reorder: drop *fromId* before *toId* (any group). */
  function move(fromId: string, toId: string): void {
    const to = find(toId);
    if (!to || fromId === toId) return;
    moveTab(fromId, to.groupId, toId);
  }

  /** A file was moved/renamed: keep its tab, point it at the new key. */
  function retarget(kind: TabKind, oldRef: string, newRef: string): void {
    const tab = find(tabId(kind, oldRef));
    if (!tab) return;
    const g = group(tab.groupId);
    const wasActive = g?.activeId === tab.id;
    const existing = find(tabId(kind, newRef));
    if (existing) {
      close(tab.id);
      if (wasActive) activate(existing.id);
      return;
    }
    const oldId = tab.id;
    tab.ref = newRef;
    tab.id = tabId(kind, newRef);
    if (status.value[oldId]) {
      const { [oldId]: st, ...rest } = status.value;
      status.value = { ...rest, [tab.id]: st };
    }
    if (g && wasActive) g.activeId = tab.id;
  }

  function setStatus(id: string, patch: TabStatus): void {
    status.value = { ...status.value, [id]: { ...status.value[id], ...patch } };
  }

  function clearEdit(id: string): void {
    const tab = find(id);
    if (tab) tab.edit = false;
  }

  // ── persistence: localStorage now, the workspace API debounced ──────────

  load(loadJson<unknown>("tabs", null));

  watch(
    [tabs, groups, root, focusedId],
    () => {
      const data = snapshot();
      saveJson("tabs", data);
      if (!hydrated) return;
      clearTimeout(serverTimer);
      serverTimer = setTimeout(() => {
        api()
          .putWorkspace(data as unknown as Record<string, unknown>)
          .catch(() => {
            /* offline: localStorage still has it */
          });
      }, 1500);
    },
    { deep: true },
  );

  /** Adopt the server-side layout when this browser has none. */
  async function hydrate(): Promise<void> {
    try {
      if (!tabs.value.length) {
        const ws = await api().getWorkspace();
        if (ws.layout && !tabs.value.length) load(ws.layout);
      }
    } catch {
      /* no workspace yet */
    } finally {
      hydrated = true;
    }
  }

  return {
    tabs,
    groups,
    root,
    focusedId,
    focused,
    activeId,
    active,
    docked,
    floats,
    status,
    find,
    group,
    tabsIn,
    focusGroup,
    raise,
    setFloat,
    setSizes,
    activate,
    open,
    pin,
    pinRef,
    setSticky,
    moveTab,
    splitTo,
    floatTab,
    dockFloat,
    close,
    closeMany,
    closeGroup,
    move,
    retarget,
    setStatus,
    clearEdit,
    hydrate,
  };
});
