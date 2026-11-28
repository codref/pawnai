<script setup lang="ts">
// The brainmap engine: a dark-cork corkboard of cards joined by red string.
//
// World: one transformed div (pan: drag bare cork; zoom: wheel) holding the
// string layer and the cards. A small force layout places cards; anything the
// user drags stays pinned where they put it (saved per board on the server).
//
// Human in the loop — every gesture is a verdict Pawn learns from:
//   drag a card's push-pin onto another card → tie a string (role picker)
//   click a string → inspect / cut it (scissors mode: X)
//   ghost strings → ✓ tie or ✂ snip; the queue walks them with Y / N / →
//   shift-drag on cork → lasso cards into a new or existing thread
//   card verbs → accept / burn proposed cards, triage items
//   shift-click → add a card to the selection the board chat asks about
//
// The animation loop only runs while the board is the visible tab and the
// window is shown; a hidden board costs nothing.
import { computed, markRaw, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { Dialog, Notify } from "quasar";
import { api, type BrainEdge, type BrainKind, type BrainNode, type BrainPos, type BrainSuggestion, type ItemAction } from "@/api";
import { useBrainmapStore } from "@/stores/brainmap";
import { useBoardsStore } from "@/stores/boards";
import { openBrainmap } from "@/router";
import { boardDef, kindMeta, KIND_FILTERS } from "@/lib/brainmap/boards";
import { neighbourhood, searchGraph, isDone } from "@/lib/brainmap/graph";
import type { TieRole } from "@/lib/brainmap/rules";
import {
  bounds,
  createSim,
  hashSeed,
  reheat,
  setSprings,
  settle,
  settled,
  syncBodies,
  tick,
  type BodyInit,
  type Sim,
} from "@/lib/brainmap/physics";
import { pluck, pointInPolygon, stepWobble, type Pt, type Wobble } from "@/lib/brainmap/string";
import { confirmAction } from "@/lib/dialogs";
import { notifyError } from "@/lib/errors";
import BrainCard from "./BrainCard.vue";
import StringLayer from "./StringLayer.vue";
import DossierDrawer from "./DossierDrawer.vue";
import InterrogationQueue from "./InterrogationQueue.vue";

const props = withDefaults(defineProps<{ board: string; focus?: string; active?: boolean }>(), {
  focus: "",
  active: true,
});
const emit = defineEmits<{ title: [title: string] }>();

const store = useBrainmapStore(props.board);
const boardsStore = useBoardsStore();
const def = computed(() => boardDef(props.board, store.meta));
const uid = `bm${hashSeed(props.board + Math.random()).toString(36)}`;
const reducedMotion =
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;

// ── viewport ──────────────────────────────────────────────────────────────
const root = ref<HTMLElement | null>(null);
const hud = ref<HTMLElement | null>(null);
const view = reactive({ x: 0, y: 0, zoom: 0.7 });
const size = reactive({ w: 800, h: 600 });
const frame = ref(0);

function toWorld(clientX: number, clientY: number): Pt {
  const rect = root.value?.getBoundingClientRect();
  const left = rect?.left ?? 0;
  const top = rect?.top ?? 0;
  return { x: (clientX - left - view.x) / view.zoom, y: (clientY - top - view.y) / view.zoom };
}

function toScreen(p: Pt): Pt {
  return { x: p.x * view.zoom + view.x, y: p.y * view.zoom + view.y };
}

// ── simulation ────────────────────────────────────────────────────────────
let sim: Sim | null = null;
const wobbles = new Map<string, Wobble>();
const wobble = reactive<Record<string, number>>({});
let flight: { from: typeof view; to: typeof view; t: number } | null = null;
let raf = 0;
let layoutDirty = false;
let fitted = false;

function cardBox(n: BrainNode): { w: number; h: number } {
  const m = kindMeta(n.kind);
  return { w: m.w, h: m.h };
}

function bodyInits(): BodyInit[] {
  const saved = store.positions;
  return store.visible.nodes.map((n) => {
    const p: BrainPos | undefined = saved[n.id];
    return { id: n.id, kind: n.kind, ...cardBox(n), x: p?.x, y: p?.y, pinned: p?.pinned };
  });
}

/** New cards without a saved spot appear next to something they are tied to. */
function placeNearNeighbours(inits: BodyInit[]): void {
  if (!sim) return;
  const edges = [...store.visible.edges, ...store.visible.suggestions];
  for (const init of inits) {
    if (init.x !== undefined || sim.byId.has(init.id)) continue;
    const e = edges.find((x) => (x.from === init.id && sim!.byId.has(x.to)) || (x.to === init.id && sim!.byId.has(x.from)));
    if (!e) continue;
    const other = sim.byId.get(e.from === init.id ? e.to : e.from)!;
    const angle = ((hashSeed(init.id) % 360) * Math.PI) / 180;
    init.x = other.x + Math.cos(angle) * 260;
    init.y = other.y + Math.sin(angle) * 200;
  }
}

function rebuild(): void {
  const inits = bodyInits();
  const springs = store.visible.edges.map((e) => ({ from: e.from, to: e.to, role: e.role }));
  if (!sim) {
    sim = markRaw(createSim(inits, springs, {}, hashSeed(props.board)));
    const fresh = inits.some((i) => i.x === undefined);
    if (fresh) {
      settle(sim, reducedMotion ? 900 : 320);
      layoutDirty = true;
    } else {
      sim.alpha = 0.08;
    }
  } else {
    const known = new Set(sim.bodies.map((b) => b.id));
    placeNearNeighbours(inits);
    syncBodies(sim, inits, hashSeed(props.board) + frame.value);
    setSprings(sim, springs);
    const added = inits.some((i) => !known.has(i.id));
    if (added) {
      reheat(sim, 0.35);
      layoutDirty = true;
    } else reheat(sim, 0.05);
    if (reducedMotion) settle(sim, 600);
  }
  kick();
}

const pageHidden = ref(typeof document !== "undefined" && document.hidden);
const running = () => props.active && !pageHidden.value;

function kick(): void {
  if (!raf && running()) raf = requestAnimationFrame(loop);
}

function onVisibility(): void {
  pageHidden.value = document.hidden;
  if (!running() && raf) {
    cancelAnimationFrame(raf);
    raf = 0;
  } else kick();
}

function loop(): void {
  raf = 0;
  let alive = false;
  if (sim && !settled(sim) && !reducedMotion) {
    tick(sim);
    tick(sim);
    alive = true;
  }
  for (const [id, w] of wobbles) {
    if (stepWobble(w)) {
      wobble[id] = w.amp;
      alive = true;
    } else {
      wobbles.delete(id);
      delete wobble[id];
    }
  }
  if (flight) {
    flight.t = Math.min(1, flight.t + 0.06);
    const k = 1 - Math.pow(1 - flight.t, 3);
    view.x = flight.from.x + (flight.to.x - flight.from.x) * k;
    view.y = flight.from.y + (flight.to.y - flight.from.y) * k;
    view.zoom = flight.from.zoom + (flight.to.zoom - flight.from.zoom) * k;
    if (flight.t >= 1) flight = null;
    alive = true;
  }
  frame.value++;
  if (!running()) return; // resumes from kick() when the board is shown again
  if (alive || drag.value) kick();
  else onSettled();
}

function onSettled(): void {
  if (!fitted && sim?.bodies.length) {
    fitted = true;
    if (!store.viewport) fit(false);
    focusFromProp();
  }
  if (layoutDirty && sim) {
    layoutDirty = false;
    persistLayout();
  }
}

function persistLayout(): void {
  if (!sim) return;
  const out: Record<string, BrainPos> = { ...store.positions };
  for (const b of sim.bodies) out[b.id] = { x: Math.round(b.x), y: Math.round(b.y), pinned: b.pinned || undefined };
  store.saveLayout(out, { x: Math.round(view.x), y: Math.round(view.y), zoom: Math.round(view.zoom * 1000) / 1000 });
}

function pluckEdgesOf(id: string, strength = 14): void {
  if (reducedMotion) return;
  for (const e of store.visible.edges) {
    if (e.from === id || e.to === id) wobbles.set(e.id, pluck(strength * (hashSeed(e.id) % 2 ? 1 : -1)));
  }
}

/** Pin positions (top centre of each card), recomputed per frame. */
const pins = computed(() => {
  void frame.value;
  const out = new Map<string, Pt>();
  if (!sim) return out;
  for (const b of sim.bodies) out.set(b.id, { x: b.x, y: b.y - b.h / 2 + 10 });
  return out;
});

function slotStyle(id: string): Record<string, string> {
  void frame.value;
  const b = sim?.byId.get(id);
  if (!b) return { display: "none" };
  return { transform: `translate(${b.x - b.w / 2}px, ${b.y - b.h / 2}px)` };
}

function isPinned(id: string): boolean {
  void frame.value;
  return !!sim?.byId.get(id)?.pinned;
}

// ── camera ────────────────────────────────────────────────────────────────
function flyTo(target: { x: number; y: number; zoom: number }, animate = true): void {
  if (!animate || reducedMotion) {
    Object.assign(view, target);
    frame.value++;
    return;
  }
  flight = { from: { ...view }, to: target, t: 0 };
  kick();
}

function frameIds(ids: string[], maxZoom = 1.1, animate = true): void {
  if (!sim) return;
  const bodies = ids.map((id) => sim!.byId.get(id)).filter((b): b is NonNullable<typeof b> => !!b);
  const bb = bounds(bodies);
  if (!bb) return;
  const pad = 80;
  const right = selected.value ? 380 : 0;
  // Keep clear of the HUD on top and the queue strip at the bottom.
  const top = (hud.value?.offsetHeight ?? 50) + 16;
  const bottom = 64;
  const availW = size.w - right - pad;
  const availH = size.h - top - bottom - pad / 2;
  const zoom = Math.max(
    0.2,
    Math.min(maxZoom, availW / (bb.maxX - bb.minX + 1), availH / (bb.maxY - bb.minY + 1)),
  );
  const cx = (bb.minX + bb.maxX) / 2;
  const cy = (bb.minY + bb.maxY) / 2;
  flyTo({ x: (size.w - right) / 2 - cx * zoom, y: top + (size.h - top - bottom) / 2 - cy * zoom, zoom }, animate);
}

function fit(animate = true): void {
  if (!sim) return;
  frameIds(
    sim.bodies.map((b) => b.id),
    0.9,
    animate,
  );
}

function onWheel(ev: WheelEvent): void {
  ev.preventDefault();
  flight = null;
  const rect = root.value!.getBoundingClientRect();
  const sx = ev.clientX - rect.left;
  const sy = ev.clientY - rect.top;
  const next = Math.max(0.15, Math.min(2.5, view.zoom * Math.exp(-ev.deltaY * 0.0015)));
  view.x = sx - ((sx - view.x) * next) / view.zoom;
  view.y = sy - ((sy - view.y) * next) / view.zoom;
  view.zoom = next;
  frame.value++;
  scheduleViewSave();
}

let viewTimer: ReturnType<typeof setTimeout> | undefined;
function scheduleViewSave(): void {
  clearTimeout(viewTimer);
  viewTimer = setTimeout(() => persistLayout(), 1200);
}

// ── selection, focus, search ──────────────────────────────────────────────
const selectedId = ref<string | null>(null);
const selected = computed(() => (selectedId.value ? store.nodeById.get(selectedId.value) ?? null : null));
const focusMode = ref(false);
const hops = ref(1);
const search = ref("");
const scissors = ref(false);
const searchEl = ref<{ focus: () => void } | null>(null);

const matches = computed(() => searchGraph(store.visible, search.value));
const dimSet = computed<Set<string> | null>(() => {
  if (focusMode.value && selectedId.value) return neighbourhood(store.visible, selectedId.value, hops.value);
  if (search.value.trim()) return matches.value;
  if (queueOpen.value && current.value) return new Set(current.value.focus);
  return null;
});
const proposedIds = computed(() => new Set(store.visible.nodes.filter((n) => n.proposed).map((n) => n.id)));

function select(id: string | null): void {
  selectedId.value = id;
  selectedEdge.value = null;
  store.setSelection(id ? [id] : []);
}

/** Shift-click: grow or shrink the selection the board chat asks about. */
function toggleSelected(id: string): void {
  store.toggleSelect(id);
  selectedId.value = store.selection.has(id) ? id : null;
  selectedEdge.value = null;
}

function linksOf(id: string): number {
  return store.graph.edges.filter((e) => e.from === id || e.to === id).length;
}

function focusFromProp(): void {
  const id = props.focus;
  if (!id || !store.nodeById.has(id)) return;
  select(id);
  focusMode.value = true;
  frameIds([...neighbourhood(store.visible, id, 1)], 1);
}

// ── pointer gestures ──────────────────────────────────────────────────────
type Drag =
  | { type: "pan"; sx: number; sy: number; vx: number; vy: number; moved: boolean }
  | { type: "card"; id: string; sx: number; sy: number; ox: number; oy: number; moved: boolean }
  | { type: "tie"; from: string; to: Pt; target: string | null; ok: boolean | null; reason: string }
  | { type: "lasso"; poly: Pt[] };
const drag = ref<Drag | null>(null);
const badTarget = ref<string | null>(null);

function onPointerDown(ev: PointerEvent): void {
  if (ev.button !== 0 && ev.button !== 1) return;
  const el = ev.target as HTMLElement;
  if (el.closest(".q-btn, button, input, .bm-hud, .bm-dossier, .bm-queue, .bm-chip, .bm-rolemenu")) return;
  // Strings and ghost buttons take clicks; capturing the pointer would eat them.
  if (el instanceof SVGElement && el.closest(".hit, .btn")) return;
  root.value?.focus({ preventScroll: true });
  roleMenu.value = null;
  const card = el.closest<HTMLElement>("[data-slot]");
  flight = null;
  if (card && ev.button === 0) {
    const id = card.dataset.slot!;
    if (el.closest(".pin")) {
      drag.value = { type: "tie", from: id, to: toWorld(ev.clientX, ev.clientY), target: null, ok: null, reason: "" };
    } else {
      const b = sim?.byId.get(id);
      if (!b) return;
      drag.value = { type: "card", id, sx: ev.clientX, sy: ev.clientY, ox: b.x, oy: b.y, moved: false };
    }
  } else if (ev.shiftKey && ev.button === 0) {
    drag.value = { type: "lasso", poly: [toWorld(ev.clientX, ev.clientY)] };
  } else {
    drag.value = { type: "pan", sx: ev.clientX, sy: ev.clientY, vx: view.x, vy: view.y, moved: false };
  }
  try {
    (ev.currentTarget as HTMLElement).setPointerCapture(ev.pointerId);
  } catch {
    /* synthetic or already-released pointer: moves still reach the board */
  }
  kick();
}

function onPointerMove(ev: PointerEvent): void {
  const d = drag.value;
  if (!d) return;
  if (d.type === "pan") {
    const dx = ev.clientX - d.sx;
    const dy = ev.clientY - d.sy;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    view.x = d.vx + dx;
    view.y = d.vy + dy;
  } else if (d.type === "card") {
    const b = sim?.byId.get(d.id);
    if (!b || !sim) return;
    const dx = (ev.clientX - d.sx) / view.zoom;
    const dy = (ev.clientY - d.sy) / view.zoom;
    if (!d.moved && Math.abs(dx) + Math.abs(dy) > 4 / view.zoom) {
      d.moved = true;
      pluckEdgesOf(d.id, 10);
    }
    if (d.moved) {
      b.x = d.ox + dx;
      b.y = d.oy + dy;
      b.pinned = true;
      reheat(sim, 0.12);
    }
  } else if (d.type === "tie") {
    d.to = toWorld(ev.clientX, ev.clientY);
    const hit = document.elementFromPoint(ev.clientX, ev.clientY)?.closest<HTMLElement>("[data-slot]");
    const target = hit?.dataset.slot ?? null;
    if (target !== d.target) {
      d.target = target && target !== d.from ? target : null;
      const a = store.nodeById.get(d.from);
      const b = d.target ? store.nodeById.get(d.target) : null;
      if (a && b) {
        const v = def.value.tieRule(store.graph, a, b);
        d.ok = v.ok;
        d.reason = v.ok ? "" : v.reason;
      } else {
        d.ok = null;
        d.reason = "";
      }
    }
  } else if (d.type === "lasso") {
    const p = toWorld(ev.clientX, ev.clientY);
    const last = d.poly[d.poly.length - 1];
    if (Math.hypot(p.x - last.x, p.y - last.y) > 8 / view.zoom) d.poly = [...d.poly, p];
  }
  frame.value++;
}

function onPointerUp(ev: PointerEvent): void {
  const d = drag.value;
  drag.value = null;
  if (!d) return;
  if (d.type === "pan") {
    if (!d.moved) {
      select(null);
      selectedEdge.value = null;
    } else scheduleViewSave();
  } else if (d.type === "card") {
    if (!d.moved) {
      if (ev.shiftKey) toggleSelected(d.id);
      else select(d.id);
    } else {
      layoutDirty = true;
      pluckEdgesOf(d.id, 18);
    }
  } else if (d.type === "tie") {
    void finishTie(d, ev);
  } else if (d.type === "lasso") {
    finishLasso(d.poly);
  }
  kick();
}

function onDoubleClick(ev: MouseEvent): void {
  const card = (ev.target as HTMLElement).closest<HTMLElement>("[data-slot]");
  const b = card ? sim?.byId.get(card.dataset.slot!) : null;
  if (!b || !sim) return;
  b.pinned = false;
  reheat(sim, 0.3);
  layoutDirty = true;
  Notify.create({ message: "Unpinned: the card drifts with its strings again.", position: "bottom-right", timeout: 1600 });
  kick();
}

// ── tying ─────────────────────────────────────────────────────────────────
const roleMenu = ref<{ x: number; y: number; from: string; to: string; roles: TieRole[] } | null>(null);

async function finishTie(d: Extract<Drag, { type: "tie" }>, ev: PointerEvent): Promise<void> {
  if (!d.target) return;
  const a = store.nodeById.get(d.from);
  const b = store.nodeById.get(d.target);
  if (!a || !b) return;
  const v = def.value.tieRule(store.graph, a, b);
  if (!v.ok) {
    badTarget.value = b.id;
    setTimeout(() => (badTarget.value = null), 400);
    Notify.create({ message: v.reason, position: "bottom-right", timeout: 2600, icon: "link_off" });
    return;
  }
  if (v.mode === "lasso") {
    openLasso([a.id, b.id]);
    return;
  }
  if (v.mode === "parent") {
    await tieAndTwang(v.from, v.to, "parent");
    return;
  }
  if (v.mode === "note") {
    await tieAndTwang(v.from, v.to, "mentions");
    return;
  }
  const rect = root.value!.getBoundingClientRect();
  roleMenu.value = { x: ev.clientX - rect.left, y: ev.clientY - rect.top, from: v.from, to: v.to, roles: v.roles };
}

async function tieAndTwang(from: string, to: string, role: string): Promise<void> {
  roleMenu.value = null;
  const edge = await store.tie(from, to, role);
  if (edge && !reducedMotion) {
    wobbles.set(edge.id, pluck(42));
    kick();
  }
}

// ── strings: inspect, cut, verdicts ───────────────────────────────────────
const selectedEdge = ref<{ edge: BrainEdge; at: Pt } | null>(null);
const hoverGhost = ref<{ s: BrainSuggestion; at: Pt } | null>(null);

async function onEdge(edge: BrainEdge, at: Pt): Promise<void> {
  if (scissors.value && edge.cuttable) {
    await store.cut(edge);
    return;
  }
  selectedEdge.value = { edge, at };
}

async function cutSelected(): Promise<void> {
  const e = selectedEdge.value?.edge;
  selectedEdge.value = null;
  if (e) await store.cut(e);
}

function edgeLabel(e: BrainEdge): string {
  const role = String(e.role).replace(/_/g, " ");
  const by = e.by === "user" ? "you" : e.by === "goal" ? "Goals.md" : e.by === "derived" ? "the transcript" : "Pawn";
  return `${role} · tied by ${by}${typeof e.score === "number" ? ` · d=${e.score.toFixed(2)}` : ""}`;
}

function nodeTitle(id: string): string {
  return store.nodeById.get(id)?.title ?? id;
}

// ── lasso ─────────────────────────────────────────────────────────────────
const lassoDialog = ref<{ ids: string[]; title: string; into: string | null; board: string } | null>(null);
const threadOptions = computed(() =>
  store.graph.nodes
    .filter((n) => n.kind === "thread" && !isDone(n) && n.meta.subject_id)
    .map((n) => ({ label: n.title, value: n.id })),
);

function finishLasso(poly: Pt[]): void {
  if (poly.length < 3 || !sim) return;
  const ids = sim.bodies
    .filter((b) => b.kind !== "person" && b.kind !== "session" && pointInPolygon({ x: b.x, y: b.y }, poly))
    .map((b) => b.id);
  if (!ids.length) return;
  openLasso(ids);
}

function openLasso(ids: string[]): void {
  lassoDialog.value = { ids, title: "", into: null, board: "" };
}

function selectLassoed(): void {
  const d = lassoDialog.value;
  lassoDialog.value = null;
  if (!d) return;
  store.setSelection(d.ids);
  selectedId.value = null;
  Notify.create({ message: `${d.ids.length} card(s) selected for the board chat.`, position: "bottom-right", timeout: 1600 });
}

async function spinOff(ids: string[], title: string): Promise<void> {
  const name = title.trim();
  if (!name || !ids.length) return;
  const nodes = ids.filter((i) => !i.startsWith("prop:") && !i.startsWith("note:"));
  const b = await boardsStore.create({ title: name, kind: "manual", nodes });
  if (b) openBrainmap(b.name);
}

async function spinOffLasso(): Promise<void> {
  const d = lassoDialog.value;
  if (!d || !d.board.trim()) return;
  lassoDialog.value = null;
  await spinOff(d.ids, d.board);
}

async function submitLasso(): Promise<void> {
  const d = lassoDialog.value;
  if (!d) return;
  if (!d.into && !d.title.trim()) return;
  lassoDialog.value = null;
  const subject = await store.lasso(
    d.ids,
    d.into ? undefined : d.title.trim(),
    d.into ? store.nodeById.get(d.into)?.meta.subject_id : undefined,
  );
  if (subject) {
    await nextTick();
    select(subject.id);
  }
}

// ── card verbs ────────────────────────────────────────────────────────────
const ITEM_VERBS = new Set(["todo", "file", "task", "later", "ignore", "delete", "approve", "reject"]);

async function onVerb(node: BrainNode, verb: string): Promise<void> {
  if (node.meta.proposal_id && (verb === "accept" || verb === "dismiss")) {
    await store.decide(node, verb === "accept" ? "accept" : "reject");
    if (selectedId.value === node.id) select(null);
    return;
  }
  if (verb === "remove") {
    await store.removeCards([node.id]);
    if (selectedId.value === node.id) select(null);
    return;
  }
  if (verb === "pin_to") return pinTo([node.id]);
  if (verb === "open_board" && node.meta.subject_id) {
    const b = await boardsStore.create({ title: node.title, kind: "lens", spec: { subject_ids: [node.meta.subject_id] } });
    if (b) openBrainmap(b.name);
    return;
  }
  if (verb === "investigate" && node.meta.subject_id) {
    const b = await boardsStore.create({ title: node.title, kind: "question", subject_id: node.meta.subject_id });
    if (b) {
      Notify.create({ message: "The night shift will work on this question.", position: "bottom-right" });
      openBrainmap(b.name);
    }
    return;
  }
  if (ITEM_VERBS.has(verb)) {
    if (verb === "delete" && !(await confirmAction("Delete item", `Delete “${node.title.slice(0, 80)}”?`))) return;
    await store.itemAction(node, verb as ItemAction);
    if (["ignore", "delete", "reject"].includes(verb) && selectedId.value === node.id) select(null);
  } else if (verb === "accept") await store.subjectStatus(node, "active");
  else if (verb === "dismiss") {
    await store.subjectStatus(node, "dismissed");
    if (selectedId.value === node.id) select(null);
  } else if (verb === "done" || verb === "resolved") await store.subjectStatus(node, verb);
  else if (verb === "spill") {
    const added = await store.spill(node);
    if (added) frameIds([...neighbourhood(store.visible, node.id, 1)], 1);
  } else if (verb === "council" && node.meta.subject_id) {
    try {
      await api().emergenceCouncil(node.meta.subject_id);
      Notify.create({ message: "Council convened. The note appears when the personas finish.", position: "bottom-right" });
    } catch (e) {
      notifyError(e, "Could not convene the council");
    }
  }
}

// ── interrogation queue ───────────────────────────────────────────────────
const queueOpen = ref(false);
const queueIndex = ref(0);
const current = computed(() => {
  const q = store.queue;
  if (!q.length) return null;
  return q[Math.min(queueIndex.value, q.length - 1)];
});
const activeSuggestion = computed(() =>
  queueOpen.value && current.value?.type === "suggestion" ? current.value.id : null,
);
const glowIds = computed(() => {
  const s = new Set(matches.value);
  if (queueOpen.value && current.value?.type === "proposed") s.add(current.value.id);
  return s;
});

watch([queueOpen, current], ([open, entry]) => {
  if (open && entry) frameIds(entry.focus, 0.95);
});

async function queueYes(): Promise<void> {
  const e = current.value;
  if (!e) return;
  if (e.type === "suggestion") await store.verdict(e.id, "confirm");
  else if (e.node.meta.proposal_id) await store.decide(e.node, "accept");
  else await store.subjectStatus(e.node, "active");
}

async function queueNo(): Promise<void> {
  const e = current.value;
  if (!e) return;
  if (e.type === "suggestion") await store.verdict(e.id, "snip");
  else if (e.node.meta.proposal_id) await store.decide(e.node, "reject");
  else await store.subjectStatus(e.node, "dismissed");
}

function queueSkip(): void {
  const n = store.queue.length;
  if (n) queueIndex.value = (queueIndex.value + 1) % n;
}

// ── boards: pin elsewhere, new since last look ────────────────────────────
const pinDialog = ref<{ ids: string[]; target: string | null; title: string } | null>(null);
const pinTargets = computed(() =>
  boardsStore.userBoards.filter((b) => b.name !== props.board).map((b) => ({ label: b.title, value: b.name })),
);

function pinTo(ids: string[]): void {
  const clean = ids.filter((i) => !i.startsWith("prop:") && !i.startsWith("note:"));
  if (!clean.length) return;
  if (!boardsStore.loaded) void boardsStore.refresh();
  pinDialog.value = { ids: clean, target: null, title: "" };
}

async function submitPin(): Promise<void> {
  const d = pinDialog.value;
  if (!d) return;
  pinDialog.value = null;
  if (d.target) {
    if (await boardsStore.pin(d.target, d.ids)) {
      const t = boardsStore.byName.get(d.target)?.title ?? "the board";
      Notify.create({ message: `Pinned ${d.ids.length} card(s) to “${t}”.`, position: "bottom-right", timeout: 2000 });
    }
  } else if (d.title.trim()) await spinOff(d.ids, d.title);
}

const newCount = computed(() => store.newIds.size);
const newSeenLabel = computed(() => {
  const at = store.meta?.seen_at;
  if (!at) return "";
  const d = new Date(at.endsWith("Z") || at.includes("+") ? at : `${at}Z`);
  return d.toLocaleString(undefined, { weekday: "short", hour: "2-digit", minute: "2-digit" });
});

function showNew(): void {
  const ids = [...store.newIds].filter((i) => store.nodeById.has(i));
  if (!ids.length) return;
  store.setSelection(ids);
  selectedId.value = null;
  frameIds(ids, 1);
}

const selectionCount = computed(() => store.selection.size);

// ── keyboard ──────────────────────────────────────────────────────────────
function onKey(ev: KeyboardEvent): void {
  const t = ev.target as HTMLElement;
  if (t.closest("input, textarea, [contenteditable=true]")) return;
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const k = ev.key;
  if (queueOpen.value && (k === "y" || k === "Y")) void queueYes();
  else if (queueOpen.value && (k === "n" || k === "N")) void queueNo();
  else if (queueOpen.value && k === "ArrowRight") queueSkip();
  else if (k === "q") queueOpen.value = !queueOpen.value;
  else if (k === "x") scissors.value = !scissors.value;
  else if (k === "f") fit();
  else if (k === "g" && selectedId.value) focusMode.value = !focusMode.value;
  else if (k === "/") searchEl.value?.focus();
  else if (k === "Escape") {
    if (queueOpen.value) queueOpen.value = false;
    else if (selectedEdge.value || roleMenu.value) {
      selectedEdge.value = null;
      roleMenu.value = null;
    } else if (scissors.value) scissors.value = false;
    else select(null);
  } else if (k === "p" && store.selection.size) {
    pinTo([...store.selection]);
  } else return;
  ev.preventDefault();
}

// ── HUD helpers ───────────────────────────────────────────────────────────
function reshuffle(): void {
  if (!sim) return;
  for (const b of sim.bodies) {
    if (b.pinned) continue;
    b.x += (Math.random() - 0.5) * 400;
    b.y += (Math.random() - 0.5) * 300;
  }
  reheat(sim, 1);
  layoutDirty = true;
  kick();
}

async function unpinAll(): Promise<void> {
  if (!sim) return;
  if (!(await confirmAction("Unpin every card", "Let Pawn lay out the whole board again?", "Unpin"))) return;
  for (const b of sim.bodies) b.pinned = false;
  reheat(sim, 1);
  layoutDirty = true;
  kick();
}

function toggleKind(kind: BrainKind): void {
  store.toggleKind(kind);
}

function showHelp(): void {
  Dialog.create({
    title: "Working the board",
    html: true,
    message: [
      "<b>Drag a card</b> to pin it where you want. Double-click to let it drift again.",
      "<b>Drag a push-pin</b> onto another card to tie a red string. You pick what the string means.",
      "<b>Click a string</b> to inspect it, or press <kbd>X</kbd> for scissors and cut wrong ones.",
      "<b>Ghost strings</b> are Pawn's guesses: ✓ ties them, ✂ snips them. <kbd>Q</kbd> opens the queue (<kbd>Y</kbd> / <kbd>N</kbd> / <kbd>→</kbd>).",
      "<b>Shift-drag</b> on bare cork to lasso cards into a new thread, or file them under one.",
      "<b>Burn</b> a proposed card to drop it. <kbd>G</kbd> focuses the selected card's web, <kbd>F</kbd> fits, <kbd>/</kbd> searches.",
      "Every verdict here becomes a training label: Pawn ties better strings next time.",
    ].join("<br><br>"),
    ok: "Got it",
  });
}

const title = computed(() =>
  store.custom
    ? `${def.value.title}${newCount.value ? ` · ${newCount.value} new` : ""}`
    : `${def.value.title}${store.counts.leads ? ` (${store.counts.leads})` : ""}`,
);
watch(title, (t) => emit("title", t), { immediate: true });

// ── lifecycle ─────────────────────────────────────────────────────────────
watch(
  () => store.visible,
  () => {
    if (store.loaded) rebuild();
  },
);

watch(
  () => store.loaded,
  (ok) => {
    if (!ok) return;
    if (store.viewport) Object.assign(view, store.viewport);
    rebuild();
  },
  { immediate: true },
);

watch(
  () => props.focus,
  () => {
    if (fitted) focusFromProp();
  },
);

watch(
  () => props.active,
  (on) => {
    if (on) kick();
    else if (raf) {
      cancelAnimationFrame(raf);
      raf = 0;
    }
  },
);

let ro: ResizeObserver | null = null;
onMounted(() => {
  store.attach(props.focus);
  document.addEventListener("visibilitychange", onVisibility);
  ro = new ResizeObserver(([entry]) => {
    size.w = entry.contentRect.width;
    size.h = entry.contentRect.height;
  });
  if (root.value) ro.observe(root.value);
});

onBeforeUnmount(() => {
  store.detach();
  document.removeEventListener("visibilitychange", onVisibility);
  ro?.disconnect();
  if (raf) cancelAnimationFrame(raf);
  clearTimeout(viewTimer);
  if (layoutDirty) persistLayout();
});

const corkStyle = computed(() => ({
  backgroundPosition: `${view.x * 0.6}px ${view.y * 0.6}px, ${view.x}px ${view.y}px, 0 0`,
}));
const worldStyle = computed(() => ({
  transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`,
}));
const liveString = computed(() => {
  const d = drag.value;
  if (!d || d.type !== "tie") return null;
  const from = pins.value.get(d.from);
  if (!from) return null;
  return { from, to: d.to, ok: d.ok };
});
const lassoPoly = computed(() => (drag.value?.type === "lasso" ? drag.value.poly : null));
const tieState = computed(() => {
  const d = drag.value;
  return d?.type === "tie" && d.target ? { id: d.target, ok: d.ok } : null;
});
const chipStyle = (p: Pt) => ({ left: `${toScreen(p).x}px`, top: `${toScreen(p).y}px` });
</script>

<template>
  <div
    ref="root"
    class="brainmap"
    :class="{ scissors, panning: drag?.type === 'pan', tying: drag?.type === 'tie' }"
    tabindex="0"
    :style="corkStyle"
    @pointerdown="onPointerDown"
    @pointermove="onPointerMove"
    @pointerup="onPointerUp"
    @pointercancel="onPointerUp"
    @dblclick="onDoubleClick"
    @wheel="onWheel"
    @keydown="onKey"
  >
    <div class="bm-world" :style="worldStyle">
      <StringLayer
        :uid="uid"
        :edges="store.visible.edges"
        :suggestions="store.visible.suggestions"
        :pins="pins"
        :wobble="wobble"
        :selected-edge="selectedEdge?.edge.id ?? null"
        :active-suggestion="activeSuggestion"
        :dim="dimSet"
        :proposed="proposedIds"
        :scissors="scissors"
        :live="liveString"
        :lasso="lassoPoly"
        :frame="frame"
        @edge="(e, at) => onEdge(e, at)"
        @suggestion="(id, v) => store.verdict(id, v)"
        @hover-suggestion="(s, at) => (hoverGhost = s && at ? { s, at } : null)"
      />
      <div
        v-for="n in store.visible.nodes"
        :key="n.id"
        class="bm-slot"
        :class="{ pinned: isPinned(n.id) }"
        :data-slot="n.id"
        :style="slotStyle(n.id)"
      >
        <BrainCard
          :node="n"
          :selected="selectedId === n.id || store.selection.has(n.id)"
          :dimmed="!!dimSet && !dimSet.has(n.id)"
          :glow="glowIds.has(n.id)"
          :burning="store.burning.has(n.id)"
          :flashing="store.flashing.has(n.id)"
          :busy="store.busy.has(n.id)"
          :tie-target="tieState?.id === n.id ? (tieState.ok ? 'ok' : 'bad') : badTarget === n.id ? 'bad' : null"
          @verb="(v) => onVerb(n, v)"
        />
      </div>
    </div>

    <div class="bm-scan" aria-hidden="true" />

    <!-- HUD -->
    <div ref="hud" class="bm-hud">
      <div class="brand">
        <span class="glitch" :data-text="`BRAINMAP // ${def.title.toUpperCase()}`">BRAINMAP // {{ def.title.toUpperCase() }}</span>
        <span class="sub">{{ def.subtitle }}</span>
      </div>
      <div class="readout">
        <span>LEADS <b>{{ store.counts.leads }}</b></span>
        <span>HUBS <b>{{ store.counts.hubs }}</b></span>
        <span>STRINGS <b>{{ store.counts.strings }}</b></span>
        <span class="hot">GHOSTS <b>{{ store.counts.suggestions }}</b></span>
      </div>
      <q-space />
      <q-input
        ref="searchEl"
        v-model="search"
        dense
        dark
        borderless
        clearable
        class="bm-search"
        placeholder="Search the board  /"
        @keydown.esc="search = ''"
      >
        <template #prepend><q-icon name="search" size="16px" /></template>
      </q-input>
      <q-btn flat dense round size="sm" icon="filter_alt" aria-label="Filter">
        <q-tooltip>Card kinds</q-tooltip>
        <q-menu dark class="bm-menu">
          <q-list dense style="min-width: 190px">
            <q-item v-for="k in KIND_FILTERS" :key="k.kind" clickable @click="toggleKind(k.kind)">
              <q-item-section side>
                <q-checkbox dense dark size="xs" :model-value="!store.hidden.has(k.kind)" @update:model-value="toggleKind(k.kind)" />
              </q-item-section>
              <q-item-section>
                <span :style="{ color: kindMeta(k.kind).pin }">● </span>{{ k.label }}
              </q-item-section>
            </q-item>
            <q-separator dark />
            <q-item clickable @click="store.setShow('suggestions', !store.showSuggestions)">
              <q-item-section side><q-checkbox dense dark size="xs" :model-value="store.showSuggestions" /></q-item-section>
              <q-item-section>Ghost strings</q-item-section>
            </q-item>
            <q-item clickable @click="store.setShow('done', !store.showDone)">
              <q-item-section side><q-checkbox dense dark size="xs" :model-value="store.showDone" /></q-item-section>
              <q-item-section>Closed cards</q-item-section>
            </q-item>
          </q-list>
        </q-menu>
      </q-btn>
      <q-btn
        flat
        dense
        round
        size="sm"
        icon="center_focus_strong"
        :class="{ on: focusMode }"
        :disable="!selectedId"
        aria-label="Focus"
        @click="focusMode = !focusMode"
      >
        <q-tooltip>Focus the selected card's web (G)</q-tooltip>
      </q-btn>
      <q-btn v-if="focusMode" flat dense no-caps size="sm" :label="`${hops} hop${hops > 1 ? 's' : ''}`" @click="hops = hops === 1 ? 2 : 1" />
      <q-btn flat dense round size="sm" icon="content_cut" :class="{ on: scissors }" aria-label="Scissors" @click="scissors = !scissors">
        <q-tooltip>Scissors: click a string to cut it (X)</q-tooltip>
      </q-btn>
      <q-btn flat dense round size="sm" icon="shuffle" aria-label="Reshuffle" @click="reshuffle">
        <q-tooltip>Shake the unpinned cards</q-tooltip>
      </q-btn>
      <q-btn flat dense round size="sm" icon="fit_screen" aria-label="Fit" @click="fit()">
        <q-tooltip>Fit the board (F)</q-tooltip>
      </q-btn>
      <q-btn flat dense round size="sm" icon="more_vert" aria-label="More">
        <q-menu dark class="bm-menu">
          <q-list dense style="min-width: 180px">
            <q-item v-close-popup clickable @click="store.load()">
              <q-item-section avatar><q-icon name="refresh" /></q-item-section>
              <q-item-section>Reload</q-item-section>
            </q-item>
            <q-item v-if="selectionCount" v-close-popup clickable @click="pinTo([...store.selection])">
              <q-item-section avatar><q-icon name="push_pin" /></q-item-section>
              <q-item-section>Pin selection to a board…</q-item-section>
            </q-item>
            <q-item v-if="search.trim() && matches.size" v-close-popup clickable @click="pinDialog = { ids: [...matches], target: null, title: search.trim() }">
              <q-item-section avatar><q-icon name="dashboard_customize" /></q-item-section>
              <q-item-section>Board from search results ({{ matches.size }})</q-item-section>
            </q-item>
            <q-item v-close-popup clickable @click="unpinAll">
              <q-item-section avatar><q-icon name="push_pin" /></q-item-section>
              <q-item-section>Unpin every card</q-item-section>
            </q-item>
            <q-item v-close-popup clickable @click="showHelp">
              <q-item-section avatar><q-icon name="help_outline" /></q-item-section>
              <q-item-section>How the board works</q-item-section>
            </q-item>
          </q-list>
        </q-menu>
      </q-btn>
    </div>

    <div v-if="store.loaded && !store.graph.emergence_enabled" class="bm-banner">
      EMERGENCE OFFLINE — triage works, strings are read-only. Set <code>emergence.enabled</code> to tie and cut.
    </div>
    <div v-if="store.graph.truncated" class="bm-banner low">Showing the newest leads only. Filter to see the rest.</div>
    <div v-if="store.custom && newCount" class="bm-banner new">
      <b>{{ newCount }} NEW</b> since you looked<template v-if="newSeenLabel"> ({{ newSeenLabel }})</template>
      <button type="button" @click="showNew">Show</button>
      <button type="button" @click="store.markSeen()">Mark seen</button>
    </div>
    <div v-if="selectionCount > 1" class="bm-selbar">
      <b>{{ selectionCount }}</b> SELECTED · the board chat asks about these ·
      <button type="button" @click="pinTo([...store.selection])">Pin to a board (P)</button>
      <button v-if="store.custom" type="button" @click="store.removeCards([...store.selection])">Take off</button>
      <button type="button" @click="select(null)">Clear</button>
    </div>

    <div v-if="store.loaded && !store.visible.nodes.length" class="bm-empty">
      <div class="big">{{ store.custom ? "EMPTY BOARD" : "NO LEADS" }}</div>
      <div>{{ def.emptyText }}</div>
    </div>
    <q-inner-loading :showing="store.loading && !store.loaded" dark color="accent" />

    <!-- Selected string -->
    <div v-if="selectedEdge" class="bm-chip" :style="chipStyle(selectedEdge.at)">
      <div class="chip-title">{{ edgeLabel(selectedEdge.edge) }}</div>
      <div class="chip-ends">{{ nodeTitle(selectedEdge.edge.from).slice(0, 50) }} ⟷ {{ nodeTitle(selectedEdge.edge.to).slice(0, 50) }}</div>
      <div class="chip-btns">
        <q-btn v-if="selectedEdge.edge.cuttable" dense no-caps unelevated class="cut" icon="content_cut" label="Cut" @click="cutSelected" />
        <span v-else class="muted">Pawn wires people and tapes from the transcript.</span>
        <q-btn dense flat round size="sm" icon="close" aria-label="Close" @click="selectedEdge = null" />
      </div>
    </div>

    <!-- Ghost string reason -->
    <div v-if="hoverGhost && !drag" class="bm-chip ghost" :style="chipStyle(hoverGhost.at)">
      <div class="chip-title">PAWN SUGGESTS · {{ String(hoverGhost.s.role).replace(/_/g, " ") }}</div>
      <div class="chip-ends">{{ hoverGhost.s.reason }}</div>
    </div>

    <!-- Role picker after a tie -->
    <div v-if="roleMenu" class="bm-rolemenu" :style="{ left: `${roleMenu.x}px`, top: `${roleMenu.y}px` }">
      <div class="chip-title">WHAT DOES THIS STRING MEAN?</div>
      <button v-for="r in roleMenu.roles" :key="r.role" type="button" @click="tieAndTwang(roleMenu!.from, roleMenu!.to, r.role)">
        {{ r.label }}
      </button>
      <button type="button" class="cancel" @click="roleMenu = null">Cancel</button>
    </div>

    <div class="bm-legend">
      <span v-for="l in def.legend" :key="l.className" :class="l.className"><i />{{ l.label }}</span>
      <a href="#" @click.prevent="showHelp">?</a>
    </div>

    <DossierDrawer
      v-if="selected"
      :node="selected"
      :busy="store.busy.has(selected.id)"
      :links="linksOf(selected.id)"
      @close="select(null)"
      @verb="(v) => onVerb(selected!, v)"
      @focus="
        focusMode = true;
        frameIds([...neighbourhood(store.visible, selected!.id, hops)], 1);
      "
    />

    <InterrogationQueue
      :entry="current"
      :index="Math.min(queueIndex, Math.max(store.queue.length - 1, 0))"
      :total="store.queue.length"
      :nodes="store.nodeById"
      :open="queueOpen"
      @toggle="queueOpen = !queueOpen"
      @yes="queueYes"
      @no="queueNo"
      @skip="queueSkip"
    />

    <q-dialog :model-value="!!lassoDialog" @update:model-value="(v) => !v && (lassoDialog = null)">
      <q-card v-if="lassoDialog" dark class="bm-dialog">
        <q-card-section>
          <div class="chip-title">LASSO · {{ lassoDialog.ids.length }} CARD(S)</div>
          <div class="row items-center q-gutter-sm q-mt-xs">
            <q-btn dense no-caps outline icon="select_all" label="Select them for the chat" @click="selectLassoed" />
          </div>
          <div class="row items-center no-wrap q-gutter-sm q-mt-sm">
            <q-input
              v-model="lassoDialog.board"
              dark
              outlined
              dense
              class="col"
              label="…or spin them off into a new board"
              @keydown.enter="spinOffLasso"
            />
            <q-btn dense no-caps unelevated class="neon" label="New board" :disable="!lassoDialog.board.trim()" @click="spinOffLasso" />
          </div>
          <div class="text-h6 q-mt-md">Tie these into a thread</div>
          <div v-if="!store.graph.emergence_enabled" class="text-caption" style="color: #8fa3b8">
            Turn on emergence to group cards into threads.
          </div>
        </q-card-section>
        <q-card-section class="q-pt-none">
          <q-input
            v-model="lassoDialog.title"
            dark
            outlined
            dense
            autofocus
            label="New thread name"
            :disable="!!lassoDialog.into"
            @keydown.enter="submitLasso"
          />
          <div class="text-caption q-my-sm" style="color: #8fa3b8">or file them under an existing thread</div>
          <q-select
            v-model="lassoDialog.into"
            dark
            outlined
            dense
            clearable
            emit-value
            map-options
            :options="threadOptions"
            label="Existing thread"
          />
        </q-card-section>
        <q-card-actions align="right">
          <q-btn flat no-caps label="Cancel" @click="lassoDialog = null" />
          <q-btn
            unelevated
            no-caps
            class="neon"
            :label="lassoDialog.into ? 'File under thread' : 'Make thread'"
            :disable="!store.graph.emergence_enabled || (!lassoDialog.into && !lassoDialog.title.trim())"
            @click="submitLasso"
          />
        </q-card-actions>
      </q-card>
    </q-dialog>

    <q-dialog :model-value="!!pinDialog" @update:model-value="(v) => !v && (pinDialog = null)">
      <q-card v-if="pinDialog" dark class="bm-dialog">
        <q-card-section>
          <div class="chip-title">PIN · {{ pinDialog.ids.length }} CARD(S)</div>
          <div class="text-h6 q-mt-xs">Put these on a board</div>
        </q-card-section>
        <q-card-section class="q-pt-none">
          <q-select
            v-model="pinDialog.target"
            dark
            outlined
            dense
            clearable
            emit-value
            map-options
            :options="pinTargets"
            label="Existing board"
          />
          <div class="text-caption q-my-sm" style="color: #8fa3b8">or start a new one</div>
          <q-input
            v-model="pinDialog.title"
            dark
            outlined
            dense
            label="New board name"
            :disable="!!pinDialog.target"
            @keydown.enter="submitPin"
          />
        </q-card-section>
        <q-card-actions align="right">
          <q-btn flat no-caps label="Cancel" @click="pinDialog = null" />
          <q-btn
            unelevated
            no-caps
            class="neon"
            :label="pinDialog.target ? 'Pin' : 'New board'"
            :disable="!pinDialog.target && !pinDialog.title.trim()"
            @click="submitPin"
          />
        </q-card-actions>
      </q-card>
    </q-dialog>
  </div>
</template>

<style scoped>
.brainmap {
  --bm-neon: #5ce1ff;
  --bm-hot: #ff4d9a;
  --bm-red: #ff2a4d;
  position: relative;
  height: 100%;
  min-height: 360px;
  overflow: hidden;
  outline: none;
  touch-action: none;
  color: #d8e6ee;
  cursor: grab;
  background-color: #0b0907;
  background-image:
    radial-gradient(rgba(255, 190, 120, 0.06) 1px, transparent 1.5px),
    linear-gradient(rgba(92, 225, 255, 0.035) 1px, transparent 1px),
    radial-gradient(ellipse at 50% 40%, #1a120c 0%, #0b0907 55%, #050403 100%);
  background-size:
    7px 7px,
    100% 56px,
    100% 100%;
}
.brainmap.panning {
  cursor: grabbing;
}
.brainmap.tying {
  cursor: crosshair;
}
.brainmap::before {
  /* cork grain + vignette */
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    radial-gradient(circle at 20% 30%, rgba(120, 70, 30, 0.12), transparent 40%),
    radial-gradient(circle at 80% 70%, rgba(120, 70, 30, 0.1), transparent 45%),
    radial-gradient(ellipse at center, transparent 55%, rgba(0, 0, 0, 0.75) 100%);
  z-index: 1;
}
.bm-scan {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 3;
  background: repeating-linear-gradient(180deg, rgba(255, 255, 255, 0.018) 0 1px, transparent 1px 3px);
  mix-blend-mode: overlay;
}
.bm-world {
  position: absolute;
  left: 0;
  top: 0;
  transform-origin: 0 0;
  z-index: 2;
}
.bm-slot {
  position: absolute;
  left: 0;
  top: 0;
  will-change: transform;
}
.bm-slot.pinned :deep(.pin) {
  box-shadow:
    0 0 0 2px rgba(255, 255, 255, 0.85),
    0 0 12px var(--pin);
}

/* HUD */
.bm-hud {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  z-index: 5;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: linear-gradient(180deg, rgba(5, 8, 12, 0.95), rgba(5, 8, 12, 0.7));
  border-bottom: 1px solid rgba(92, 225, 255, 0.25);
  cursor: default;
  flex-wrap: wrap;
}
.brand {
  display: flex;
  flex-direction: column;
  line-height: 1.1;
  margin-right: 10px;
}
.glitch {
  position: relative;
  font: 800 13px var(--pawn-mono);
  letter-spacing: 0.22em;
  color: var(--bm-neon);
  text-shadow: 0 0 10px rgba(92, 225, 255, 0.6);
}
.glitch::before,
.glitch::after {
  content: attr(data-text);
  position: absolute;
  left: 0;
  top: 0;
  opacity: 0.7;
  pointer-events: none;
}
.glitch::before {
  color: var(--bm-hot);
  animation: bm-glitch 5s infinite steps(1);
  clip-path: inset(0 0 55% 0);
}
.glitch::after {
  color: #3dffb0;
  animation: bm-glitch 7s infinite reverse steps(1);
  clip-path: inset(55% 0 0 0);
}
.sub {
  font-size: 10.5px;
  color: #8fa3b8;
}
.readout {
  display: flex;
  gap: 12px;
  font: 10px var(--pawn-mono);
  letter-spacing: 0.14em;
  color: #8fa3b8;
}
.readout b {
  color: #e9fbff;
  font-weight: 700;
}
.readout .hot b {
  color: var(--bm-hot);
}
.bm-search {
  width: 220px;
  padding: 0 8px;
  border: 1px solid rgba(92, 225, 255, 0.3);
  font-size: 12px;
}
.bm-hud :deep(.q-btn) {
  color: #9fd8ea;
}
.bm-hud :deep(.q-btn.on) {
  color: #071017;
  background: var(--bm-neon);
  box-shadow: 0 0 12px var(--bm-neon);
}
.bm-banner {
  position: absolute;
  top: 52px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 4;
  padding: 4px 12px;
  font: 10.5px var(--pawn-mono);
  letter-spacing: 0.1em;
  color: #ffd23f;
  background: rgba(40, 30, 0, 0.85);
  border: 1px solid rgba(255, 210, 63, 0.5);
}
.bm-banner.low {
  top: 80px;
  color: #8fa3b8;
  border-color: rgba(143, 163, 184, 0.4);
  background: rgba(5, 8, 12, 0.85);
}
.bm-banner.new {
  top: 52px;
  display: flex;
  gap: 10px;
  align-items: center;
}
.bm-banner button,
.bm-selbar button {
  font: inherit;
  letter-spacing: inherit;
  color: #5ce1ff;
  background: transparent;
  border: 1px solid rgba(92, 225, 255, 0.4);
  padding: 0 6px;
  cursor: pointer;
}
.bm-selbar {
  position: absolute;
  bottom: 56px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 4;
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 4px 12px;
  font: 10.5px var(--pawn-mono);
  letter-spacing: 0.08em;
  color: #cfe9f5;
  background: rgba(5, 8, 12, 0.9);
  border: 1px solid rgba(92, 225, 255, 0.35);
}
.bm-empty {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  z-index: 4;
  pointer-events: none;
  color: #8fa3b8;
  text-align: center;
  padding: 0 20px;
}
.bm-empty .big {
  font: 900 42px var(--pawn-mono);
  letter-spacing: 0.3em;
  color: var(--bm-red);
  text-shadow: 0 0 24px rgba(255, 42, 77, 0.6);
}

/* chips and menus */
.bm-chip,
.bm-rolemenu {
  position: absolute;
  z-index: 6;
  transform: translate(-50%, 14px);
  max-width: 320px;
  padding: 8px 10px;
  background: rgba(7, 11, 16, 0.96);
  border: 1px solid rgba(255, 42, 77, 0.6);
  box-shadow: 0 0 24px rgba(0, 0, 0, 0.7);
  font-size: 12px;
  cursor: default;
}
.bm-chip.ghost {
  border-color: rgba(92, 225, 255, 0.6);
  pointer-events: none;
}
.chip-title {
  font: 600 10px var(--pawn-mono);
  letter-spacing: 0.14em;
  color: var(--bm-hot);
  text-transform: uppercase;
  margin-bottom: 3px;
}
.chip-ends {
  color: #b9cad6;
}
.chip-btns {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  margin-top: 6px;
}
.chip-btns .cut {
  background: #7a1530;
  color: #fff;
}
.muted {
  color: #8fa3b8;
  font-size: 11px;
}
.bm-rolemenu {
  transform: translate(-50%, 10px);
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 180px;
}
.bm-rolemenu button {
  text-align: left;
  padding: 4px 8px;
  color: #e9fbff;
  background: rgba(255, 42, 77, 0.08);
  border: 1px solid rgba(255, 42, 77, 0.35);
  font-size: 12.5px;
  cursor: pointer;
}
.bm-rolemenu button:hover {
  background: rgba(255, 42, 77, 0.3);
}
.bm-rolemenu button.cancel {
  background: transparent;
  border-color: transparent;
  color: #8fa3b8;
}

.bm-legend {
  position: absolute;
  left: 10px;
  bottom: 56px;
  z-index: 5;
  display: flex;
  flex-direction: column;
  gap: 2px;
  font: 9.5px var(--pawn-mono);
  letter-spacing: 0.1em;
  color: #8fa3b8;
  text-transform: uppercase;
  background: rgba(5, 8, 12, 0.7);
  padding: 6px 8px;
  border: 1px solid rgba(92, 225, 255, 0.15);
}
.bm-legend span {
  display: flex;
  align-items: center;
  gap: 6px;
}
.bm-legend i {
  display: inline-block;
  width: 22px;
  height: 0;
  border-top: 2px solid #e0324f;
}
.bm-legend .by-user i {
  border-top: 3px solid #ff1f45;
  box-shadow: 0 0 6px #ff1f45;
}
.bm-legend .by-goal i {
  border-top-color: #5ce1ff;
}
.bm-legend .role-evidence_against i {
  border-top: 2px dashed #ffb020;
}
.bm-legend .role-contradicts i {
  border-top: 2px dotted #ff4df0;
}
.bm-legend .ghost i {
  border-top: 2px dashed #5ce1ff;
  opacity: 0.7;
}
.bm-legend .new i {
  border-top: 2px solid #ffd23f;
}
.bm-legend a {
  color: var(--bm-neon);
  text-decoration: none;
  align-self: flex-end;
}
.bm-dialog {
  min-width: min(420px, 92vw);
  background: #070b10;
  border: 1px solid rgba(92, 225, 255, 0.4);
}
.bm-dialog .neon {
  background: #0c6a80;
  color: #fff;
}
@keyframes bm-glitch {
  0%,
  92%,
  100% {
    transform: translate(0, 0);
  }
  93% {
    transform: translate(-2px, 1px);
  }
  95% {
    transform: translate(2px, -1px);
  }
  97% {
    transform: translate(-1px, 0);
  }
}
@media (max-width: 720px) {
  .readout,
  .sub,
  .bm-legend {
    display: none;
  }
  .bm-search {
    width: 140px;
  }
}
@media (prefers-reduced-motion: reduce) {
  .glitch::before,
  .glitch::after {
    animation: none;
  }
}
</style>
