<script setup lang="ts">
// Annotations of one or more targets of a view (a file body, a clip, every
// message of a chat): loads them, re-anchors each one on the rendered text,
// paints the highlights (lib/highlights) and shows margin markers. Anchors
// that no longer match are listed as orphaned, never dropped.
//
// Place it next to the annotated content, inside the same scrolling element:
// "gutter" mode is a column of note cards beside the body; "inline" mode is an
// absolute overlay (parent must be position: relative) with dots on the right.
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Notify } from "quasar";
import type { Annotation, AnnotationKind } from "@/api";
import { useAnnotationsStore } from "@/stores/annotations";
import { anchorRootOf, isTextAnchor, resolveAnchor } from "@/lib/anchoring";
import { colorOf, hitTest, register, setActive, unregister, type HighlightEntry } from "@/lib/highlights";
import { anchorFor } from "@/lib/targets";
import AnnotationPopover from "./AnnotationPopover.vue";

const props = withDefaults(
  defineProps<{
    kind: AnnotationKind;
    refs: string[];
    /** The v-target element of *ref*, or null when it isn't rendered. */
    rootFor: (ref: string) => HTMLElement | null;
    mode?: "gutter" | "inline";
    /** Chat of message annotations (side chats nest under it). */
    chatId?: string;
  }>(),
  { mode: "gutter", chatId: undefined },
);

interface Placed {
  ann: Annotation;
  range: Range | null;
  /** Offset from the layer's top, in px. */
  y: number;
  orphan: boolean;
}

const hostId = `ann-layer-${Math.random().toString(36).slice(2, 10)}`;
const store = useAnnotationsStore();
const layer = ref<HTMLElement | null>(null);
const placed = ref<Placed[]>([]);
const showOrphans = ref(false);

const anns = computed(() => store.forTargets(props.kind, props.refs));
const located = computed(() => placed.value.filter((p) => !p.orphan));
const orphans = computed(() => placed.value.filter((p) => p.orphan));

watch(
  () => [props.kind, props.refs.join("\n")] as const,
  () => void store.load(props.kind, props.refs),
  { immediate: true },
);

// ── measuring ────────────────────────────────────────────────────────────
function measure(): void {
  const host = layer.value;
  if (!host) return;
  const base = host.getBoundingClientRect().top;
  const out: Placed[] = [];
  const entries: HighlightEntry[] = [];
  for (const ann of anns.value) {
    const root = props.rootFor(ann.target_ref);
    if (!root) continue; // not rendered (yet)
    const top = root.getBoundingClientRect().top - base;
    if (!isTextAnchor(ann.anchor)) {
      out.push({ ann, range: null, y: top, orphan: false });
      continue;
    }
    const hit = resolveAnchor(root, ann.anchor);
    if (!hit) {
      out.push({ ann, range: null, y: top, orphan: true });
      continue;
    }
    entries.push({ annId: ann.id, color: ann.color, range: hit.range });
    out.push({ ann, range: hit.range, y: hit.range.getBoundingClientRect().top - base, orphan: false });
  }
  register(hostId, entries);
  placed.value = out.sort((a, b) => a.y - b.y);
  void handlePending();
}

let frame = 0;
function schedule(): void {
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    measure();
  });
}

watch(anns, schedule);

let mutations: MutationObserver | null = null;
let resizes: ResizeObserver | null = null;

onMounted(() => {
  const parent = layer.value?.parentElement;
  if (parent) {
    mutations = new MutationObserver((records) => {
      // Our own markers re-rendering must not trigger another pass.
      if (records.every((r) => layer.value?.contains(r.target))) return;
      schedule();
    });
    mutations.observe(parent, { childList: true, subtree: true, characterData: true });
    resizes = new ResizeObserver(schedule);
    resizes.observe(parent);
  }
  document.addEventListener("click", onDocClick);
  schedule();
});

onBeforeUnmount(() => {
  mutations?.disconnect();
  resizes?.disconnect();
  if (frame) cancelAnimationFrame(frame);
  document.removeEventListener("click", onDocClick);
  unregister(hostId);
  if (popAnn.value) setActive("");
});

// ── gutter layout ────────────────────────────────────────────────────────
const CARD_H = 54;
const ORPHAN_H = 30;
const cards = computed(() => {
  let bottom = orphans.value.length && props.mode === "gutter" ? ORPHAN_H : 0;
  return located.value.map((p) => {
    const top = Math.max(p.y, bottom);
    bottom = top + CARD_H + 6;
    return { ...p, top };
  });
});

/** Inline dots: side by side when several start on the same line. */
const dots = computed(() => {
  const seen = new Map<number, number>();
  return placed.value.map((p) => {
    const row = Math.round(p.y / 8);
    const n = seen.get(row) ?? 0;
    seen.set(row, n + 1);
    return { ...p, nth: n };
  });
});

function firstLine(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

// ── popover ──────────────────────────────────────────────────────────────
const popOpen = ref(false);
const popAnn = ref<Annotation | null>(null);
const popOrphan = ref(false);
const popX = ref(0);
const popY = ref(0);

function openFor(id: string, at?: { x: number; y: number }): void {
  const p = placed.value.find((x) => x.ann.id === id);
  if (!p) return;
  let x = at?.x ?? 0;
  let y = at?.y ?? 0;
  if (!at) {
    const rect = p.range?.getBoundingClientRect() ?? props.rootFor(p.ann.target_ref)?.getBoundingClientRect();
    if (rect) {
      x = rect.left;
      y = rect.bottom;
    }
  }
  popAnn.value = p.ann;
  popOrphan.value = p.orphan;
  popX.value = x;
  popY.value = y;
  popOpen.value = true;
  setActive(p.orphan ? "" : id);
}

watch(popOpen, (v) => {
  if (!v) setActive("");
});

function onMarker(p: Placed, ev: MouseEvent): void {
  const r = (ev.currentTarget as HTMLElement).getBoundingClientRect();
  openFor(p.ann.id, p.orphan || props.mode === "inline" ? { x: r.left, y: r.bottom } : undefined);
}

function onDocClick(ev: MouseEvent): void {
  if (ev.button !== 0 || ev.defaultPrevented) return;
  const el = ev.target as HTMLElement | null;
  if (!el || el.closest("a, button, input, textarea, .q-menu")) return;
  const sel = window.getSelection();
  if (sel && !sel.isCollapsed) return;
  const id = hitTest(ev.clientX, ev.clientY);
  if (id && placed.value.some((p) => p.ann.id === id)) openFor(id, { x: ev.clientX, y: ev.clientY + 4 });
}

/** Re-attach an orphan to the text selected now inside its target. */
async function reattach(id: string): Promise<void> {
  const p = placed.value.find((x) => x.ann.id === id);
  const root = p ? props.rootFor(p.ann.target_ref) : null;
  const sel = window.getSelection();
  const range = sel && sel.rangeCount && !sel.isCollapsed ? sel.getRangeAt(0) : null;
  if (!p || !root || !range || !root.contains(range.commonAncestorContainer)) {
    Notify.create({ message: "Select the text to attach the note to first", position: "bottom-right" });
    return;
  }
  const anchor = anchorFor(anchorRootOf(root, range.commonAncestorContainer), range);
  if (!anchor) return;
  popOpen.value = false;
  sel?.removeAllRanges();
  await store.patch(id, { anchor });
}

// ── focus / edit requests (search hits, Annotate, side chat back-links) ──
const pending = new Map<string, { kind: "focus" | "edit"; at: number }>();
watch(
  () => store.focusRequest,
  (r) => {
    if (r) {
      pending.set(r.id, { kind: "focus", at: r.at });
      schedule();
    }
  },
);
watch(
  () => store.editRequest,
  (r) => {
    if (r) {
      pending.set(r.id, { kind: "edit", at: r.at });
      schedule();
    }
  },
);

let flashTimer = 0;
async function handlePending(): Promise<void> {
  for (const [id, req] of [...pending]) {
    if (Date.now() - req.at > 15_000) {
      pending.delete(id);
      continue;
    }
    const p = placed.value.find((x) => x.ann.id === id);
    if (!p) continue;
    pending.delete(id);
    const node = p.range?.startContainer ?? props.rootFor(p.ann.target_ref);
    const el = node instanceof HTMLElement ? node : node?.parentElement;
    el?.scrollIntoView({ block: req.kind === "focus" ? "center" : "nearest", behavior: "smooth" });
    if (req.kind === "edit") {
      await nextTick();
      setTimeout(() => openFor(id), 250);
    } else {
      if (p.orphan) showOrphans.value = true;
      setActive(id);
      clearTimeout(flashTimer);
      flashTimer = window.setTimeout(() => {
        if (!popOpen.value) setActive("");
      }, 1800);
    }
  }
}
</script>

<template>
  <div ref="layer" class="ann-layer" :class="[`ann-${mode}`, { 'has-notes': placed.length }]">
    <template v-if="mode === 'gutter'">
      <div v-if="orphans.length" class="ann-orphans">
        <q-btn
          flat
          dense
          no-caps
          size="sm"
          icon="link_off"
          :label="`${orphans.length} orphaned`"
          class="text-muted"
          @click="showOrphans = !showOrphans"
        />
        <div v-if="showOrphans" class="ann-orphan-list">
          <div
            v-for="p in orphans"
            :key="p.ann.id"
            class="ann-card orphan"
            :class="`ann-b-${colorOf(p.ann.color)}`"
            @click="onMarker(p, $event)"
          >
            <div class="ann-card-quote">“{{ p.ann.anchor?.exact }}”</div>
            <div v-if="p.ann.body_md" class="ann-card-body ellipsis-1">{{ firstLine(p.ann.body_md) }}</div>
          </div>
        </div>
      </div>
      <div
        v-for="c in cards"
        :key="c.ann.id"
        class="ann-card"
        :class="`ann-b-${colorOf(c.ann.color)}`"
        :style="{ top: `${c.top}px` }"
        @click="onMarker(c, $event)"
      >
        <div v-if="c.ann.body_md" class="ann-card-body ellipsis-2">{{ firstLine(c.ann.body_md) }}</div>
        <div v-else class="ann-card-quote ellipsis-2">“{{ c.ann.anchor?.exact }}”</div>
        <div v-if="c.ann.side_chat_id || c.ann.tags.length" class="row items-center no-wrap ann-card-meta">
          <q-icon v-if="c.ann.side_chat_id" name="forum" size="12px" class="q-mr-xs" />
          <span class="ellipsis-1">{{ c.ann.tags.map((t) => `#${t}`).join(" ") }}</span>
        </div>
      </div>
    </template>

    <template v-else>
      <button
        v-for="d in dots"
        :key="d.ann.id"
        type="button"
        class="ann-dot"
        :class="[`ann-c-${colorOf(d.ann.color)}`, { orphan: d.orphan, linked: d.ann.side_chat_id }]"
        :style="{ top: `${d.y + 4}px`, right: `${-14 - d.nth * 12}px` }"
        :aria-label="d.orphan ? 'Orphaned annotation' : 'Annotation'"
        @click="onMarker(d, $event)"
      >
        <q-tooltip anchor="center left" self="center right">
          {{ d.orphan ? "Orphaned: " : "" }}{{ firstLine(d.ann.body_md) || `“${d.ann.anchor?.exact ?? ""}”` }}
        </q-tooltip>
      </button>
    </template>

    <AnnotationPopover
      v-model="popOpen"
      :annotation="popAnn"
      :x="popX"
      :y="popY"
      :orphaned="popOrphan"
      :chat-id="chatId"
      @reattach="reattach"
    />
  </div>
</template>

<style scoped>
.ann-layer {
  position: relative;
}
.ann-gutter {
  width: 0;
  flex: none;
  transition: width 0.15s;
}
.ann-gutter.has-notes {
  width: 220px;
  margin-left: 12px;
}
.ann-inline {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
.ann-card {
  position: absolute;
  left: 0;
  right: 0;
  height: 54px;
  overflow: hidden;
  padding: 4px 8px;
  border-left: 3px solid;
  border-radius: 4px;
  background: var(--pawn-chip);
  font-size: 0.78rem;
  line-height: 1.3;
  cursor: pointer;
}
.ann-card:hover {
  background: var(--pawn-hover);
  outline: 1px solid var(--pawn-border);
}
.ann-card-quote {
  color: var(--pawn-muted);
  font-style: italic;
}
.ann-card-meta {
  color: var(--pawn-muted);
  font-size: 0.7rem;
}
.ann-orphans {
  position: relative;
  z-index: 1;
}
.ann-orphan-list {
  background: var(--pawn-bg);
  padding-bottom: 4px;
}
.ann-card.orphan {
  position: relative;
  height: auto;
  margin-bottom: 4px;
}
.ann-card.orphan .ann-card-quote {
  text-decoration: line-through;
}
.ann-dot {
  position: absolute;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  border: 1px solid var(--pawn-border);
  padding: 0;
  cursor: pointer;
  pointer-events: auto;
}
.ann-dot.linked {
  box-shadow: 0 0 0 2px var(--pawn-active);
}
.ann-dot.orphan {
  background: transparent !important;
  border: 2px dashed var(--pawn-muted);
}
</style>
