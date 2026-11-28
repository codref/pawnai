<script setup lang="ts">
// Shared day/week grid: hour rows, overlap columns, now line, and
// drag-to-create / move / resize. Cards are filled by the parent slot.
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { layoutOverlaps, minutesToPx, pxToMinutes, snapMinutes } from "@/lib/timegrid";
import type { DayColumn, GridItem } from "@/lib/planner";

const props = defineProps<{
  columns: DayColumn[];
  pxPerHour: number;
  nowMinutes: number | null;
  selectedId: string;
}>();

const emit = defineEmits<{
  select: [id: string];
  open: [id: string];
  pick: [date: string];
  create: [payload: { date: string; start: number; end: number; x: number; y: number }];
  commit: [payload: { id: string; date: string; start: number; end: number }];
}>();

const scrollEl = ref<HTMLElement | null>(null);
const colEls = ref<HTMLElement[]>([]);

interface Ghost {
  date: string;
  start: number;
  end: number;
}

interface Drag {
  mode: "move" | "resize" | "create" | "click";
  id: string;
  date: string;
  start: number;
  end: number;
  x: number;
  y: number;
  active: boolean;
  cancelled: boolean;
}

const ghost = ref<Ghost | null>(null);
let drag: Drag | null = null;
let clickTimer = 0;

const hours = computed(() => Array.from({ length: 24 }, (_, hour) => hour));
const canvasHeight = computed(() => `${minutesToPx(24 * 60, props.pxPerHour)}px`);
const datesKey = computed(() => props.columns.map((col) => col.date).join());

function topOf(minutes: number): string {
  return `${minutesToPx(minutes, props.pxPerHour)}px`;
}

function cardBox(item: GridItem & { col: number; cols: number }): Record<string, string> {
  const width = 100 / Math.max(1, item.cols);
  const height = Math.max(18, minutesToPx(Math.max(15, item.end - item.start), props.pxPerHour) - 2);
  return {
    top: topOf(item.start),
    height: `${height}px`,
    left: `calc(${item.col * width}% + 2px)`,
    width: `calc(${width}% - 4px)`,
  };
}

function ghostBox(item: Ghost): Record<string, string> {
  const height = Math.max(18, minutesToPx(Math.max(15, item.end - item.start), props.pxPerHour) - 2);
  return { top: topOf(item.start), height: `${height}px` };
}

function placed(col: DayColumn): (GridItem & { col: number; cols: number })[] {
  return layoutOverlaps(col.items.map((item) => ({ ...item, kind: item.kind })));
}

function bands(col: DayColumn): { start: number; end: number }[] {
  return col.items
    .filter((item) => item.focusBefore > 0)
    .map((item) => ({ start: Math.max(0, item.start - item.focusBefore), end: item.start }))
    .filter((band) => band.end > band.start);
}

function columnAt(clientX: number): DayColumn | null {
  const els = colEls.value;
  for (let i = 0; i < els.length; i++) {
    const rect = els[i]?.getBoundingClientRect();
    if (rect && clientX >= rect.left && clientX <= rect.right) return props.columns[i] ?? null;
  }
  return props.columns[props.columns.length - 1] ?? null;
}

function minutesAt(clientY: number, date: string): number {
  const el = colEls.value.find((_, i) => props.columns[i]?.date === date);
  const rect = el?.getBoundingClientRect();
  if (!rect) return 0;
  const raw = pxToMinutes(clientY - rect.top, props.pxPerHour);
  return Math.min(24 * 60 - 15, Math.max(0, snapMinutes(raw, 15)));
}

function listen(): void {
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  window.addEventListener("keydown", onEsc);
}

function silence(): void {
  window.removeEventListener("pointermove", onMove);
  window.removeEventListener("pointerup", onUp);
  window.removeEventListener("keydown", onEsc);
}

function onCardDown(item: GridItem, date: string, ev: PointerEvent): void {
  if (ev.button !== 0 || drag) return;
  const target = ev.target instanceof Element ? ev.target : null;
  if (target?.closest("button, a, input, textarea")) return;
  ev.stopPropagation();
  drag = {
    mode: item.draggable ? "move" : "click",
    id: item.id,
    date,
    start: item.start,
    end: item.end,
    x: ev.clientX,
    y: ev.clientY,
    active: false,
    cancelled: false,
  };
  listen();
}

function onResizeDown(item: GridItem, date: string, ev: PointerEvent): void {
  if (ev.button !== 0 || !item.draggable) return;
  ev.stopPropagation();
  drag = {
    mode: "resize",
    id: item.id,
    date,
    start: item.start,
    end: item.end,
    x: ev.clientX,
    y: ev.clientY,
    active: false,
    cancelled: false,
  };
  listen();
}

function onCreateDown(date: string, ev: PointerEvent): void {
  if (ev.button !== 0 || drag) return;
  const target = ev.target instanceof Element ? ev.target : null;
  if (target?.closest(".tg-card, button, a")) return;
  const start = minutesAt(ev.clientY, date);
  drag = {
    mode: "create",
    id: "",
    date,
    start,
    end: Math.min(24 * 60, start + 30),
    x: ev.clientX,
    y: ev.clientY,
    active: false,
    cancelled: false,
  };
  ghost.value = { date, start, end: drag.end };
  listen();
}

function onMove(ev: PointerEvent): void {
  if (!drag) return;
  const dx = ev.clientX - drag.x;
  const dy = ev.clientY - drag.y;
  if (!drag.active && Math.hypot(dx, dy) < 4) return;
  if (drag.mode === "click") return;
  drag.active = true;
  if (drag.mode === "create") {
    const at = minutesAt(ev.clientY, drag.date);
    const start = Math.min(drag.start, at);
    let end = Math.max(drag.start, at);
    if (end < start + 15) end = start + 15;
    ghost.value = { date: drag.date, start, end: Math.min(24 * 60, end) };
    return;
  }
  const delta = snapMinutes(pxToMinutes(dy, props.pxPerHour), 15);
  if (drag.mode === "resize") {
    let end = snapMinutes(drag.end + delta, 15);
    if (end < drag.start + 15) end = drag.start + 15;
    if (end > 24 * 60) end = 24 * 60;
    ghost.value = { date: drag.date, start: drag.start, end };
    return;
  }
  const col = columnAt(ev.clientX);
  const date = col?.date ?? drag.date;
  const duration = Math.max(15, drag.end - drag.start);
  let start = snapMinutes(drag.start + delta, 15);
  if (start < 0) start = 0;
  if (start + duration > 24 * 60) start = 24 * 60 - duration;
  ghost.value = { date, start, end: start + duration };
}

function finishClick(id: string): void {
  window.clearTimeout(clickTimer);
  clickTimer = window.setTimeout(() => emit("select", id), 220);
}

function onOpen(id: string): void {
  window.clearTimeout(clickTimer);
  emit("open", id);
}

function onUp(ev: PointerEvent): void {
  const current = drag;
  drag = null;
  silence();
  const drawn = ghost.value;
  ghost.value = null;
  if (!current || current.cancelled) return;
  if (current.mode === "create") {
    const start = drawn?.start ?? current.start;
    const end = drawn?.end ?? current.end;
    emit("create", {
      date: current.date,
      start,
      end: Math.max(start + 15, end),
      x: ev.clientX,
      y: ev.clientY,
    });
    return;
  }
  if (!current.active) {
    finishClick(current.id);
    return;
  }
  if (!drawn) return;
  emit("commit", { id: current.id, date: drawn.date, start: drawn.start, end: drawn.end });
}

function onEsc(ev: KeyboardEvent): void {
  if (ev.key !== "Escape" || !drag) return;
  drag.cancelled = true;
  ghost.value = null;
}

function scrollFocus(): void {
  const el = scrollEl.value;
  if (!el) return;
  const items = props.columns.flatMap((col) => col.items);
  const first = items.reduce((min, item) => Math.min(min, item.start), 24 * 60);
  const now = props.nowMinutes;
  const today = props.columns.some((col) => col.today);
  const nearNow =
    today &&
    now != null &&
    items.some((item) => now >= item.start - 90 && now <= item.end + 90);
  let minute = 7 * 60;
  if (nearNow && now != null) minute = Math.max(0, now - 60);
  else if (items.length) minute = Math.max(0, first - 30);
  el.scrollTop = minutesToPx(minute, props.pxPerHour);
}

watch(datesKey, () => void nextTick(scrollFocus));
onMounted(() => void nextTick(scrollFocus));
onBeforeUnmount(() => {
  silence();
  window.clearTimeout(clickTimer);
});

function setCol(el: Element | null, index: number): void {
  if (el instanceof HTMLElement) colEls.value[index] = el;
}
</script>

<template>
  <div class="tg column no-wrap">
    <div class="tg-head row no-wrap">
      <div class="tg-gutter" />
      <button
        v-for="col in columns"
        :key="col.date"
        type="button"
        class="tg-day col"
        :class="{ today: col.today }"
        @click="emit('pick', col.date)"
      >
        {{ col.label }}
      </button>
    </div>
    <div v-if="columns.some((col) => col.allDay.length)" class="tg-allday row no-wrap">
      <div class="tg-gutter tg-allday-label">all day</div>
      <div v-for="col in columns" :key="`all-${col.date}`" class="col tg-allday-col">
        <button
          v-for="chip in col.allDay"
          :key="chip.id"
          type="button"
          class="tg-allday-chip"
          :class="{ selected: chip.id === selectedId }"
          @click="emit('select', chip.id)"
          @dblclick.stop="onOpen(chip.id)"
        >
          {{ chip.title }}
        </button>
      </div>
    </div>
    <div ref="scrollEl" class="tg-scroll pawn-scroll-xy col">
      <div class="tg-canvas" :style="{ height: canvasHeight }">
        <div class="tg-gutter tg-hours">
          <div v-for="hour in hours" :key="hour" class="tg-hour" :style="{ top: topOf(hour * 60) }">
            {{ String(hour).padStart(2, "0") }}
          </div>
        </div>
        <div class="tg-cols">
          <div
            v-for="(col, index) in columns"
            :key="col.date"
            :ref="(el) => setCol(el as Element | null, index)"
            class="tg-col"
            @pointerdown="onCreateDown(col.date, $event)"
          >
            <div
              v-for="hour in hours"
              :key="`h-${hour}`"
              class="tg-line"
              :style="{ top: topOf(hour * 60) }"
            />
            <div
              v-for="hour in hours"
              :key="`m-${hour}`"
              class="tg-line half"
              :style="{ top: topOf(hour * 60 + 30) }"
            />
            <div
              v-for="(band, bi) in bands(col)"
              :key="`b-${bi}`"
              class="tg-band"
              :style="{ top: topOf(band.start), height: `${minutesToPx(band.end - band.start, pxPerHour)}px` }"
            />
            <div v-if="col.today && nowMinutes != null" class="tg-now" :style="{ top: topOf(nowMinutes) }">
              <span>now</span>
            </div>
            <div
              v-for="card in placed(col)"
              :key="card.id"
              class="tg-card"
              :class="[card.kind, { selected: card.id === selectedId }]"
              :style="cardBox(card)"
              role="button"
              :tabindex="0"
              @pointerdown="onCardDown(card, col.date, $event)"
              @dblclick.stop="onOpen(card.id)"
            >
              <slot name="card" :item="card" :compact="columns.length > 1" />
              <div
                v-if="card.draggable"
                class="tg-resize"
                @pointerdown="onResizeDown(card, col.date, $event)"
              />
            </div>
            <div v-if="ghost && ghost.date === col.date" class="tg-ghost" :style="ghostBox(ghost)" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tg {
  min-height: 0;
  min-width: 0;
  height: 100%;
}
.tg-gutter {
  width: 48px;
  flex: none;
}
.tg-head {
  border-bottom: 1px solid var(--pawn-border);
}
.tg-day {
  border: 0;
  background: transparent;
  color: inherit;
  padding: 6px 4px;
  font: inherit;
  font-size: 0.78rem;
  cursor: pointer;
}
.tg-day.today {
  color: var(--q-primary);
  font-weight: 650;
}
.tg-allday {
  border-bottom: 1px solid var(--pawn-border);
  max-height: 72px;
  overflow: auto;
  scrollbar-width: thin;
}
.tg-allday-label {
  font-size: 0.65rem;
  color: var(--pawn-muted);
  padding: 6px 4px 0 8px;
}
.tg-allday-col {
  padding: 4px;
  min-width: 0;
}
.tg-allday-chip {
  display: block;
  width: 100%;
  text-align: left;
  border: 0;
  border-radius: 4px;
  background: var(--pawn-chip);
  color: inherit;
  font: inherit;
  font-size: 0.75rem;
  padding: 2px 6px;
  margin-bottom: 3px;
  cursor: pointer;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.tg-allday-chip.selected {
  outline: 2px solid var(--q-primary);
}
.tg-scroll {
  min-height: 0;
}
.tg-canvas {
  position: relative;
  display: flex;
}
.tg-hours {
  position: relative;
}
.tg-hour {
  position: absolute;
  right: 6px;
  transform: translateY(-50%);
  font-size: 0.65rem;
  color: var(--pawn-muted);
  font-variant-numeric: tabular-nums;
}
.tg-cols {
  display: flex;
  flex: 1;
  min-width: 0;
  position: relative;
}
.tg-col {
  position: relative;
  flex: 1;
  min-width: 0;
  border-left: 1px solid var(--pawn-border);
}
.tg-line {
  position: absolute;
  left: 0;
  right: 0;
  border-top: 1px solid var(--pawn-border);
  pointer-events: none;
}
.tg-line.half {
  border-top-style: dashed;
  opacity: 0.45;
}
.tg-band {
  position: absolute;
  left: 0;
  right: 0;
  background: var(--pawn-focus-band);
  pointer-events: none;
}
.tg-now {
  position: absolute;
  left: 0;
  right: 0;
  border-top: 2px solid var(--pawn-now);
  z-index: 4;
  pointer-events: none;
}
.tg-now span {
  position: absolute;
  top: -8px;
  left: 4px;
  font-size: 0.62rem;
  color: var(--pawn-now);
  background: var(--pawn-bg);
  padding: 0 3px;
}
.tg-card {
  position: absolute;
  z-index: 2;
  overflow: hidden;
  border-radius: 6px;
  background: var(--pawn-active);
  border-left: 3px solid var(--q-primary);
  cursor: grab;
  text-align: left;
}
.tg-card.session {
  cursor: pointer;
  background: repeating-linear-gradient(
    -45deg,
    var(--pawn-session-stripe),
    var(--pawn-session-stripe) 6px,
    var(--pawn-session-stripe-2) 6px,
    var(--pawn-session-stripe-2) 12px
  );
  border-left-style: dashed;
}
.tg-card.selected {
  outline: 2px solid var(--q-primary);
  z-index: 3;
}
.tg-resize {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 6px;
  cursor: ns-resize;
}
.tg-ghost {
  position: absolute;
  left: 3px;
  right: 3px;
  border-radius: 6px;
  border: 1px dashed var(--pawn-unseen);
  background: var(--pawn-drop);
  pointer-events: none;
  z-index: 5;
}
</style>
