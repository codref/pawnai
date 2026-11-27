<script setup lang="ts">
// One node of the docked split tree: a group, or a row/column of children
// separated by drag gutters. Sizes are fractions stored on the node.
import { ref } from "vue";
import { useTabsStore, type LayoutNode } from "@/stores/tabs";
import GroupView from "./GroupView.vue";

defineOptions({ name: "SplitView" });

const props = defineProps<{ node: LayoutNode }>();
const tabs = useTabsStore();
const box = ref<HTMLElement | null>(null);

const MIN = 0.08;
let dragIdx = -1;
let start = 0;
let total = 1;
let startSizes: number[] = [];

function onDown(ev: PointerEvent, idx: number): void {
  if (ev.button !== 0 || props.node.type !== "split" || !box.value) return;
  ev.preventDefault();
  dragIdx = idx;
  const r = box.value.getBoundingClientRect();
  const row = props.node.dir === "row";
  total = row ? r.width : r.height;
  start = row ? ev.clientX : ev.clientY;
  startSizes = [...props.node.sizes];
  (ev.target as HTMLElement).setPointerCapture(ev.pointerId);
  document.body.classList.add(`pawn-resizing-${props.node.dir}`);
}

function onMove(ev: PointerEvent): void {
  if (dragIdx < 0 || props.node.type !== "split") return;
  const pos = props.node.dir === "row" ? ev.clientX : ev.clientY;
  const delta = (pos - start) / (total || 1);
  const a = startSizes[dragIdx];
  const b = startSizes[dragIdx + 1];
  const na = Math.min(a + b - MIN, Math.max(MIN, a + delta));
  const sizes = [...startSizes];
  sizes[dragIdx] = na;
  sizes[dragIdx + 1] = a + b - na;
  tabs.setSizes(props.node.id, sizes);
}

function onUp(ev: PointerEvent): void {
  (ev.target as HTMLElement).releasePointerCapture?.(ev.pointerId);
  if (props.node.type === "split") document.body.classList.remove(`pawn-resizing-${props.node.dir}`);
  dragIdx = -1;
}

/** Double-click a gutter: share the two neighbours equally. */
function equalize(idx: number): void {
  if (props.node.type !== "split") return;
  const sizes = [...props.node.sizes];
  const sum = sizes[idx] + sizes[idx + 1];
  sizes[idx] = sizes[idx + 1] = sum / 2;
  tabs.setSizes(props.node.id, sizes);
}
</script>

<template>
  <GroupView v-if="node.type === 'group'" :group-id="node.id" class="split-fill" />
  <div v-else ref="box" class="split" :class="`split-${node.dir}`">
    <template v-for="(child, i) in node.children" :key="child.type === 'group' ? child.id : child.id">
      <div class="split-cell" :style="{ flex: `${node.sizes[i] ?? 1} 1 0` }">
        <SplitView :node="child" />
      </div>
      <div
        v-if="i < node.children.length - 1"
        class="gutter"
        :class="`gutter-${node.dir}`"
        role="separator"
        :aria-orientation="node.dir === 'row' ? 'vertical' : 'horizontal'"
        title="Drag to resize · double-click to share equally"
        @pointerdown="onDown($event, i)"
        @pointermove="onMove"
        @pointerup="onUp"
        @pointercancel="onUp"
        @dblclick="equalize(i)"
      />
    </template>
  </div>
</template>

<style scoped>
.split,
.split-fill {
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
}
.split {
  display: flex;
}
.split-row {
  flex-direction: row;
}
.split-column {
  flex-direction: column;
}
.split-cell {
  position: relative;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}
.gutter {
  flex: none;
  background: var(--pawn-border);
  position: relative;
  z-index: 2;
}
.gutter-row {
  width: 1px;
  cursor: col-resize;
}
.gutter-column {
  height: 1px;
  cursor: row-resize;
}
.gutter::after {
  content: "";
  position: absolute;
  inset: -3px;
}
.gutter-row::after {
  inset: 0 -3px;
}
.gutter-column::after {
  inset: -3px 0;
}
.gutter:hover {
  background: var(--q-primary);
}
</style>
