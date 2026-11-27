<script setup lang="ts">
// Drag strip on a drawer's inner edge. Emits the new width while dragging;
// double-click resets to the default.
const props = withDefaults(
  defineProps<{ side: "left" | "right"; width: number; min?: number; max?: number; reset?: number }>(),
  { min: 200, max: 640, reset: 300 },
);
const emit = defineEmits<{ (e: "update:width", value: number): void }>();

let startX = 0;
let startWidth = 0;

function clamp(v: number): number {
  return Math.round(Math.min(props.max, Math.max(props.min, v)));
}

function onDown(ev: PointerEvent): void {
  if (ev.button !== 0) return;
  ev.preventDefault();
  startX = ev.clientX;
  startWidth = props.width;
  (ev.target as HTMLElement).setPointerCapture(ev.pointerId);
  document.body.classList.add("pawn-resizing");
}

function onMove(ev: PointerEvent): void {
  if (!document.body.classList.contains("pawn-resizing")) return;
  const dx = ev.clientX - startX;
  emit("update:width", clamp(props.side === "left" ? startWidth + dx : startWidth - dx));
}

function onUp(ev: PointerEvent): void {
  (ev.target as HTMLElement).releasePointerCapture?.(ev.pointerId);
  document.body.classList.remove("pawn-resizing");
}
</script>

<template>
  <div
    class="resize-handle"
    :class="`resize-${side}`"
    role="separator"
    aria-orientation="vertical"
    :aria-valuenow="width"
    title="Drag to resize · double-click to reset"
    @pointerdown="onDown"
    @pointermove="onMove"
    @pointerup="onUp"
    @pointercancel="onUp"
    @dblclick="emit('update:width', reset)"
  />
</template>

<style scoped>
.resize-handle {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 7px;
  z-index: 10;
  cursor: col-resize;
  touch-action: none;
}
.resize-left {
  right: 0;
}
.resize-right {
  left: 0;
}
.resize-handle::after {
  content: "";
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: transparent;
  transition: background 0.15s;
}
.resize-left::after {
  right: 0;
}
.resize-right::after {
  left: 0;
}
.resize-handle:hover::after,
:global(body.pawn-resizing) .resize-handle::after {
  background: var(--q-primary);
}
</style>
