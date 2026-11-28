<script setup lang="ts" generic="V extends string">
// One selector for view switches across the portal (sidebar views, inbox
// sections, scopes, theme mode). A tablist with a sliding indicator; arrow
// keys move the selection. The cyberpunk skin restyles it in styles.css.
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { nextSegmentIndex, type SegmentOption } from "@/lib/segment";

const props = withDefaults(
  defineProps<{
    modelValue: V;
    options: SegmentOption<V>[];
    size?: "sm" | "md";
    /** Show icons only; labels move to tooltips. */
    iconOnly?: boolean;
    /** Stretch segments to fill the row. */
    stretch?: boolean;
    ariaLabel?: string;
  }>(),
  { size: "md", iconOnly: false, stretch: false, ariaLabel: undefined },
);
const emit = defineEmits<{ (e: "update:modelValue", value: V): void }>();

const root = ref<HTMLElement | null>(null);
const buttons = ref<HTMLElement[]>([]);
const indicator = ref({ left: 0, width: 0, ready: false });

const activeIndex = computed(() => props.options.findIndex((o) => o.value === props.modelValue));

function measure(): void {
  const el = buttons.value[activeIndex.value];
  if (!el) {
    indicator.value = { left: 0, width: 0, ready: false };
    return;
  }
  indicator.value = { left: el.offsetLeft, width: el.offsetWidth, ready: true };
}

let observer: ResizeObserver | null = null;
onMounted(() => {
  measure();
  if (typeof ResizeObserver !== "undefined" && root.value) {
    observer = new ResizeObserver(() => measure());
    observer.observe(root.value);
  }
});
onBeforeUnmount(() => observer?.disconnect());
watch(
  () => [props.modelValue, props.options.map((o) => `${o.label}:${o.count ?? ""}`).join("|")],
  () => void nextTick(measure),
);

function select(value: V): void {
  if (value !== props.modelValue) emit("update:modelValue", value);
}

function onKey(ev: KeyboardEvent): void {
  const i = nextSegmentIndex(ev.key, activeIndex.value, props.options.length);
  if (i === null) return;
  ev.preventDefault();
  select(props.options[i].value);
  void nextTick(() => buttons.value[i]?.focus());
}

function setButton(el: unknown, i: number): void {
  if (el instanceof HTMLElement) buttons.value[i] = el;
}
</script>

<template>
  <div
    ref="root"
    class="seg"
    :class="[`seg-${size}`, { 'seg-stretch': stretch, 'seg-icon-only': iconOnly }]"
    role="tablist"
    :aria-label="ariaLabel"
    @keydown="onKey"
  >
    <span
      class="seg-indicator"
      :class="{ ready: indicator.ready }"
      :style="{ transform: `translateX(${indicator.left}px)`, width: `${indicator.width}px` }"
      aria-hidden="true"
    />
    <button
      v-for="(opt, i) in options"
      :key="opt.value"
      :ref="(el) => setButton(el, i)"
      type="button"
      role="tab"
      class="seg-btn"
      :class="{ active: opt.value === modelValue }"
      :aria-selected="opt.value === modelValue"
      :aria-label="iconOnly ? opt.label : undefined"
      :tabindex="opt.value === modelValue ? 0 : -1"
      @click="select(opt.value)"
    >
      <q-icon v-if="opt.icon" :name="opt.icon" class="seg-icon" />
      <span v-if="!iconOnly" class="seg-label">{{ opt.label }}</span>
      <span v-if="opt.count" class="seg-count">{{ opt.count > 999 ? "999+" : opt.count }}</span>
      <q-tooltip v-if="iconOnly || opt.tooltip" :delay="400">{{ opt.tooltip || opt.label }}</q-tooltip>
    </button>
  </div>
</template>

<style scoped>
.seg {
  position: relative;
  display: inline-flex;
  align-items: stretch;
  gap: 2px;
  padding: 2px;
  border-radius: 8px;
  background: var(--pawn-chip);
  max-width: 100%;
  min-width: 0;
}
.seg-stretch {
  display: flex;
}
.seg-stretch .seg-btn {
  flex: 1 1 0;
}
.seg-indicator {
  position: absolute;
  top: 2px;
  bottom: 2px;
  left: 0;
  border-radius: 6px;
  background: var(--pawn-bg);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.18), 0 0 0 1px var(--pawn-border);
  opacity: 0;
  pointer-events: none;
}
.seg-indicator.ready {
  opacity: 1;
  transition:
    transform 0.18s ease,
    width 0.18s ease;
}
.seg-btn {
  position: relative;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  min-width: 0;
  border: 0;
  background: transparent;
  color: var(--pawn-muted);
  font: inherit;
  font-weight: 500;
  border-radius: 6px;
  cursor: pointer;
  white-space: nowrap;
  transition: color 0.12s;
}
.seg-btn:hover {
  color: inherit;
}
.seg-btn.active {
  color: var(--q-primary);
}
.body--dark .seg-btn.active {
  color: inherit;
}
.seg-btn:focus-visible {
  outline: 2px solid var(--q-primary);
  outline-offset: -2px;
}
.seg-md .seg-btn {
  height: 28px;
  padding: 0 10px;
  font-size: 0.82rem;
}
.seg-sm .seg-btn {
  height: 24px;
  padding: 0 8px;
  font-size: 0.76rem;
}
.seg-icon-only .seg-btn {
  padding: 0 7px;
}
.seg-icon {
  font-size: 16px;
  flex: none;
}
.seg-sm .seg-icon {
  font-size: 15px;
}
.seg-label {
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}
.seg-count {
  flex: none;
  font-size: 0.68rem;
  line-height: 1;
  padding: 2px 5px;
  border-radius: 8px;
  background: var(--q-primary);
  color: #fff;
  font-variant-numeric: tabular-nums;
}
@media (prefers-reduced-motion: reduce) {
  .seg-indicator.ready {
    transition: none;
  }
}
</style>
