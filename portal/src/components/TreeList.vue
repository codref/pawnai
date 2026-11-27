<script setup lang="ts" generic="T extends TreeItem">
// Flat, VS Code-style tree: chevrons, indentation and hover rows; indent
// guides only appear while the pointer is over the tree. Row content comes
// from the default slot; the component owns indentation, expand/collapse,
// lazy loading and selection highlighting.
import { computed } from "vue";
import type { TreeItem } from "@/lib/tree";

const props = withDefaults(
  defineProps<{ nodes: T[]; expanded: string[]; selected?: string | null; indent?: number; emptyLabel?: string }>(),
  { selected: null, indent: 14, emptyLabel: "" },
);
const emit = defineEmits<{
  (e: "update:expanded", value: string[]): void;
  (e: "select", node: T): void;
  (e: "lazy-load", node: T): void;
}>();

interface Row {
  node: T;
  depth: number;
  hasKids: boolean;
  open: boolean;
}

const expandedSet = computed(() => new Set(props.expanded));

const rows = computed<Row[]>(() => {
  const out: Row[] = [];
  const walk = (nodes: T[], depth: number) => {
    for (const node of nodes) {
      const kids = (node.children ?? []) as T[];
      const hasKids = Boolean(node.lazy || node.expandable || kids.length);
      const open = hasKids && expandedSet.value.has(node.key);
      out.push({ node, depth, hasKids, open });
      if (open) walk(kids, depth + 1);
    }
  };
  walk(props.nodes, 0);
  return out;
});

function toggle(row: Row): void {
  if (!row.hasKids) return;
  const key = row.node.key;
  if (row.open) {
    emit(
      "update:expanded",
      props.expanded.filter((k) => k !== key),
    );
  } else {
    emit("update:expanded", [...props.expanded, key]);
    if (row.node.lazy) emit("lazy-load", row.node);
  }
}

function onClick(row: Row): void {
  if (row.hasKids) toggle(row);
  emit("select", row.node);
}

function onKey(ev: KeyboardEvent, row: Row): void {
  if (ev.key === "Enter" || ev.key === " ") {
    ev.preventDefault();
    onClick(row);
  } else if (ev.key === "ArrowRight" && row.hasKids && !row.open) {
    toggle(row);
  } else if (ev.key === "ArrowLeft" && row.open) {
    toggle(row);
  }
}
</script>

<template>
  <div class="tree-list" role="tree" :style="{ '--indent': `${indent}px` }">
    <div
      v-for="row in rows"
      :key="row.node.key"
      class="tree-row"
      :class="{ selected: selected === row.node.key }"
      :style="{ '--depth': row.depth }"
      role="treeitem"
      :aria-expanded="row.hasKids ? row.open : undefined"
      tabindex="0"
      @click="onClick(row)"
      @keydown="onKey($event, row)"
    >
      <span class="tree-chevron" @click.stop="toggle(row)">
        <svg v-if="row.hasKids" viewBox="0 0 16 16" :class="{ open: row.open }" aria-hidden="true">
          <path d="M6 4l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </span>
      <div class="tree-content">
        <slot :node="row.node" :depth="row.depth" :open="row.open" />
      </div>
    </div>
    <div v-if="!rows.length && emptyLabel" class="text-caption text-muted q-pa-sm">{{ emptyLabel }}</div>
  </div>
</template>

<style scoped>
.tree-list {
  padding: 2px 4px;
  --guide: transparent;
}
.tree-list:hover {
  --guide: var(--pawn-guide);
}
.tree-row {
  display: flex;
  align-items: center;
  min-height: 28px;
  padding-left: calc(var(--depth) * var(--indent) + 2px);
  padding-right: 6px;
  border-radius: 6px;
  cursor: pointer;
  user-select: none;
  outline: none;
  /* Indent guides: one faint line per ancestor level, centered on its chevron. */
  background-image: repeating-linear-gradient(
    90deg,
    transparent 0,
    transparent calc(var(--indent) / 2 + 2px),
    var(--guide) calc(var(--indent) / 2 + 2px),
    var(--guide) calc(var(--indent) / 2 + 3px),
    transparent calc(var(--indent) / 2 + 3px),
    transparent var(--indent)
  );
  background-size: calc(var(--depth) * var(--indent)) 100%;
  background-repeat: no-repeat;
  transition: background-color 0.1s;
}
.tree-row:hover {
  background-color: var(--pawn-hover);
}
.tree-row:focus-visible {
  box-shadow: inset 0 0 0 1px var(--q-primary);
}
.tree-row.selected {
  background-color: var(--pawn-active);
  color: var(--q-primary);
}
.tree-chevron {
  flex: none;
  width: 16px;
  height: 16px;
  margin-right: 2px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--pawn-muted);
}
.tree-chevron svg {
  width: 14px;
  height: 14px;
  transition: transform 0.12s ease;
}
.tree-chevron svg.open {
  transform: rotate(90deg);
}
.tree-content {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
}
</style>
