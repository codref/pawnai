<script setup lang="ts">
// A floating tab group above the split area: drag it by its header, resize
// from any edge or corner, dock it back, maximize it, or close it.
import { computed } from "vue";
import { useTabsStore } from "@/stores/tabs";
import { syncRoute } from "@/router";
import { confirmAction } from "@/lib/dialogs";
import { workspaceEl } from "@/lib/workspace";
import GroupView from "./GroupView.vue";

const props = defineProps<{ groupId: string }>();
const tabs = useTabsStore();
const g = computed(() => tabs.group(props.groupId));
const MIN_W = 280;
const MIN_H = 180;

const style = computed(() => {
  const f = g.value?.float;
  if (!f) return {};
  if (f.max) return { inset: "8px", zIndex: String(f.z) };
  const bounds = workspaceEl()?.getBoundingClientRect();
  const maxW = bounds ? bounds.width : 4000;
  const maxH = bounds ? bounds.height : 4000;
  const w = Math.min(Math.max(MIN_W, f.w), maxW);
  const h = Math.min(Math.max(MIN_H, f.h), maxH);
  // Keep the whole window inside the workspace (a float opened by the
  // cursor near an edge shifts in rather than hanging off it).
  const x = Math.min(Math.max(0, f.x), Math.max(0, maxW - w));
  const y = Math.min(Math.max(0, f.y), Math.max(0, maxH - h));
  return { left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${h}px`, zIndex: String(f.z) };
});

type Mode = "move" | string;
let mode: Mode = "";
let sx = 0;
let sy = 0;
let start = { x: 0, y: 0, w: 0, h: 0 };

function begin(ev: PointerEvent, m: Mode): void {
  const f = g.value?.float;
  if (!f || ev.button !== 0) return;
  if (f.max && m !== "move") return;
  ev.preventDefault();
  mode = m;
  sx = ev.clientX;
  sy = ev.clientY;
  start = { x: f.x, y: f.y, w: f.w, h: f.h };
  if (f.max && m === "move") {
    // Dragging a maximized window restores it under the pointer.
    tabs.setFloat(props.groupId, { max: false });
  }
  document.body.classList.add("pawn-moving");
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", end, { once: true });
  window.addEventListener("pointercancel", end, { once: true });
}

function onMove(ev: PointerEvent): void {
  const dx = ev.clientX - sx;
  const dy = ev.clientY - sy;
  if (mode === "move") {
    tabs.setFloat(props.groupId, { x: start.x + dx, y: start.y + dy });
    return;
  }
  const patch: { x?: number; y?: number; w?: number; h?: number } = {};
  if (mode.includes("e")) patch.w = Math.max(MIN_W, start.w + dx);
  if (mode.includes("s")) patch.h = Math.max(MIN_H, start.h + dy);
  if (mode.includes("w")) {
    const w = Math.max(MIN_W, start.w - dx);
    patch.w = w;
    patch.x = start.x + (start.w - w);
  }
  if (mode.includes("n")) {
    const h = Math.max(MIN_H, start.h - dy);
    patch.h = h;
    patch.y = start.y + (start.h - h);
  }
  tabs.setFloat(props.groupId, patch);
}

function end(): void {
  mode = "";
  document.body.classList.remove("pawn-moving");
  window.removeEventListener("pointermove", onMove);
}

function dock(): void {
  tabs.dockFloat(props.groupId);
  syncRoute();
}

function toggleMax(): void {
  tabs.setFloat(props.groupId, { max: !g.value?.float?.max });
}

async function closeAll(): Promise<void> {
  const dirty = tabs.tabsIn(props.groupId).filter((t) => tabs.status[t.id]?.dirty);
  if (dirty.length) {
    const ok = await confirmAction(
      "Close window",
      `${dirty.length} tab(s) have unsaved changes. They stay open; close the rest?`,
      "Close the rest",
    );
    if (!ok) return;
  }
  tabs.closeGroup(props.groupId);
  syncRoute();
}

const EDGES = ["n", "s", "e", "w", "ne", "nw", "se", "sw"];
</script>

<template>
  <div v-if="g?.float" class="float-window shadow-8" :style="style" @pointerdown="tabs.raise(groupId)">
    <GroupView :group-id="groupId" floating @header-down="begin($event, 'move')">
      <template #actions>
        <div class="float-actions row no-wrap items-center" @dblclick="toggleMax">
          <q-btn flat dense round size="sm" icon="vertical_split" aria-label="Dock" @click="dock">
            <q-tooltip>Dock into the workspace</q-tooltip>
          </q-btn>
          <q-btn
            flat
            dense
            round
            size="sm"
            :icon="g.float.max ? 'close_fullscreen' : 'open_in_full'"
            :aria-label="g.float.max ? 'Restore' : 'Maximize'"
            @click="toggleMax"
          />
          <q-btn flat dense round size="sm" icon="close" aria-label="Close window" @click="closeAll" />
        </div>
      </template>
    </GroupView>
    <template v-if="!g.float.max">
      <div v-for="e in EDGES" :key="e" class="edge" :class="`edge-${e}`" @pointerdown.stop="begin($event, e)" />
    </template>
  </div>
</template>

<style scoped>
.float-window {
  position: absolute;
  pointer-events: auto;
  border: 1px solid var(--pawn-border);
  border-radius: 6px;
  overflow: hidden;
  background: var(--pawn-bg);
}
.float-actions {
  flex: none;
  padding: 0 4px;
  cursor: move;
  border-bottom: 1px solid var(--pawn-border);
}
.edge {
  position: absolute;
  z-index: 60;
}
.edge-n,
.edge-s {
  left: 8px;
  right: 8px;
  height: 6px;
  cursor: ns-resize;
}
.edge-n {
  top: -3px;
}
.edge-s {
  bottom: -3px;
}
.edge-e,
.edge-w {
  top: 8px;
  bottom: 8px;
  width: 6px;
  cursor: ew-resize;
}
.edge-e {
  right: -3px;
}
.edge-w {
  left: -3px;
}
.edge-ne,
.edge-nw,
.edge-se,
.edge-sw {
  width: 12px;
  height: 12px;
}
.edge-ne {
  top: -3px;
  right: -3px;
  cursor: nesw-resize;
}
.edge-sw {
  bottom: -3px;
  left: -3px;
  cursor: nesw-resize;
}
.edge-nw {
  top: -3px;
  left: -3px;
  cursor: nwse-resize;
}
.edge-se {
  bottom: -3px;
  right: -3px;
  cursor: nwse-resize;
}
</style>
