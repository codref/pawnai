<script setup lang="ts">
// One tab group: its tab strip and an empty body that Workspace teleports the
// group's panels into. While a tab (or a chat from the sidebar) is dragged
// over the body, edge zones split the group and the center moves into it.
import { computed, onBeforeUnmount, ref } from "vue";
import { registerSlot } from "@/lib/groupSlots";
import { useTabsStore, tabId, type Side } from "@/stores/tabs";
import { focusGroup, openChat, openFile, syncRoute } from "@/router";
import { DND_CHAT, DND_FILE, DND_TAB, draggingTab, hasType } from "@/lib/dnd";
import TabBar from "./TabBar.vue";
import HomePanel from "@/panels/HomePanel.vue";

const props = defineProps<{ groupId: string; floating?: boolean }>();
const emit = defineEmits<{ (e: "header-down", ev: PointerEvent): void }>();
const tabs = useTabsStore();

type Zone = Side | "center";
const zone = ref<Zone | null>(null);
const isFocused = computed(() => tabs.focusedId === props.groupId);
const empty = computed(() => !tabs.tabsIn(props.groupId).length);

function accepts(ev: DragEvent): boolean {
  return Boolean(draggingTab.value) || hasType(ev, DND_TAB) || hasType(ev, DND_CHAT) || hasType(ev, DND_FILE);
}

function zoneAt(ev: DragEvent, el: HTMLElement): Zone {
  const r = el.getBoundingClientRect();
  const x = (ev.clientX - r.left) / (r.width || 1);
  const y = (ev.clientY - r.top) / (r.height || 1);
  const edge = props.floating ? 0 : 0.22;
  if (x < edge) return "left";
  if (x > 1 - edge) return "right";
  if (y < edge) return "top";
  if (y > 1 - edge) return "bottom";
  return "center";
}

/** A panel inside took the drop itself (a chat attaching a dragged file). */
function takenInside(ev: DragEvent): boolean {
  return ev.defaultPrevented && !draggingTab.value && hasType(ev, DND_FILE) && !hasType(ev, DND_CHAT);
}

function onDragOver(ev: DragEvent): void {
  if (!accepts(ev)) return;
  if (takenInside(ev)) {
    zone.value = null;
    return;
  }
  ev.preventDefault();
  if (ev.dataTransfer) ev.dataTransfer.dropEffect = "move";
  zone.value = zoneAt(ev, ev.currentTarget as HTMLElement);
}

function onDragLeave(ev: DragEvent): void {
  const to = ev.relatedTarget as Node | null;
  if (!to || !(ev.currentTarget as HTMLElement).contains(to)) zone.value = null;
}

function onDrop(ev: DragEvent): void {
  const z = zone.value;
  zone.value = null;
  if (!z || !accepts(ev) || takenInside(ev)) return;
  ev.preventDefault();
  ev.stopPropagation();
  let id = draggingTab.value || ev.dataTransfer?.getData(DND_TAB) || "";
  draggingTab.value = null;
  if (!id) {
    const chatId = ev.dataTransfer?.getData(DND_CHAT);
    const filePath = chatId ? "" : ev.dataTransfer?.getData(DND_FILE);
    if (!chatId && !filePath) return;
    id = chatId ? tabId("chat", chatId) : tabId("file", filePath!);
    if (!tabs.find(id)) {
      tabs.focusGroup(props.groupId);
      if (chatId) openChat(chatId, { pin: true });
      else openFile(filePath!, { pin: true });
      if (z === "center") return;
    }
  }
  const tab = tabs.find(id);
  if (!tab) return;
  if (z === "center") {
    if (tab.groupId !== props.groupId) tabs.moveTab(id, props.groupId);
    else tabs.activate(id);
  } else {
    // Splitting a group with only this tab would leave it empty: no-op.
    if (tab.groupId === props.groupId && tabs.tabsIn(props.groupId).length === 1) return;
    tabs.splitTo(id, props.groupId, z);
  }
  syncRoute();
}

let slotEl: HTMLElement | null = null;
function setSlot(el: unknown): void {
  const next = el instanceof HTMLElement ? el : null;
  if (next === slotEl) return;
  const prev = slotEl;
  slotEl = next;
  registerSlot(props.groupId, next, prev);
}
onBeforeUnmount(() => registerSlot(props.groupId, null, slotEl));

function onHeaderDown(ev: PointerEvent): void {
  const el = ev.target as HTMLElement;
  if (el.closest(".tab, button, .q-btn, input")) return;
  emit("header-down", ev);
}
</script>

<template>
  <div
    class="group-view column no-wrap"
    :class="{ focused: isFocused, floating }"
    @pointerdown.capture="focusGroup(groupId)"
    @focusin="focusGroup(groupId)"
  >
    <div class="group-head row no-wrap items-stretch" @pointerdown="onHeaderDown">
      <TabBar :group-id="groupId" class="col" />
      <slot name="actions" />
    </div>
    <div
      class="col relative-position group-body"
      @dragover="onDragOver"
      @dragleave="onDragLeave"
      @drop="onDrop"
    >
      <div :id="`grp-${groupId}`" :ref="setSlot" class="group-slot" />
      <HomePanel v-if="empty && !floating" class="group-slot" />
      <div v-if="zone" class="drop-zone" :class="`zone-${zone}`" />
    </div>
  </div>
</template>

<style scoped>
.group-view {
  height: 100%;
  min-width: 0;
  min-height: 0;
  background: var(--pawn-bg);
}
.group-head {
  flex: none;
  border-bottom: 1px solid var(--pawn-border);
  background: var(--pawn-tabbar);
  min-height: 36px;
}
.group-body {
  min-height: 0;
}
.group-slot {
  position: absolute;
  inset: 0;
}
.drop-zone {
  position: absolute;
  background: var(--pawn-drop);
  border: 2px solid var(--q-primary);
  border-radius: 4px;
  pointer-events: none;
  z-index: 50;
  transition: all 80ms ease-out;
}
.zone-center {
  inset: 6px;
}
.zone-left {
  inset: 0 50% 0 0;
}
.zone-right {
  inset: 0 0 0 50%;
}
.zone-top {
  inset: 0 0 50% 0;
}
.zone-bottom {
  inset: 50% 0 0 0;
}
</style>
