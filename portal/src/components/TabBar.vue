<script setup lang="ts">
import { computed, ref } from "vue";
import { useTabsStore, type Tab } from "@/stores/tabs";
import { useChatsStore } from "@/stores/chats";
import { activateTab, closeTab, closeTabs, syncRoute } from "@/router";
import { confirmAction } from "@/lib/dialogs";
import { DND_TAB, draggingTab } from "@/lib/dnd";

const props = defineProps<{ groupId: string }>();
const tabs = useTabsStore();
const chats = useChatsStore();
const overId = ref("");
const overEnd = ref(false);

const group = computed(() => tabs.group(props.groupId));
const floating = computed(() => Boolean(group.value?.float));
const focused = computed(() => tabs.focusedId === props.groupId);
const multiGroup = computed(() => tabs.groups.length > 1);

function baseName(path: string): string {
  return path.split("/").pop() ?? path;
}

function title(tab: Tab): string {
  const st = tabs.status[tab.id];
  if (st?.title) return st.title;
  if (tab.kind === "chat") return chats.byId.get(tab.ref)?.title || "New chat";
  if (tab.kind === "clip") return "Clip";
  if (tab.kind === "inbox") return "Inbox";
  if (tab.kind === "settings") return "Settings";
  if (tab.kind === "sessions") return "Sessions";
  if (tab.kind === "session") return "Session";
  if (tab.kind === "day") return "Day";
  if (tab.kind === "speakers") return "Speakers";
  if (tab.kind === "assistants") return "Assistants";
  if (tab.kind === "brainmap") return "Brainmap";
  return baseName(tab.ref);
}

function icon(tab: Tab): string {
  if (tab.kind === "chat") {
    const c = chats.byId.get(tab.ref);
    return c?.origin && !c.origin.promoted ? "question_answer" : "forum";
  }
  if (tab.kind === "clip") return "content_cut";
  if (tab.kind === "inbox") return "inbox";
  if (tab.kind === "settings") return "settings";
  if (tab.kind === "sessions" || tab.kind === "session") return "graphic_eq";
  if (tab.kind === "day") return "calendar_today";
  if (tab.kind === "speakers") return "record_voice_over";
  if (tab.kind === "assistants") return "devices";
  if (tab.kind === "brainmap") return "hub";
  if (/\.(png|jpe?g|gif|webp|svg|bmp|avif)$/i.test(tab.ref)) return "image";
  if (/\.pdf$/i.test(tab.ref)) return "picture_as_pdf";
  return "description";
}

const items = computed(() =>
  tabs
    .tabsIn(props.groupId)
    .map((t) => ({ tab: t, title: title(t), icon: icon(t), status: tabs.status[t.id] ?? {} })),
);

async function close(tab: Tab): Promise<void> {
  if (tabs.status[tab.id]?.dirty) {
    const ok = await confirmAction("Close tab", `${title(tab)} has unsaved changes. Discard them?`, "Discard");
    if (!ok) return;
  }
  closeTab(tab.id);
}

function closable(t: Tab): boolean {
  return !t.sticky && !tabs.status[t.id]?.dirty;
}

function closeOthers(tab: Tab): void {
  closeTabs(tabs.tabsIn(props.groupId).filter((t) => t.id !== tab.id && closable(t)).map((t) => t.id));
  activateTab(tab.id);
}

function closeRight(tab: Tab): void {
  const list = tabs.tabsIn(props.groupId);
  const idx = list.findIndex((t) => t.id === tab.id);
  closeTabs(list.slice(idx + 1).filter(closable).map((t) => t.id));
}

function layout(fn: () => void): void {
  fn();
  syncRoute();
}

function onMouseDown(ev: MouseEvent, tab: Tab): void {
  if (ev.button === 1) {
    ev.preventDefault();
    if (!tab.sticky) void close(tab);
  }
}

function onDragStart(ev: DragEvent, tab: Tab): void {
  draggingTab.value = tab.id;
  ev.dataTransfer?.setData(DND_TAB, tab.id);
  if (ev.dataTransfer) ev.dataTransfer.effectAllowed = "move";
}

function onDragOver(ev: DragEvent, tab: Tab): void {
  if (!draggingTab.value) return;
  ev.preventDefault();
  ev.stopPropagation();
  overId.value = tab.id;
  overEnd.value = false;
}

function onDrop(ev: DragEvent, tab: Tab): void {
  ev.preventDefault();
  ev.stopPropagation();
  const id = draggingTab.value;
  if (id && id !== tab.id) layout(() => tabs.move(id, tab.id));
  reset();
}

function onBarDragOver(ev: DragEvent): void {
  if (!draggingTab.value) return;
  ev.preventDefault();
  overEnd.value = true;
  overId.value = "";
}

function onBarDrop(ev: DragEvent): void {
  ev.preventDefault();
  const id = draggingTab.value;
  if (id) layout(() => tabs.moveTab(id, props.groupId));
  reset();
}

function reset(): void {
  draggingTab.value = null;
  overId.value = "";
  overEnd.value = false;
}
</script>

<template>
  <div
    class="tab-bar"
    :class="{ focused, 'over-end': overEnd }"
    role="tablist"
    @dragover="onBarDragOver"
    @dragleave="overEnd = false"
    @drop="onBarDrop"
  >
    <div
      v-for="it in items"
      :key="it.tab.id"
      class="tab"
      :class="{
        active: it.tab.id === group?.activeId,
        preview: !it.tab.pinned,
        sticky: it.tab.sticky,
        busy: it.status.busy,
        over: overId === it.tab.id && draggingTab !== it.tab.id,
      }"
      role="tab"
      :aria-selected="it.tab.id === group?.activeId"
      draggable="true"
      :title="it.tab.kind === 'file' ? it.tab.ref : it.title"
      @click="activateTab(it.tab.id)"
      @dblclick="tabs.pin(it.tab.id)"
      @mousedown="onMouseDown($event, it.tab)"
      @dragstart="onDragStart($event, it.tab)"
      @dragover="onDragOver($event, it.tab)"
      @dragleave="overId = ''"
      @drop="onDrop($event, it.tab)"
      @dragend="reset"
    >
      <q-spinner v-if="it.status.busy" size="13px" class="tab-icon" />
      <q-icon v-else :name="it.icon" size="15px" class="tab-icon" />
      <span class="tab-title ellipsis-1">{{ it.title }}</span>
      <span v-if="it.status.dirty" class="tab-dirty" aria-label="Unsaved" />
      <button
        v-if="it.tab.sticky"
        class="tab-close tab-pin"
        :aria-label="`Unpin ${it.title}`"
        @click.stop="tabs.setSticky(it.tab.id, false)"
      >
        <q-icon name="push_pin" size="13px" />
      </button>
      <button v-else class="tab-close" :aria-label="`Close ${it.title}`" @click.stop="close(it.tab)">
        <q-icon name="close" size="14px" />
      </button>

      <q-menu context-menu touch-position auto-close>
        <q-list dense style="min-width: 200px">
          <q-item v-if="!it.tab.pinned" clickable @click="tabs.pin(it.tab.id)">
            <q-item-section>Keep open</q-item-section>
          </q-item>
          <q-item clickable @click="tabs.setSticky(it.tab.id, !it.tab.sticky)">
            <q-item-section>{{ it.tab.sticky ? "Unpin" : "Pin" }}</q-item-section>
          </q-item>
          <q-separator />
          <q-item clickable @click="layout(() => tabs.splitTo(it.tab.id, groupId, 'right'))">
            <q-item-section>Split right</q-item-section>
          </q-item>
          <q-item clickable @click="layout(() => tabs.splitTo(it.tab.id, groupId, 'bottom'))">
            <q-item-section>Split down</q-item-section>
          </q-item>
          <q-item v-if="!floating" clickable @click="layout(() => tabs.floatTab(it.tab.id))">
            <q-item-section>Open in floating window</q-item-section>
          </q-item>
          <q-item v-else clickable @click="layout(() => tabs.moveTab(it.tab.id, tabs.docked[0]))">
            <q-item-section>Move to the workspace</q-item-section>
          </q-item>
          <q-separator />
          <q-item v-if="!it.tab.sticky" clickable @click="close(it.tab)"><q-item-section>Close</q-item-section></q-item>
          <q-item clickable @click="closeOthers(it.tab)"><q-item-section>Close others</q-item-section></q-item>
          <q-item clickable @click="closeRight(it.tab)"><q-item-section>Close to the right</q-item-section></q-item>
          <q-item v-if="multiGroup && !floating" clickable @click="layout(() => tabs.closeGroup(groupId))">
            <q-item-section>Close group</q-item-section>
          </q-item>
        </q-list>
      </q-menu>
    </div>
  </div>
</template>

<style scoped>
.tab-bar {
  display: flex;
  align-items: stretch;
  overflow-x: auto;
  overflow-y: hidden;
  min-height: 36px;
  min-width: 0;
  scrollbar-width: thin;
  background: var(--pawn-tabbar);
}
.tab {
  display: flex;
  align-items: center;
  gap: 6px;
  max-width: 220px;
  min-width: 90px;
  padding: 0 6px 0 10px;
  border-right: 1px solid var(--pawn-border);
  cursor: pointer;
  user-select: none;
  font-size: 0.85rem;
  color: var(--pawn-muted);
  position: relative;
  flex: none;
}
.tab:hover {
  background: var(--pawn-hover);
}
.tab.active {
  background: var(--pawn-bg);
  color: inherit;
}
.tab-bar.over-end {
  box-shadow: inset -2px 0 0 var(--q-primary);
}
.tab.active::before {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  height: 2px;
  background: var(--pawn-muted);
}
.tab-bar.focused .tab.active::before {
  background: var(--q-primary);
}
.tab.sticky {
  min-width: 0;
}
.tab.sticky .tab-pin {
  opacity: 0.7;
}
.tab.preview .tab-title {
  font-style: italic;
}
.tab.over {
  box-shadow: inset 2px 0 0 var(--q-primary);
}
.tab-icon {
  flex: none;
  opacity: 0.8;
}
.tab-title {
  flex: 1;
}
.tab-dirty {
  flex: none;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
}
.tab-close {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border: 0;
  border-radius: 4px;
  background: transparent;
  color: inherit;
  cursor: pointer;
  opacity: 0;
}
.tab:hover .tab-close,
.tab.active .tab-close {
  opacity: 0.7;
}
.tab-close:hover {
  opacity: 1;
  background: var(--pawn-hover);
}
</style>
