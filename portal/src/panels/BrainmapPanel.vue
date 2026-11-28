<script setup lang="ts">
// Brainmap tool tab: one tab per board (`brainmap:{board}`). The board is
// always cyberpunk, whatever the portal skin. `?focus=` flies to a card.
// User boards dock their board chat on the right (C toggles it): it sees the
// board and the selected cards, and proposes ghosts the user accepts or burns.
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { api, type Chat } from "@/api";
import { useTabsStore } from "@/stores/tabs";
import { useBoardsStore } from "@/stores/boards";
import { useBrainmapStore } from "@/stores/brainmap";
import { brainmapRoute } from "@/router";
import { notifyError } from "@/lib/errors";
import { loadJson, saveJson } from "@/lib/storage";
import BrainBoard from "@/components/brainmap/BrainBoard.vue";
import ChatPanel from "@/panels/ChatPanel.vue";

const props = withDefaults(defineProps<{ board: string; tabId: string; active?: boolean }>(), { active: true });
const tabs = useTabsStore();
const boards = useBoardsStore();
const store = useBrainmapStore(props.board);

const focus = computed(() => (brainmapRoute.board === props.board ? brainmapRoute.focus : ""));
const custom = computed(() => props.board !== "items");

function setTitle(title: string): void {
  tabs.setStatus(props.tabId, { title });
}

// ── docked board chat ─────────────────────────────────────────────────────
const saved = loadJson<{ open: boolean; width: number }>(`brainmap.${props.board}.chat`, { open: true, width: 400 });
const chatOpen = ref(saved.open);
const chatWidth = ref(Math.min(Math.max(saved.width, 300), 720));
const chat = ref<Chat | null>(null);
const ensuring = ref(false);
const firstPrompt = ref("");

watch([chatOpen, chatWidth], () =>
  saveJson(`brainmap.${props.board}.chat`, { open: chatOpen.value, width: chatWidth.value }),
);

async function ensureChat(): Promise<void> {
  if (chat.value || ensuring.value || !custom.value) return;
  ensuring.value = true;
  try {
    chat.value = await api().boardChat(props.board);
    firstPrompt.value = boards.takePrompt(props.board);
  } catch (e) {
    notifyError(e, "Could not open the board chat");
  } finally {
    ensuring.value = false;
  }
}

watch(
  () => chatOpen.value && custom.value && store.loaded,
  (on) => {
    if (on) void ensureChat();
  },
  { immediate: true },
);

function onKey(ev: KeyboardEvent): void {
  if (!props.active || !custom.value) return;
  const t = ev.target as HTMLElement;
  if (t.closest("input, textarea, [contenteditable=true]")) return;
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  if (ev.key === "c") {
    chatOpen.value = !chatOpen.value;
    ev.preventDefault();
  }
}
onMounted(() => window.addEventListener("keydown", onKey));
onBeforeUnmount(() => window.removeEventListener("keydown", onKey));

let dragStart: { x: number; w: number } | null = null;
function startResize(ev: PointerEvent): void {
  dragStart = { x: ev.clientX, w: chatWidth.value };
  (ev.currentTarget as HTMLElement).setPointerCapture(ev.pointerId);
}
function moveResize(ev: PointerEvent): void {
  if (!dragStart) return;
  chatWidth.value = Math.min(720, Math.max(300, dragStart.w - (ev.clientX - dragStart.x)));
}
function endResize(): void {
  dragStart = null;
}

const selectedTitles = computed(() =>
  [...store.selection].map((id) => store.nodeById.get(id)?.title ?? id).filter(Boolean),
);
</script>

<template>
  <div class="brainmap-panel" :class="{ 'with-chat': custom && chatOpen }">
    <div class="board-area">
      <BrainBoard :board="board" :focus="focus" :active="active" @title="setTitle" />
      <q-btn
        v-if="custom"
        class="chat-toggle"
        dense
        unelevated
        no-caps
        :icon="chatOpen ? 'chevron_right' : 'forum'"
        :label="chatOpen ? '' : 'Board chat'"
        @click="chatOpen = !chatOpen"
      >
        <q-tooltip>{{ chatOpen ? "Hide the board chat (C)" : "Talk about this board (C)" }}</q-tooltip>
      </q-btn>
    </div>
    <template v-if="custom && chatOpen">
      <div class="resizer" @pointerdown="startResize" @pointermove="moveResize" @pointerup="endResize" />
      <aside class="board-chat" :style="{ width: `${chatWidth}px` }">
        <div class="sel-strip">
          <template v-if="selectedTitles.length">
            <q-icon name="select_all" size="14px" />
            Asking about {{ selectedTitles.length }} card{{ selectedTitles.length > 1 ? "s" : "" }}:
            <span class="ellipsis-1">{{ selectedTitles.slice(0, 3).join(" · ") }}{{ selectedTitles.length > 3 ? " …" : "" }}</span>
          </template>
          <template v-else>Click or shift-click cards to ask about them; the chat sees the whole board.</template>
        </div>
        <ChatPanel v-if="chat" :id="chat.id" :board="board" :initial-prompt="firstPrompt" class="chat-body" />
        <div v-else class="col flex flex-center"><q-spinner size="28px" /></div>
      </aside>
    </template>
  </div>
</template>

<style scoped>
.brainmap-panel {
  position: absolute;
  inset: 0;
  display: flex;
  background: #050403;
}
.board-area {
  position: relative;
  flex: 1;
  min-width: 0;
}
.chat-toggle {
  position: absolute;
  right: 12px;
  bottom: 12px;
  z-index: 6;
  color: #5ce1ff;
  background: rgba(5, 8, 12, 0.9);
  border: 1px solid rgba(92, 225, 255, 0.4);
}
.resizer {
  width: 5px;
  cursor: col-resize;
  background: rgba(92, 225, 255, 0.12);
}
.resizer:hover {
  background: rgba(92, 225, 255, 0.35);
}
.board-chat {
  display: flex;
  flex-direction: column;
  min-width: 300px;
  background: var(--pawn-bg, #fff);
  color: var(--pawn-fg, inherit);
}
.chat-body {
  flex: 1 1 0;
  min-height: 0;
  height: auto !important;
}
.sel-strip {
  display: flex;
  gap: 6px;
  align-items: center;
  padding: 6px 12px;
  font-size: 12px;
  opacity: 0.8;
  border-bottom: 1px solid var(--pawn-border, rgba(0, 0, 0, 0.1));
}
</style>
