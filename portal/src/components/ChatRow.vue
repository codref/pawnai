<script setup lang="ts">
import { computed, ref } from "vue";
import type { Chat } from "@/api";
import { isSideChat, useChatsStore } from "@/stores/chats";
import { useTabsStore } from "@/stores/tabs";
import { closeRef, openChat } from "@/router";
import { confirmAction, promptTags, promptText } from "@/lib/dialogs";
import { DND_CHAT } from "@/lib/dnd";
import { fullTime, relativeTime } from "@/lib/time";
import { loadJson, saveJson } from "@/lib/storage";
import { vTarget, type LocalAction, type Target } from "@/lib/targets";

defineOptions({ name: "ChatRow" });

/** `compact`: a tree row (TreeList owns padding, hover and selection).
 * `depth`: nesting level for side chats shown under their parent. */
const props = defineProps<{ chat: Chat; dense?: boolean; compact?: boolean; depth?: number }>();
const chats = useChatsStore();
const tabs = useTabsStore();

const active = computed(() => tabs.active?.kind === "chat" && tabs.active.ref === props.chat.id);
const side = computed(() => isSideChat(props.chat));
const children = computed(() => (props.depth ?? 0) < 4 ? (chats.sideChildren.get(props.chat.id) ?? []) : []);

// Expanded parents (side chats nested under them), remembered per browser.
const OPEN_KEY = "chat.sideOpen";
const openSet = ref<string[]>(loadJson<string[]>(OPEN_KEY, []));
const expanded = computed(() => openSet.value.includes(props.chat.id));
function toggle(): void {
  const cur = loadJson<string[]>(OPEN_KEY, []);
  const next = cur.includes(props.chat.id) ? cur.filter((x) => x !== props.chat.id) : [...cur, props.chat.id];
  saveJson(OPEN_KEY, next);
  openSet.value = next;
}

const icon = computed(() => {
  if (props.chat.pinned) return "push_pin";
  if (side.value) return props.chat.origin?.kind === "file" ? "find_in_page" : "question_answer";
  return "chat_bubble_outline";
});

const local = computed<LocalAction[]>(() => [
  { label: "Rename…", icon: "edit", run: rename },
  { label: "Edit tags…", icon: "sell", run: editTags },
  {
    label: props.chat.pinned ? "Unpin" : "Pin",
    icon: "push_pin",
    run: () => void chats.patchChat(props.chat.id, { pinned: !props.chat.pinned }),
  },
  ...(side.value
    ? [{ label: "Promote to a normal chat", icon: "upgrade", run: () => void chats.patchChat(props.chat.id, { promoted: true }) }]
    : []),
  ...(props.chat.folder_id
    ? [{ label: "Move to Unfiled", icon: "inbox", run: () => void chats.moveChat(props.chat.id, null) }]
    : []),
  {
    label: props.chat.archived ? "Unarchive" : "Archive",
    icon: "inventory_2",
    run: () => void chats.patchChat(props.chat.id, { archived: !props.chat.archived }),
  },
  { label: "Delete…", icon: "delete", danger: true, run: remove },
]);

const target = computed<Target>(() => ({
  kind: "chat",
  ref: props.chat.id,
  label: props.chat.title || "Untitled chat",
  local: local.value,
}));
const stamp = computed(() => props.chat.last_message_at ?? props.chat.updated_at ?? props.chat.created_at);
const when = computed(() => relativeTime(stamp.value));

function onDragStart(ev: DragEvent): void {
  ev.dataTransfer?.setData(DND_CHAT, props.chat.id);
  ev.dataTransfer?.setData("text/plain", props.chat.title || props.chat.id);
  if (ev.dataTransfer) ev.dataTransfer.effectAllowed = "move";
}

async function rename(): Promise<void> {
  const title = await promptText("Rename chat", props.chat.title);
  if (title !== null) await chats.patchChat(props.chat.id, { title });
}

async function editTags(): Promise<void> {
  const tags = await promptTags(props.chat.tags);
  if (tags !== null) await chats.patchChat(props.chat.id, { tags });
}

async function remove(): Promise<void> {
  const ok = await confirmAction(
    "Delete chat",
    `Delete “${props.chat.title || "Untitled"}” and its messages? This also resets Pawn's memory of it.`,
  );
  if (!ok) return;
  if (await chats.deleteChat(props.chat.id)) closeRef("chat", props.chat.id);
}
</script>

<template>
  <div
    v-if="compact"
    v-target="target"
    class="row items-center no-wrap full-width chat-compact"
    draggable="true"
    @dragstart="onDragStart"
    @click="openChat(chat.id)"
    @dblclick="openChat(chat.id, { pin: true })"
  >
    <q-icon :name="icon" size="15px" class="q-mr-xs text-muted" />
    <span class="col ellipsis-1">{{ chat.title || "Untitled chat" }}</span>
    <span v-for="t in chat.tags.slice(0, 1)" :key="t" class="tag-chip ellipsis-1 q-ml-xs" style="max-width: 90px">
      #{{ t }}
    </span>
    <span v-if="children.length" class="side-count" @click.stop="toggle">
      <q-icon name="question_answer" size="12px" /> {{ children.length }}
      <q-tooltip>{{ children.length }} side chat(s)</q-tooltip>
    </span>
    <span class="age">
      {{ when }}
      <q-tooltip :delay="600">{{ fullTime(stamp) }}</q-tooltip>
    </span>
  </div>
  <template v-else>
    <q-item
      v-target="target"
      clickable
      :dense="dense"
      :active="active"
      active-class="chat-active"
      class="chat-row q-pr-sm"
      :style="depth ? { paddingLeft: `${8 + depth * 16}px` } : undefined"
      draggable="true"
      @dragstart="onDragStart"
      @click="openChat(chat.id)"
      @dblclick="openChat(chat.id, { pin: true })"
    >
      <q-item-section v-if="children.length" side class="side-toggle" @click.stop="toggle">
        <q-icon :name="expanded ? 'expand_more' : 'chevron_right'" size="16px" />
      </q-item-section>
      <q-item-section>
        <q-item-label class="ellipsis-1">
          <q-icon v-if="chat.pinned || side" :name="icon" size="14px" class="q-mr-xs text-muted" />
          {{ chat.title || (side && chat.origin?.quote ? `“${chat.origin.quote}”` : "Untitled chat") }}
        </q-item-label>
        <q-item-label v-if="chat.tags.length" caption class="ellipsis-1">
          <span v-for="t in chat.tags" :key="t" class="tag-chip">#{{ t }}</span>
        </q-item-label>
      </q-item-section>
      <q-item-section side class="age-side">
        <span class="age">
          <span v-if="children.length && !expanded" class="side-count q-mr-xs">
            <q-icon name="question_answer" size="12px" /> {{ children.length }}
          </span>
          {{ when }}
          <q-tooltip :delay="600">{{ fullTime(stamp) }}</q-tooltip>
        </span>
      </q-item-section>
    </q-item>
    <template v-if="expanded">
      <ChatRow v-for="c in children" :key="c.id" :chat="c" :dense="dense" :depth="(depth ?? 0) + 1" />
    </template>
  </template>
</template>

<style scoped>
.chat-row {
  min-height: 34px;
  border-radius: 6px;
}
.chat-row.chat-active {
  background: var(--pawn-active);
  color: var(--q-primary);
}
.age-side {
  padding-left: 4px;
}
.chat-compact {
  min-height: 28px;
}
.side-toggle {
  min-width: 0;
  padding-right: 2px;
  cursor: pointer;
}
.side-count {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  font-size: 0.7rem;
  color: var(--pawn-muted);
  margin-left: 4px;
  cursor: pointer;
}
</style>
