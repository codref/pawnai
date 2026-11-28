<script setup lang="ts">
// One user board in the sidebar: open it, rename, tag, pin, file, archive, delete.
import { computed } from "vue";
import type { Board, FolderNode } from "@/api";
import { useBoardsStore } from "@/stores/boards";
import { useChatsStore } from "@/stores/chats";
import { useTabsStore } from "@/stores/tabs";
import { closeRef, openBrainmap } from "@/router";
import { confirmAction, promptTags, promptText } from "@/lib/dialogs";
import { relativeTime } from "@/lib/time";

const props = defineProps<{ board: Board; dense?: boolean }>();
const boards = useBoardsStore();
const chats = useChatsStore();
const tabs = useTabsStore();

const ICON: Record<string, string> = { manual: "dashboard", lens: "filter_alt", question: "bedtime", items: "hub" };
const icon = computed(() => ICON[props.board.kind] ?? "dashboard");
const kindLabel = computed(
  () => ({ manual: "Board", lens: "Lens", question: "Question, worked overnight", items: "Items" })[props.board.kind],
);
const flatFolders = computed(() => {
  const out: FolderNode[] = [];
  const walk = (list: FolderNode[]) => list.forEach((f) => (out.push(f), walk(f.children)));
  walk(chats.folders);
  return out;
});
const active = computed(() => tabs.active?.kind === "brainmap" && tabs.active.ref === props.board.name);

async function rename(): Promise<void> {
  const title = await promptText("Rename board", props.board.title);
  if (title?.trim()) await boards.patch(props.board.name, { title: title.trim() });
}

async function editTags(): Promise<void> {
  const tags = await promptTags(props.board.tags);
  if (tags) await boards.patch(props.board.name, { tags });
}

async function remove(): Promise<void> {
  const ok = await confirmAction(
    "Delete board",
    `Delete “${props.board.title}”? Its cards stay where they came from; its chat becomes an ordinary chat.`,
    "Delete",
  );
  if (ok && (await boards.remove(props.board.name))) closeRef("brainmap", props.board.name);
}
</script>

<template>
  <div class="hit board-row" :class="{ active, dense }" @click="openBrainmap(board.name)">
    <q-icon :name="icon" class="text-muted hit-icon" />
    <span class="col ellipsis-1">{{ board.title }}</span>
    <q-icon v-if="board.pinned" name="push_pin" size="12px" class="text-muted q-mr-xs" />
    <span v-if="board.updated_at" class="age">{{ relativeTime(board.updated_at) }}</span>
    <q-tooltip :delay="600">{{ kindLabel }}<template v-if="board.tags.length"> · #{{ board.tags.join(" #") }}</template></q-tooltip>
    <q-menu context-menu touch-position auto-close>
      <q-list dense style="min-width: 180px">
        <q-item clickable @click="openBrainmap(board.name)"><q-item-section>Open</q-item-section></q-item>
        <q-item clickable @click="rename"><q-item-section>Rename…</q-item-section></q-item>
        <q-item clickable @click="editTags"><q-item-section>Edit tags…</q-item-section></q-item>
        <q-item clickable @click="boards.patch(board.name, { pinned: !board.pinned })">
          <q-item-section>{{ board.pinned ? "Unpin" : "Pin" }}</q-item-section>
        </q-item>
        <q-item clickable>
          <q-item-section>Move to folder</q-item-section>
          <q-item-section side><q-icon name="chevron_right" /></q-item-section>
          <q-menu anchor="top end" self="top start" auto-close>
            <q-list dense style="min-width: 160px">
              <q-item v-if="board.folder_id" clickable @click="boards.patch(board.name, { unfile: true })">
                <q-item-section>Unfiled</q-item-section>
              </q-item>
              <q-item
                v-for="f in flatFolders"
                :key="f.id"
                clickable
                @click="boards.patch(board.name, { folder_id: f.id })"
              >
                <q-item-section>{{ f.name }}</q-item-section>
              </q-item>
            </q-list>
          </q-menu>
        </q-item>
        <q-item clickable @click="boards.patch(board.name, { archived: !board.archived })">
          <q-item-section>{{ board.archived ? "Unarchive" : "Archive" }}</q-item-section>
        </q-item>
        <q-separator />
        <q-item clickable class="text-negative" @click="remove"><q-item-section>Delete…</q-item-section></q-item>
      </q-list>
    </q-menu>
  </div>
</template>

<style scoped>
.board-row.active {
  background: var(--pawn-selected, rgba(0, 0, 0, 0.06));
}
</style>
