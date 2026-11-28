<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { Annotation, Board, Chat, ChatSort, Clip, FileHit, FolderNode, SearchResults, TagNode } from "@/api";
import { api } from "@/api";
import { useAnnotationsStore } from "@/stores/annotations";
import { useChatsStore, type ChatView } from "@/stores/chats";
import { useFilesStore } from "@/stores/files";
import { useTabsStore } from "@/stores/tabs";
import { closeRef, openChat, openClip, openFile, openSession } from "@/router";
import { confirmAction, promptText } from "@/lib/dialogs";
import { DND_CHAT, DND_FILE, DND_FOLDER, hasType, leftElement } from "@/lib/dnd";
import { ICONS } from "@/lib/icons";
import type { SegmentOption } from "@/lib/segment";
import { useComposerStore } from "@/stores/composer";
import { notifyError } from "@/lib/errors";
import { loadJson, saveJson } from "@/lib/storage";
import { fullTime, relativeTime } from "@/lib/time";
import { colorOf } from "@/lib/highlights";
import type { TreeItem } from "@/lib/tree";
import { vTarget, type Target } from "@/lib/targets";
import { useBoardsStore } from "@/stores/boards";
import BoardRow from "./BoardRow.vue";
import NewBoardDialog from "./NewBoardDialog.vue";
import ChatRow from "./ChatRow.vue";
import SegmentedControl from "./SegmentedControl.vue";
import TreeList from "./TreeList.vue";

const chats = useChatsStore();
const files = useFilesStore();
const tabs = useTabsStore();
const boards = useBoardsStore();

// ── boards ───────────────────────────────────────────────────────────────
const boardsOpen = ref(loadJson<boolean>("chat.boardsOpen", true));
watch(boardsOpen, (v) => saveJson("chat.boardsOpen", v));
const newBoard = ref<{ folderId: string | null; tags: string[] } | null>(null);
function openNewBoard(folderId: string | null = null, tags: string[] = []): void {
  newBoard.value = { folderId, tags };
}
onMounted(() => void boards.refresh());

// ── search ───────────────────────────────────────────────────────────────
const query = ref("");
const results = ref<SearchResults | null>(null);
const searching = ref(false);
let searchTimer: ReturnType<typeof setTimeout> | undefined;
let searchSeq = 0;

watch(query, (q) => {
  clearTimeout(searchTimer);
  if (!q.trim()) {
    results.value = null;
    return;
  }
  searchTimer = setTimeout(async () => {
    const seq = ++searchSeq;
    searching.value = true;
    try {
      const res = await api().search(q.trim());
      if (seq === searchSeq) results.value = res;
    } catch (e) {
      notifyError(e, "Search failed");
    } finally {
      if (seq === searchSeq) searching.value = false;
    }
  }, 250);
});
// Ctrl/Cmd+K focuses search from anywhere.
const searchInput = ref<{ focus: () => void; select?: () => void } | null>(null);
function onGlobalKey(ev: KeyboardEvent): void {
  if ((ev.ctrlKey || ev.metaKey) && !ev.altKey && ev.key.toLowerCase() === "k") {
    ev.preventDefault();
    searchInput.value?.focus();
    searchInput.value?.select?.();
  }
}
onMounted(() => window.addEventListener("keydown", onGlobalKey));
onBeforeUnmount(() => {
  clearTimeout(searchTimer);
  window.removeEventListener("keydown", onGlobalKey);
});

function searchTag(path: string): void {
  query.value = `#${path} `;
  searchInput.value?.focus();
}

const FILE_ICONS: Record<string, string> = {
  markdown: ICONS.markdown,
  pdf: ICONS.pdf,
  docx: ICONS.doc,
  image: ICONS.image,
  audio: "audiotrack",
};
function fileHitIcon(kind: string): string {
  return FILE_ICONS[kind] ?? "insert_drive_file";
}

function dirOf(key: string): string {
  const i = key.lastIndexOf("/");
  return i > 0 ? key.slice(0, i) : "";
}

function openHit(hit: FileHit, pin = false): void {
  files.markSeen(hit.key);
  openFile(hit.key, { pin });
}

/** Open what an annotation is on and scroll to its highlight. */
function openAnnotation(a: Annotation): void {
  const annotations = useAnnotationsStore();
  if (a.target_kind === "file") openFile(a.target_ref);
  else if (a.target_kind === "clip") openClip(a.target_ref);
  else if (a.target_kind === "message" && a.context?.chat_id) {
    openChat(a.context.chat_id);
    chats.requestFocus(a.context.chat_id, a.target_ref);
  } else if (a.side_chat_id) {
    openChat(a.side_chat_id);
    return;
  } else {
    notifyError(new Error("Its target is gone"), "Cannot open annotation");
    return;
  }
  annotations.requestFocus(a.id);
}

function annotationWhere(a: Annotation): string {
  const label = a.context?.label || a.target_ref.split("/").pop() || a.target_ref;
  const kind = a.target_kind === "message" ? "chat" : a.target_kind;
  return `${kind} · ${label}`;
}

const resultCount = computed(() =>
  results.value
    ? results.value.chats.length +
      results.value.messages.length +
      (results.value.files?.length ?? 0) +
      results.value.clips.length +
      results.value.annotations.length +
      (results.value.sessions?.length ?? 0)
    : 0,
);

// ── create ───────────────────────────────────────────────────────────────
async function newChat(opts: { folder_id?: string | null; tags?: string[] } = {}): Promise<void> {
  const chat = await chats.createChat(opts);
  if (chat) openChat(chat.id, { pin: true });
}

async function newFolder(parentId: string | null = null): Promise<void> {
  const name = await promptText(parentId ? "New subfolder" : "New folder");
  if (name) await chats.createFolder(name, parentId);
  if (parentId) expandKey(`f:${parentId}`);
}

// ── view + sort ──────────────────────────────────────────────────────────
const views: SegmentOption<ChatView>[] = [
  { value: "folders", label: "Folders", icon: ICONS.folder },
  { value: "tags", label: "Tags", icon: ICONS.tag },
  { value: "all", label: "All", icon: ICONS.list },
];
const sortOptions: { value: ChatSort; label: string }[] = [
  { value: "updated", label: "Last updated" },
  { value: "created", label: "Created" },
  { value: "name", label: "Name" },
  { value: "tag", label: "Tag" },
];
const sortLabel = computed(
  () => sortOptions.find((o) => o.value === chats.sort.sort)?.label ?? chats.sort.sort,
);

// Pinned chats always show on top, so the views list the rest.
// Side chats nest under their parent row, so the flat lists skip them.
const unpinned = computed(() => chats.chats.filter((c) => !c.pinned && !chats.nested(c)));
const pinnedTop = computed(() => chats.pinned.filter((c) => !chats.nested(c)));

// ── folders view (QTree) ─────────────────────────────────────────────────
interface TreeNode extends TreeItem {
  kind: "folder" | "chat" | "unfiled" | "board";
  label: string;
  folder?: FolderNode;
  chat?: Chat;
  board?: Board;
  count?: number;
  children?: TreeNode[];
}

const expanded = ref<string[]>(loadJson<string[]>("chat.expanded", ["unfiled"]));
watch(expanded, (v) => saveJson("chat.expanded", v));
function expandKey(key: string): void {
  if (!expanded.value.includes(key)) expanded.value = [...expanded.value, key];
}

const folderNodes = computed<TreeNode[]>(() => {
  // Pinned chats also stay in their folder, so folder counts match.
  const byFolder = new Map<string | null, Chat[]>();
  for (const c of chats.chats) {
    if (chats.nested(c)) continue;
    const k = c.folder_id ?? null;
    byFolder.set(k, [...(byFolder.get(k) ?? []), c]);
  }
  const chatNode = (c: Chat, depth = 0): TreeNode => {
    const kids = depth < 4 ? (chats.sideChildren.get(c.id) ?? []) : [];
    return {
      key: `c:${c.id}`,
      kind: "chat",
      label: c.title,
      chat: c,
      expandable: kids.length > 0,
      children: kids.map((k) => chatNode(k, depth + 1)),
    };
  };
  // Boards filed in a folder sit there too; unfiled ones live in the Boards section.
  const boardsIn = new Map<string, Board[]>();
  for (const b of boards.userBoards) {
    if (b.folder_id) boardsIn.set(b.folder_id, [...(boardsIn.get(b.folder_id) ?? []), b]);
  }
  const boardNode = (b: Board): TreeNode => ({ key: `b:${b.name}`, kind: "board", label: b.title, board: b });
  const build = (f: FolderNode): TreeNode => ({
    key: `f:${f.id}`,
    kind: "folder",
    label: f.name,
    folder: f,
    count: f.chat_count + (boardsIn.get(f.id)?.length ?? 0),
    expandable: true,
    children: [
      ...f.children.map(build),
      ...(boardsIn.get(f.id) ?? []).map(boardNode),
      ...(byFolder.get(f.id) ?? []).map((c) => chatNode(c)),
    ],
  });
  return [
    ...chats.folders.map(build),
    {
      key: "unfiled",
      kind: "unfiled",
      label: "Unfiled",
      count: (byFolder.get(null) ?? []).length,
      expandable: true,
      children: (byFolder.get(null) ?? []).map((c) => chatNode(c)),
    },
  ];
});

function folderContains(root: FolderNode, id: string): boolean {
  return root.id === id || root.children.some((c) => folderContains(c, id));
}
function findFolder(nodes: FolderNode[], id: string): FolderNode | null {
  for (const n of nodes) {
    if (n.id === id) return n;
    const hit = findFolder(n.children, id);
    if (hit) return hit;
  }
  return null;
}

function onFolderDragStart(ev: DragEvent, f: FolderNode): void {
  ev.dataTransfer?.setData(DND_FOLDER, f.id);
  ev.dataTransfer?.setData("text/plain", f.name);
  if (ev.dataTransfer) ev.dataTransfer.effectAllowed = "move";
}

/** The folder a tree row stands for when something is dropped on it. */
function folderOfNode(node: TreeNode): string | null {
  if (node.kind === "folder") return node.folder?.id ?? null;
  if (node.kind === "chat") return node.chat?.folder_id ?? null;
  return null;
}

function acceptsChatDrag(ev: DragEvent): boolean {
  return hasType(ev, DND_CHAT) || hasType(ev, DND_FOLDER) || hasType(ev, DND_FILE);
}

function onRowDragOver(ev: DragEvent, node: TreeNode): void {
  if (!acceptsChatDrag(ev)) return;
  // A file only lands on a folder (or Unfiled): it starts a chat there.
  if (hasType(ev, DND_FILE) && node.kind === "chat") return;
  ev.preventDefault();
  if (ev.dataTransfer) ev.dataTransfer.dropEffect = hasType(ev, DND_FILE) ? "copy" : "move";
}

function onRowDrop(ev: DragEvent, node: TreeNode): void {
  void dropInto(ev, folderOfNode(node));
}

/** Drop a chat, a folder or a vault file into a folder (targetId) or Unfiled/root (null). */
async function dropInto(ev: DragEvent, targetId: string | null): Promise<void> {
  ev.preventDefault();
  ev.stopPropagation();
  rootOver.value = false;
  const chatId = ev.dataTransfer?.getData(DND_CHAT);
  const folderId = ev.dataTransfer?.getData(DND_FOLDER);
  const filePath = ev.dataTransfer?.getData(DND_FILE);
  if (chatId) {
    const chat = chats.byId.get(chatId);
    if (chat && (chat.folder_id ?? null) === targetId) return;
    await chats.moveChat(chatId, targetId);
    expandKey(targetId ? `f:${targetId}` : "unfiled");
  } else if (folderId) {
    if (folderId === targetId) return;
    const moving = findFolder(chats.folders, folderId);
    if (moving && targetId && folderContains(moving, targetId)) {
      notifyError(new Error("A folder can't move into itself"));
      return;
    }
    if (moving && (moving.parent_id ?? null) === targetId) return;
    await chats.moveFolder(folderId, targetId);
    if (targetId) expandKey(`f:${targetId}`);
  } else if (filePath) {
    const chat = await chats.createChat({ folder_id: targetId });
    if (!chat) return;
    useComposerStore().addNote(chat.id, filePath);
    openChat(chat.id, { pin: true });
  }
}

// Empty space below the folder tree is the top level: chats go to Unfiled,
// folders to the root.
const rootOver = ref(false);
function onRootDragOver(ev: DragEvent): void {
  if (ev.defaultPrevented || chats.view !== "folders" || results.value || !acceptsChatDrag(ev)) {
    rootOver.value = false;
    return;
  }
  ev.preventDefault();
  if (ev.dataTransfer) ev.dataTransfer.dropEffect = hasType(ev, DND_FILE) ? "copy" : "move";
  rootOver.value = true;
}
function onRootDragLeave(ev: DragEvent): void {
  if (leftElement(ev)) rootOver.value = false;
}
function onRootDrop(ev: DragEvent): void {
  if (!rootOver.value) return;
  void dropInto(ev, null);
}

async function renameFolder(f: FolderNode): Promise<void> {
  const name = await promptText("Rename folder", f.name);
  if (name) await chats.renameFolder(f.id, name);
}

async function deleteFolder(f: FolderNode): Promise<void> {
  const ok = await confirmAction(
    "Delete folder",
    `Delete “${f.name}”? Its chats and subfolders move to the parent folder; no chat is deleted.`,
  );
  if (ok) await chats.deleteFolder(f.id);
}

// ── tags view ────────────────────────────────────────────────────────────
const selectedChatKey = computed(() => (tabs.active?.kind === "chat" ? `c:${tabs.active.ref}` : null));

interface TagTreeNode extends TreeItem {
  label: string;
  path: string;
  count: number;
  children: TagTreeNode[];
}
const tagNodes = computed<TagTreeNode[]>(() => {
  const map = (n: TagNode): TagTreeNode => ({
    key: n.path,
    label: n.name,
    path: n.path,
    count: n.count,
    children: n.children.map(map),
  });
  return chats.allTags.map(map);
});
const selectedTag = ref<string>("");
function selectTag(node: TagTreeNode): void {
  selectedTag.value = selectedTag.value === node.path ? "" : node.path;
}

// Files carrying the selected tag (frontmatter or stored), via search.
const tagFiles = ref<FileHit[]>([]);
watch([selectedTag, () => chats.allTags], async ([tag]) => {
  tagFiles.value = [];
  if (!tag) return;
  try {
    const res = await api().search(`#${tag}`, 100);
    if (selectedTag.value === tag) tagFiles.value = res.files ?? [];
  } catch (e) {
    notifyError(e, "Could not load tagged files");
  }
});
function onTagDragOver(ev: DragEvent): void {
  if (!hasType(ev, DND_CHAT)) return;
  ev.preventDefault();
  if (ev.dataTransfer) ev.dataTransfer.dropEffect = "link";
}
async function onTagDrop(ev: DragEvent, node: TagTreeNode): Promise<void> {
  const chatId = ev.dataTransfer?.getData(DND_CHAT);
  if (!chatId) return;
  ev.preventDefault();
  const chat = chats.byId.get(chatId);
  if (!chat || chat.tags.includes(node.path)) return;
  await chats.patchChat(chatId, { tags: [...chat.tags, node.path] });
}

const tagExpanded = ref<string[]>(loadJson<string[]>("chat.tagExpanded", []));
watch(tagExpanded, (v) => saveJson("chat.tagExpanded", v));

function tagMatches(tags: string[], prefix: string): boolean {
  return tags.some((t) => t === prefix || t.startsWith(prefix + "/"));
}
const tagBoards = computed(() =>
  selectedTag.value ? boards.userBoards.filter((b) => tagMatches(b.tags, selectedTag.value)) : [],
);
const tagChats = computed(() =>
  selectedTag.value
    ? chats.chats.filter((c) => tagMatches(c.tags, selectedTag.value) && !chats.nested(c))
    : unpinned.value.filter((c) => !c.tags.length),
);

async function renameTag(path: string): Promise<void> {
  const to = await promptText("Rename tag", path, {
    message: "Renames this tag and every tag under it, on chats, clips and annotations.",
  });
  if (to && to !== path) {
    await chats.renameTag(path, to);
    if (selectedTag.value === path || selectedTag.value.startsWith(path + "/"))
      selectedTag.value = to + selectedTag.value.slice(path.length);
  }
}

// ── archived + clips ─────────────────────────────────────────────────────
const archivedOpen = ref(false);
watch(archivedOpen, (open) => {
  if (open) chats.loadArchived().catch((e) => notifyError(e, "Could not load archived chats"));
});

function openClipRow(clip: Clip, pin = false): void {
  openClip(clip.id, { pin });
}

function clipTarget(c: Clip): Target {
  return {
    kind: "clip",
    ref: c.id,
    label: c.title || "Clip",
    text: c.content_md,
    local: [
      ...(c.vault_key
        ? [{ label: "Open saved note", icon: "description", run: () => openFile(c.vault_key!, { pin: true }) }]
        : []),
      ...(c.chat_id ? [{ label: "Open source chat", icon: "forum", run: () => openChat(c.chat_id!) }] : []),
      { label: "Delete clip…", icon: "delete", danger: true, run: () => deleteClip(c) },
    ],
  };
}

async function deleteClip(c: Clip): Promise<void> {
  const ok = await confirmAction("Delete clip", `Delete “${c.title || "Clip"}”? A note saved to the scratchpad stays.`);
  if (!ok) return;
  try {
    await api().deleteClip(c.id);
    closeRef("clip", c.id);
    await chats.refreshClips();
  } catch (e) {
    notifyError(e, "Could not delete clip");
  }
}
</script>

<template>
  <div class="column no-wrap full-height chat-sidebar sidebar-root">
    <div class="q-pa-sm border-b">
      <q-input
        ref="searchInput"
        v-model="query"
        dense
        outlined
        clearable
        placeholder="Search everything"
        debounce="0"
        class="search-box"
      >
        <template #prepend><q-icon name="search" /></template>
        <template #append>
          <q-spinner v-if="searching" size="16px" />
          <span v-else-if="!query" class="kbd">Ctrl K</span>
        </template>
      </q-input>
      <div v-if="query" class="text-caption text-muted q-mt-xs q-px-xs">
        Chats, messages, files (incl. PDFs, docs, images), clips. Filter with <code>#tag</code>.
      </div>
      <div v-else class="row q-gutter-xs q-mt-xs">
        <q-btn class="col" unelevated dense color="primary" icon="add" label="New chat" no-caps @click="newChat()" />
        <q-btn dense flat :icon="ICONS.newFolder" aria-label="New folder" class="head-btn" @click="newFolder()">
          <q-tooltip>New folder</q-tooltip>
        </q-btn>
      </div>
    </div>

    <div
      class="col pawn-scroll relative-position"
      :class="{ 'root-over': rootOver }"
      @dragover="onRootDragOver"
      @dragleave="onRootDragLeave"
      @drop="onRootDrop"
    >
      <!-- Search results replace the views while a query is typed. -->
      <div v-if="results" class="q-pa-xs search-results">
        <div v-if="results.tags?.length" class="q-px-sm q-py-xs">
          <span v-for="t in results.tags" :key="t" class="tag-chip">#{{ t }}</span>
        </div>
        <div v-if="!resultCount" class="text-caption text-muted q-pa-md text-center">Nothing matches.</div>

        <template v-if="results.chats.length">
          <div class="section-label">Chats</div>
          <ChatRow v-for="c in results.chats" :key="c.id" :chat="c" dense />
        </template>

        <template v-if="results.files?.length">
          <div class="section-label">Files</div>
          <div
            v-for="f in results.files"
            :key="f.key"
            class="hit"
            :class="{ active: tabs.active?.kind === 'file' && tabs.active.ref === f.key }"
            @click="openHit(f)"
            @dblclick="openHit(f, true)"
          >
            <q-icon :name="fileHitIcon(f.kind)" class="text-muted hit-icon" />
            <div class="col" style="min-width: 0">
              <div class="row no-wrap items-center">
                <span class="col ellipsis-1">{{ f.title }}</span>
                <span v-if="f.modified" class="age">
                  {{ relativeTime(f.modified) }}
                  <q-tooltip :delay="600">{{ fullTime(f.modified) }}</q-tooltip>
                </span>
              </div>
              <div class="text-caption text-muted ellipsis-1">{{ dirOf(f.key) || "/" }}</div>
              <div v-if="f.snippet" class="text-caption snippet">{{ f.snippet }}</div>
              <div v-if="f.tags.length">
                <span v-for="t in f.tags" :key="t" class="tag-chip">#{{ t }}</span>
              </div>
            </div>
          </div>
        </template>

        <template v-if="results.sessions?.length">
          <div class="section-label">Sessions</div>
          <div v-for="s in results.sessions" :key="s.session_id" class="hit" @click="openSession(s.session_id)">
            <q-icon name="graphic_eq" class="text-muted hit-icon" />
            <div class="col" style="min-width: 0">
              <div class="row no-wrap items-center">
                <span class="col ellipsis-1">{{ s.title }}</span>
                <span v-if="s.created_at" class="age">{{ relativeTime(s.created_at) }}</span>
              </div>
              <div v-if="s.snippet" class="text-caption snippet">{{ s.snippet }}</div>
            </div>
          </div>
        </template>

        <template v-if="results.messages.length">
          <div class="section-label">Messages</div>
          <div v-for="m in results.messages" :key="m.id" class="hit" @click="openChat(m.chat_id)">
            <q-icon :name="m.role === 'user' ? 'person' : 'smart_toy'" class="text-muted hit-icon" />
            <div class="col" style="min-width: 0">
              <div class="row no-wrap items-center">
                <span class="col ellipsis-1">{{ m.chat_title || "Untitled chat" }}</span>
                <span v-if="m.created_at" class="age">{{ relativeTime(m.created_at) }}</span>
              </div>
              <div class="text-caption snippet">{{ m.snippet }}</div>
            </div>
          </div>
        </template>

        <template v-if="results.clips.length">
          <div class="section-label">Clips</div>
          <div v-for="c in results.clips" :key="c.id" v-target="clipTarget(c)" class="hit" @click="openClipRow(c)">
            <q-icon name="content_cut" class="text-muted hit-icon" />
            <span class="col ellipsis-1">{{ c.title || "Clip" }}</span>
            <span v-if="c.created_at" class="age">{{ relativeTime(c.created_at) }}</span>
          </div>
        </template>

        <template v-if="results.annotations.length">
          <div class="section-label">Annotations</div>
          <div v-for="a in results.annotations" :key="a.id" class="hit" @click="openAnnotation(a)">
            <span class="ann-swatch hit-icon ann-hit-swatch" :class="`ann-c-${colorOf(a.color)}`" />
            <div class="col" style="min-width: 0">
              <div v-if="a.anchor?.exact" class="text-caption ellipsis-1 ann-hit-quote">“{{ a.anchor.exact }}”</div>
              <div v-if="a.body_md" class="text-caption snippet ellipsis-1">{{ a.body_md }}</div>
              <div class="text-caption text-muted ellipsis-1">
                <q-icon v-if="a.side_chat_id" name="forum" size="12px" />
                {{ annotationWhere(a) }}
              </div>
            </div>
          </div>
        </template>
      </div>

      <template v-else>
        <div v-if="chats.pinned.length || boards.pinned.length" class="q-pa-xs border-b">
          <div class="section-label"><q-icon name="push_pin" size="13px" />Pinned</div>
          <BoardRow v-for="b in boards.pinned" :key="b.name" :board="b" dense />
          <ChatRow v-for="c in pinnedTop" :key="c.id" :chat="c" dense />
        </div>

        <q-expansion-item
          v-model="boardsOpen"
          dense
          icon="dashboard"
          :label="`Boards${boards.userBoards.length ? ` (${boards.userBoards.length})` : ''}`"
          header-class="drawer-section"
          class="border-b"
        >
          <template #header>
            <q-item-section avatar><q-icon name="dashboard" /></q-item-section>
            <q-item-section>Boards{{ boards.userBoards.length ? ` (${boards.userBoards.length})` : "" }}</q-item-section>
            <q-item-section side>
              <q-btn flat dense round size="sm" icon="add" aria-label="New board" @click.stop="openNewBoard()">
                <q-tooltip>New board</q-tooltip>
              </q-btn>
            </q-item-section>
          </template>
          <div class="q-pa-xs">
            <BoardRow v-if="boards.boards.find((b) => b.kind === 'items')" :board="boards.boards.find((b) => b.kind === 'items')!" dense />
            <BoardRow
              v-for="b in boards.userBoards.filter((x) => !x.folder_id && !x.pinned)"
              :key="b.name"
              :board="b"
              dense
            />
            <div v-if="boards.loaded && !boards.userBoards.length" class="text-caption text-muted q-px-sm q-py-xs">
              Make a board to investigate something, and talk to it.
            </div>
          </div>
        </q-expansion-item>

        <div class="row items-center no-wrap q-px-sm q-py-xs">
          <SegmentedControl
            :model-value="chats.view"
            :options="views"
            size="sm"
            stretch
            aria-label="Chat view"
            class="col"
            @update:model-value="chats.setView"
          />
          <q-btn flat dense round size="sm" :icon="ICONS.sort" class="q-ml-xs" aria-label="Sort">
            <q-tooltip>Sort: {{ sortLabel }} ({{ chats.sort.order }})</q-tooltip>
            <q-menu auto-close>
              <q-list dense style="min-width: 160px">
                <q-item
                  v-for="o in sortOptions"
                  :key="o.value"
                  clickable
                  :active="chats.sort.sort === o.value"
                  @click="chats.setSort({ sort: o.value })"
                >
                  <q-item-section>{{ o.label }}</q-item-section>
                </q-item>
                <q-separator />
                <q-item clickable :active="chats.sort.order === 'asc'" @click="chats.setSort({ order: 'asc' })">
                  <q-item-section avatar><q-icon name="arrow_upward" size="16px" /></q-item-section>
                  <q-item-section>Ascending</q-item-section>
                </q-item>
                <q-item clickable :active="chats.sort.order === 'desc'" @click="chats.setSort({ order: 'desc' })">
                  <q-item-section avatar><q-icon name="arrow_downward" size="16px" /></q-item-section>
                  <q-item-section>Descending</q-item-section>
                </q-item>
              </q-list>
            </q-menu>
          </q-btn>
        </div>

        <!-- Folders -->
        <TreeList
          v-if="chats.view === 'folders'"
          v-model:expanded="expanded"
          :nodes="folderNodes"
          :selected="selectedChatKey"
          empty-label="No folders yet"
          @row-dragover="onRowDragOver"
          @row-drop="onRowDrop"
        >
          <template #default="{ node, open: isOpen }">
            <ChatRow v-if="node.kind === 'chat' && node.chat" :chat="node.chat" compact />
            <BoardRow v-else-if="node.kind === 'board' && node.board" :board="node.board" dense />
            <div
              v-else
              class="row items-center no-wrap full-width folder-header"
              :draggable="node.kind === 'folder'"
              @dragstart="node.folder && onFolderDragStart($event, node.folder)"
            >
              <q-icon
                :name="node.kind === 'unfiled' ? ICONS.inbox : isOpen ? ICONS.folderOpen : ICONS.folder"
                class="row-icon"
                :class="{ 'folder-icon': node.kind !== 'unfiled' }"
              />
              <span class="col ellipsis-1">{{ node.label }}</span>
              <span class="count">{{ node.count || "" }}</span>

              <q-menu context-menu touch-position auto-close>
                <q-list dense style="min-width: 170px">
                  <q-item clickable @click="newChat({ folder_id: node.folder?.id ?? null })">
                    <q-item-section>New chat here</q-item-section>
                  </q-item>
                  <q-item v-if="node.folder" clickable @click="openNewBoard(node.folder.id)">
                    <q-item-section>New board here</q-item-section>
                  </q-item>
                  <template v-if="node.folder">
                    <q-item clickable @click="newFolder(node.folder.id)">
                      <q-item-section>New subfolder…</q-item-section>
                    </q-item>
                    <q-item clickable @click="renameFolder(node.folder)">
                      <q-item-section>Rename…</q-item-section>
                    </q-item>
                    <q-item v-if="node.folder.parent_id" clickable @click="chats.moveFolder(node.folder.id, null)">
                      <q-item-section>Move to top level</q-item-section>
                    </q-item>
                    <q-separator />
                    <q-item clickable class="text-negative" @click="deleteFolder(node.folder)">
                      <q-item-section>Delete…</q-item-section>
                    </q-item>
                  </template>
                </q-list>
              </q-menu>
            </div>
          </template>
        </TreeList>

        <!-- Tags: chats and vault files share one tag space -->
        <template v-else-if="chats.view === 'tags'">
          <TreeList
            v-model:expanded="tagExpanded"
            :nodes="tagNodes"
            :selected="selectedTag || null"
            empty-label="No tags yet. Right-click a chat or file → Edit tags."
            @select="selectTag"
            @row-dragover="onTagDragOver"
            @row-drop="onTagDrop"
          >
            <template #default="{ node }">
              <div class="row items-center no-wrap full-width">
                <q-icon :name="ICONS.tag" class="row-icon" />
                <span class="col ellipsis-1">{{ node.label }}</span>
                <span class="count">{{ node.count }}</span>
                <q-menu context-menu touch-position auto-close>
                  <q-list dense style="min-width: 170px">
                    <q-item clickable @click="newChat({ tags: [node.path] })">
                      <q-item-section>New chat with #{{ node.path }}</q-item-section>
                    </q-item>
                    <q-item clickable @click="searchTag(node.path)">
                      <q-item-section>Search #{{ node.path }}</q-item-section>
                    </q-item>
                    <q-item clickable @click="renameTag(node.path)">
                      <q-item-section>Rename tag…</q-item-section>
                    </q-item>
                  </q-list>
                </q-menu>
              </div>
            </template>
          </TreeList>
          <div class="q-pa-xs border-t q-mt-xs">
            <div class="section-label">{{ selectedTag ? `#${selectedTag}` : "Untagged chats" }}</div>
            <ChatRow v-for="c in tagChats" :key="c.id" :chat="c" dense />
            <div v-if="!tagChats.length" class="text-caption text-muted q-px-sm q-py-xs">No chats.</div>
            <template v-if="selectedTag && tagBoards.length">
              <div class="section-label">Boards</div>
              <BoardRow v-for="b in tagBoards" :key="b.name" :board="b" dense />
            </template>
            <template v-if="selectedTag">
              <div class="section-label">Files</div>
              <div
                v-for="f in tagFiles"
                :key="f.key"
                class="hit"
                @click="openHit(f)"
                @dblclick="openHit(f, true)"
              >
                <q-icon :name="fileHitIcon(f.kind)" class="text-muted hit-icon" />
                <span class="col ellipsis-1">{{ f.title }}</span>
                <span v-if="f.modified" class="age">{{ relativeTime(f.modified) }}</span>
              </div>
              <div v-if="!tagFiles.length" class="text-caption text-muted q-px-sm q-py-xs">No files.</div>
            </template>
          </div>
        </template>

        <!-- All -->
        <div v-else class="q-pa-xs">
          <ChatRow v-for="c in unpinned" :key="c.id" :chat="c" dense />
          <div v-if="!unpinned.length" class="text-caption text-muted q-pa-sm">No chats yet.</div>
        </div>

        <q-separator class="q-mt-sm" />
        <q-expansion-item v-model="archivedOpen" dense :icon="ICONS.archive" label="Archived" header-class="drawer-section">
          <div class="q-pa-xs">
            <ChatRow v-for="c in chats.archived" :key="c.id" :chat="c" dense />
            <div v-if="chats.archivedLoaded && !chats.archived.length" class="text-caption text-muted q-pa-sm">
              Nothing archived.
            </div>
          </div>
        </q-expansion-item>
        <q-expansion-item
          dense
          :icon="ICONS.clip"
          :label="`Saved clips${chats.clips.length ? ` (${chats.clips.length})` : ''}`"
          header-class="drawer-section"
        >
          <div class="q-pa-xs">
            <div
              v-for="c in chats.clips"
              :key="c.id"
              v-target="clipTarget(c)"
              class="hit"
              @click="openClipRow(c)"
              @dblclick="openClipRow(c, true)"
            >
              <q-icon :name="c.vault_key ? 'description' : 'content_cut'" class="text-muted hit-icon" />
              <div class="col" style="min-width: 0">
                <div class="row no-wrap items-center">
                  <span class="col ellipsis-1">{{ c.title || "Clip" }}</span>
                  <span v-if="c.created_at" class="age">{{ relativeTime(c.created_at) }}</span>
                </div>
                <div v-if="c.tags.length">
                  <span v-for="t in c.tags" :key="t" class="tag-chip">#{{ t }}</span>
                </div>
              </div>
            </div>
            <div v-if="!chats.clips.length" class="text-caption text-muted q-pa-sm">No clips yet.</div>
          </div>
        </q-expansion-item>
        <div class="root-drop-pad" />
      </template>
      <div v-if="rootOver" class="drop-overlay root-drop-hint">
        <q-icon :name="ICONS.inbox" size="18px" />Drop to move to the top level
      </div>
    </div>

    <NewBoardDialog
      :model-value="!!newBoard"
      :folder-id="newBoard?.folderId ?? null"
      :tags="newBoard?.tags ?? []"
      @update:model-value="(v: boolean) => !v && (newBoard = null)"
    />
  </div>
</template>

<style scoped>
.folder-header {
  min-height: 28px;
}
.count {
  flex: none;
  font-size: 0.72rem;
  color: var(--pawn-muted);
  margin-left: 6px;
}
.hit-icon.ann-hit-swatch {
  width: 12px;
  height: 12px;
  margin: 3px 3px 0;
}
.ann-hit-quote {
  font-style: italic;
}
.hit {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 6px;
  cursor: pointer;
}
.hit:hover {
  background: var(--pawn-hover);
}
.hit.active {
  background: var(--pawn-active);
}
.hit-icon {
  margin-top: 1px;
  flex: none;
  width: 18px;
  font-size: var(--pawn-row-icon);
}
.snippet {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  overflow-wrap: anywhere;
}
.kbd {
  font-size: 0.68rem;
  color: var(--pawn-muted);
  border: 1px solid var(--pawn-border);
  border-radius: 4px;
  padding: 0 4px;
}
.head-btn {
  width: 32px;
}
.root-drop-pad {
  min-height: 48px;
}
.root-drop-hint {
  position: sticky;
  inset: auto;
  bottom: 6px;
  margin: 0 6px;
  height: 44px;
}
:deep(.drawer-section) {
  color: var(--pawn-muted);
  min-height: 34px;
  font-size: 0.8rem;
}
:deep(.drawer-section .q-item__section--avatar) {
  min-width: 28px;
  padding-right: 6px;
}
:deep(.drawer-section .q-icon) {
  font-size: 18px;
}
</style>
