<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { api } from "@/api";
import { useFilesStore } from "@/stores/files";
import { useTabsStore } from "@/stores/tabs";
import { closeRef, openFile, retargetFile } from "@/router";
import { confirmAction, promptTags, promptText } from "@/lib/dialogs";
import { DND_FILE, hasOsFiles, hasType, leftElement, moveDestination, setDragLabel } from "@/lib/dnd";
import { fileIcon, ICONS } from "@/lib/icons";
import type { SegmentOption } from "@/lib/segment";
import { notifyError } from "@/lib/errors";
import { loadJson, loadString, saveJson, saveString } from "@/lib/storage";
import { fullTime, relativeTime } from "@/lib/time";
import type { TreeItem } from "@/lib/tree";
import SegmentedControl from "./SegmentedControl.vue";
import TreeList from "./TreeList.vue";
import { vTarget, type Target } from "@/lib/targets";

const files = useFilesStore();
const tabs = useTabsStore();

type FilesView = "tree" | "recent";
const view = ref<FilesView>((loadString("files.view") as FilesView) || "tree");
watch(view, (v) => saveString("files.view", v));

const unseenCount = computed(() => files.unseen.length);
const viewOptions = computed<SegmentOption<FilesView>[]>(() => [
  { value: "tree", label: "Vault", icon: ICONS.tree, tooltip: "Browse the vault" },
  {
    value: "recent",
    label: "Recent",
    icon: ICONS.recent,
    count: unseenCount.value,
    tooltip: "Files Pawn and you changed lately",
  },
]);

/** Folder that Upload / New note target: the last folder clicked, else the scratchpad. */
const targetFolder = ref(loadString("files.target", ""));
watch(targetFolder, (v) => saveString("files.target", v));
const folderForWrites = computed(() => targetFolder.value || files.scratchDir);
const targetIsScratch = computed(() => folderForWrites.value === files.scratchDir);
const targetLabel = computed(() => (targetIsScratch.value ? "Scratchpad" : baseName(folderForWrites.value) || "Vault root"));

/** Recently used target folders, newest first (this browser). */
const recentTargets = ref<string[]>(loadJson<string[]>("files.recentTargets", []));
function setTarget(path: string): void {
  targetFolder.value = path === files.scratchDir ? "" : path;
  if (!path || path === files.scratchDir) return;
  recentTargets.value = [path, ...recentTargets.value.filter((p) => p !== path)].slice(0, 5);
  saveJson("files.recentTargets", recentTargets.value);
}

interface FileNode extends TreeItem {
  label: string;
  path: string;
  kind: "folder" | "file";
  scratch?: boolean;
  modified?: string | null;
  tags?: string[];
  children?: FileNode[];
}

/** Branch prefix keeps keys unique: the scratchpad also appears under Pawn/. */
function nodesFor(branch: string, prefix: string): FileNode[] | undefined {
  const listing = files.listings[prefix];
  if (!listing) return undefined;
  return [
    ...listing.folders.map((f) => folderNode(branch, f.path, f.name, false, f.modified)),
    ...listing.files.map(
      (f): FileNode => ({
        key: `${branch}|${f.path}`,
        label: f.name,
        path: f.path,
        kind: "file",
        modified: f.modified,
        tags: f.tags ?? [],
      }),
    ),
  ];
}

function folderNode(
  branch: string,
  path: string,
  label: string,
  scratch = false,
  modified: string | null = null,
): FileNode {
  const children = nodesFor(branch, path);
  return {
    key: `${branch}|${path}/`,
    label,
    path,
    kind: "folder",
    scratch,
    modified,
    lazy: children === undefined,
    expandable: true,
    children: children ?? [],
  };
}

const treeNodes = computed<FileNode[]>(() => {
  const root = nodesFor("V", "") ?? [];
  return [folderNode("S", files.scratchDir, "Scratchpad", true), ...root];
});

const expanded = ref<string[]>(loadJson<string[]>("files.expanded", []));
watch(expanded, (v) => saveJson("files.expanded", v));

// Keys of expanded folders load on start (the tree only lazy-loads on click).
for (const key of expanded.value) {
  const [, path] = key.split("|");
  if (path?.endsWith("/")) void files.load(path.slice(0, -1));
}

function onLazyLoad(node: FileNode): void {
  void files.load(node.path);
}

const activeKey = computed(() => (tabs.active?.kind === "file" ? tabs.active.ref : ""));
const selected = computed(() => {
  const key = activeKey.value;
  if (!key) return null;
  return key.startsWith(files.scratchDir + "/") ? `S|${key}` : `V|${key}`;
});
const markedKey = computed(() =>
  targetIsScratch.value ? `S|${files.scratchDir}/` : `V|${folderForWrites.value}/`,
);

function onSelect(node: FileNode): void {
  if (node.kind === "folder") setTarget(node.path);
  else open(node.path);
}

function open(path: string, pin = false): void {
  files.markSeen(path);
  openFile(path, { pin });
}


async function editTags(path: string): Promise<void> {
  try {
    const current = await api().getFileTags(path);
    const next = await promptTags(current.tags);
    if (next === null) return;
    await api().setFileTags(path, next);
    await files.refreshFor([path]);
  } catch (e) {
    notifyError(e, "Could not save tags");
  }
}

function onDragStart(ev: DragEvent, path: string): void {
  ev.dataTransfer?.setData(DND_FILE, path);
  ev.dataTransfer?.setData("text/plain", path);
  if (ev.dataTransfer) ev.dataTransfer.effectAllowed = "copyMove";
  setDragLabel(ev, baseName(path));
}

// ── drop: vault files move into a folder, OS files upload ───────────────
function onRowDragOver(ev: DragEvent, node: FileNode): void {
  if (node.kind !== "folder") return;
  if (hasType(ev, DND_FILE)) {
    ev.preventDefault();
    if (ev.dataTransfer) ev.dataTransfer.dropEffect = "move";
  } else if (hasOsFiles(ev)) {
    ev.preventDefault();
    if (ev.dataTransfer) ev.dataTransfer.dropEffect = "copy";
  }
}

function onRowDrop(ev: DragEvent, node: FileNode): void {
  if (node.kind !== "folder") return;
  ev.preventDefault();
  ev.stopPropagation();
  osOver.value = false;
  const path = ev.dataTransfer?.getData(DND_FILE);
  if (path) void moveInto(path, node.path);
  else if (ev.dataTransfer?.files.length) void uploadFiles(Array.from(ev.dataTransfer.files), node.path);
}

async function moveInto(src: string, folder: string): Promise<void> {
  const dst = moveDestination(src, folder);
  if (!dst) return;
  const listing = files.listings[folder];
  const exists = Boolean(listing?.files.some((f) => f.path === dst));
  if (exists && !(await confirmAction("Replace file", `${dst} already exists. Replace it?`))) return;
  const key = await files.move(src, dst, exists);
  if (key) retargetFile(src, key);
}

// OS files dragged anywhere over the sidebar upload into the target folder,
// unless they land on a folder row.
const osOver = ref(false);
function onPaneDragOver(ev: DragEvent): void {
  if (!hasOsFiles(ev)) return;
  // A folder row took it: that row shows the highlight instead.
  if (ev.defaultPrevented) {
    osOver.value = false;
    return;
  }
  ev.preventDefault();
  if (ev.dataTransfer) ev.dataTransfer.dropEffect = "copy";
  osOver.value = true;
}
function onPaneDragLeave(ev: DragEvent): void {
  if (leftElement(ev)) osOver.value = false;
}
function onPaneDrop(ev: DragEvent): void {
  osOver.value = false;
  if (!hasOsFiles(ev) || ev.defaultPrevented) return;
  ev.preventDefault();
  void uploadFiles(Array.from(ev.dataTransfer?.files ?? []), folderForWrites.value);
}

// ── actions ─────────────────────────────────────────────────────────────
const uploader = ref<HTMLInputElement | null>(null);
const uploadFolder = ref("");
const uploading = ref(false);

function pickUpload(folder = folderForWrites.value): void {
  uploadFolder.value = folder;
  uploader.value?.click();
}

async function onUploadPicked(ev: Event): Promise<void> {
  const input = ev.target as HTMLInputElement;
  const picked = Array.from(input.files ?? []);
  input.value = "";
  await uploadFiles(picked, uploadFolder.value);
}

async function uploadFiles(picked: File[], folder: string): Promise<void> {
  if (!picked.length) return;
  uploading.value = true;
  try {
    let last: string | null = null;
    for (const f of picked) last = (await files.upload(f, folder)) ?? last;
    if (last && picked.length === 1) open(last);
  } finally {
    uploading.value = false;
  }
}

async function newNote(folder = folderForWrites.value): Promise<void> {
  const name = await promptText("New note", "", { message: `In ${folder}/`, label: "Name" });
  if (!name) return;
  const file = /\.[a-z0-9]+$/i.test(name) ? name : `${name}.md`;
  const key = `${folder}/${file}`.replace(/\/+/g, "/");
  const title = file.replace(/\.md$/i, "");
  if (await files.createNote(key, `# ${title}\n\n`)) openFile(key, { pin: true, edit: true });
}

function fileTarget(path: string): Target {
  return {
    kind: "file",
    ref: path,
    label: path.split("/").pop() ?? path,
    local: [
      { label: "Edit tags…", icon: "sell", run: () => editTags(path) },
      { label: "Move / rename…", icon: "drive_file_move", run: () => moveFile(path) },
      { label: "Delete…", icon: "delete", danger: true, run: () => deleteFile(path) },
    ],
  };
}

async function moveFile(path: string): Promise<void> {
  const dst = await promptText("Move / rename", path, { message: "New vault path" });
  if (!dst || dst === path) return;
  const key = await files.move(path, dst);
  if (key) retargetFile(path, key);
}

async function deleteFile(path: string): Promise<void> {
  if (!(await confirmAction("Delete file", `Delete ${path}? This can't be undone.`))) return;
  if (await files.remove(path)) closeRef("file", path);
}

function copyPath(path: string): void {
  void navigator.clipboard?.writeText(path).catch(() => {});
}

function refresh(): void {
  for (const prefix of Object.keys(files.listings)) void files.load(prefix);
}

const recentVisible = computed(() => files.recent.slice(0, 60));
function baseName(path: string): string {
  return path.split("/").pop() ?? path;
}
function dirName(path: string): string {
  const i = path.lastIndexOf("/");
  return i > 0 ? path.slice(0, i) : "";
}
</script>

<template>
  <div
    class="column no-wrap full-height sidebar-root relative-position"
    @dragover="onPaneDragOver"
    @dragleave="onPaneDragLeave"
    @drop="onPaneDrop"
  >
    <div class="q-pa-sm border-b files-head">
      <div class="row items-center no-wrap q-gutter-x-xs">
        <SegmentedControl v-model="view" :options="viewOptions" size="sm" stretch aria-label="Files view" class="col" />
        <q-btn flat dense round size="sm" :icon="ICONS.noteAdd" aria-label="New note" @click="newNote()">
          <q-tooltip>New note in {{ targetLabel }}</q-tooltip>
        </q-btn>
        <q-btn
          flat
          dense
          round
          size="sm"
          :icon="ICONS.upload"
          aria-label="Upload"
          :loading="uploading"
          @click="pickUpload()"
        >
          <q-tooltip>Upload to {{ targetLabel }} (or drop files here)</q-tooltip>
        </q-btn>
        <q-btn flat dense round size="sm" :icon="ICONS.refresh" aria-label="Refresh" @click="refresh">
          <q-tooltip>Refresh</q-tooltip>
        </q-btn>
      </div>
      <div class="row items-center no-wrap q-mt-xs target-row">
        <span class="target-caption">Save to</span>
        <button type="button" class="target-chip" :title="folderForWrites">
          <q-icon :name="targetIsScratch ? ICONS.scratch : ICONS.folder" class="target-icon" />
          <span class="ellipsis-1">{{ targetLabel }}</span>
          <q-icon name="expand_more" size="14px" class="text-muted" />
          <q-menu auto-close anchor="bottom left" self="top left">
            <q-list dense style="min-width: 220px">
              <q-item-label header class="q-py-xs">New notes and uploads go here</q-item-label>
              <q-item clickable :active="targetIsScratch" @click="setTarget(files.scratchDir)">
                <q-item-section avatar class="menu-icon"><q-icon :name="ICONS.scratch" /></q-item-section>
                <q-item-section>Scratchpad</q-item-section>
              </q-item>
              <q-item
                v-for="p in recentTargets"
                :key="p"
                clickable
                :active="p === folderForWrites"
                @click="setTarget(p)"
              >
                <q-item-section avatar class="menu-icon"><q-icon :name="ICONS.folder" /></q-item-section>
                <q-item-section>
                  <q-item-label class="ellipsis-1">{{ baseName(p) }}</q-item-label>
                  <q-item-label caption class="ellipsis-1">{{ p }}</q-item-label>
                </q-item-section>
              </q-item>
              <q-separator />
              <q-item-label caption class="q-px-md q-py-sm">Click a folder in the tree to target it.</q-item-label>
            </q-list>
          </q-menu>
        </button>
      </div>
      <input ref="uploader" type="file" multiple hidden @change="onUploadPicked" />
    </div>

    <div class="col pawn-scroll">
      <TreeList
        v-if="view === 'tree'"
        v-model:expanded="expanded"
        :nodes="treeNodes"
        :selected="selected"
        :marked="markedKey"
        @select="onSelect"
        @lazy-load="onLazyLoad"
        @row-dragover="onRowDragOver"
        @row-drop="onRowDrop"
      >
        <template #default="{ node, open: isOpen }">
          <div
            v-target="node.kind === 'file' ? fileTarget(node.path) : null"
            class="row items-center no-wrap full-width"
            :draggable="node.kind === 'file'"
            @dragstart="node.kind === 'file' && onDragStart($event, node.path)"
            @dblclick="node.kind === 'file' && open(node.path, true)"
          >
            <q-icon
              :name="
                node.scratch
                  ? ICONS.scratch
                  : node.kind === 'folder'
                    ? isOpen
                      ? ICONS.folderOpen
                      : ICONS.folder
                    : fileIcon(node.path)
              "
              class="row-icon"
              :class="node.scratch ? 'scratch-icon' : node.kind === 'folder' ? 'folder-icon' : ''"
            />
            <span class="col ellipsis-1" :class="{ 'text-weight-medium': node.scratch }">{{ node.label }}</span>
            <q-icon v-if="node.tags?.length" :name="ICONS.tag" size="12px" class="text-muted q-ml-xs row-tag">
              <q-tooltip>{{ node.tags.map((t: string) => "#" + t).join(" ") }}</q-tooltip>
            </q-icon>
            <span
              v-if="node.kind === 'file' ? files.isUnseen(node.path) : files.folderUnseen(node.path)"
              class="unseen-dot q-ml-xs"
            />
            <span v-if="node.modified" class="age">
              {{ relativeTime(node.modified) }}
              <q-tooltip :delay="600">{{ fullTime(node.modified) }}</q-tooltip>
            </span>

            <q-menu v-if="node.kind !== 'file'" context-menu touch-position auto-close>
              <q-list dense style="min-width: 170px">
                <q-item clickable @click="setTarget(node.path)"><q-item-section>Save new files here</q-item-section></q-item>
                <q-item clickable @click="newNote(node.path)"><q-item-section>New note here…</q-item-section></q-item>
                <q-item clickable @click="pickUpload(node.path)"><q-item-section>Upload here…</q-item-section></q-item>
                <q-item clickable @click="files.load(node.path)"><q-item-section>Refresh</q-item-section></q-item>
                <q-item clickable @click="copyPath(node.path)"><q-item-section>Copy path</q-item-section></q-item>
              </q-list>
            </q-menu>
          </div>
        </template>
      </TreeList>

      <q-list v-else dense class="q-pa-xs">
        <q-item v-if="unseenCount" dense clickable class="text-caption text-primary" @click="files.markAllSeen()">
          <q-item-section>Mark all as seen</q-item-section>
        </q-item>
        <q-item
          v-for="r in recentVisible"
          :key="r.path"
          :clickable="!r.deleted"
          dense
          v-target="r.deleted ? null : fileTarget(r.path)"
          :draggable="!r.deleted"
          @dragstart="onDragStart($event, r.path)"
          @click="!r.deleted && open(r.path)"
          @dblclick="!r.deleted && open(r.path, true)"
        >
          <q-item-section avatar style="min-width: 18px">
            <span v-if="files.isUnseen(r.path)" class="unseen-dot" />
          </q-item-section>
          <q-item-section>
            <q-item-label class="ellipsis-1" :class="{ 'text-strike text-muted': r.deleted }">
              {{ baseName(r.path) }}
            </q-item-label>
            <q-item-label caption class="ellipsis-1">{{ dirName(r.path) }}</q-item-label>
          </q-item-section>
          <q-item-section side class="text-caption text-muted">
            <div>
              {{ relativeTime(new Date(r.at).toISOString()) }}
              <q-tooltip :delay="600">{{ fullTime(new Date(r.at).toISOString()) }}</q-tooltip>
            </div>
            <div>{{ r.source }}</div>
          </q-item-section>
        </q-item>
        <q-item v-if="!recentVisible.length" dense class="text-caption text-muted">
          Files Pawn writes show up here.
        </q-item>
      </q-list>
    </div>
    <div v-if="osOver" class="drop-overlay">
      <q-icon :name="ICONS.upload" size="20px" />Drop to upload to {{ targetLabel }}
    </div>
  </div>
</template>

<style scoped>
.scratch-icon {
  color: var(--q-primary);
}
.target-row {
  gap: 6px;
  min-width: 0;
}
.target-caption {
  flex: none;
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--pawn-muted);
}
.target-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
  max-width: 100%;
  height: 24px;
  padding: 0 6px 0 4px;
  border: 1px solid var(--pawn-border);
  border-radius: 6px;
  background: transparent;
  color: inherit;
  font: inherit;
  font-size: 0.78rem;
  cursor: pointer;
}
.target-chip:hover {
  background: var(--pawn-hover);
}
.target-icon {
  font-size: 15px;
  color: var(--pawn-folder);
  flex: none;
}
.menu-icon {
  min-width: 28px;
}
</style>
