<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { api } from "@/api";
import { useFilesStore } from "@/stores/files";
import { useTabsStore } from "@/stores/tabs";
import { closeRef, openFile, retargetFile } from "@/router";
import { confirmAction, promptTags, promptText } from "@/lib/dialogs";
import { DND_FILE } from "@/lib/dnd";
import { notifyError } from "@/lib/errors";
import { loadJson, loadString, saveJson, saveString } from "@/lib/storage";
import { fullTime, relativeTime } from "@/lib/time";
import type { TreeItem } from "@/lib/tree";
import TreeList from "./TreeList.vue";
import { vTarget, type Target } from "@/lib/targets";

const files = useFilesStore();
const tabs = useTabsStore();

type FilesView = "tree" | "recent";
const view = ref<FilesView>((loadString("files.view") as FilesView) || "tree");
watch(view, (v) => saveString("files.view", v));

/** Folder that Upload / New note target: the last folder clicked, else the scratchpad. */
const targetFolder = ref("");
const folderForWrites = computed(() => targetFolder.value || files.scratchDir);

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

function onSelect(node: FileNode): void {
  if (node.kind === "folder") targetFolder.value = node.path;
  else open(node.path);
}

function open(path: string, pin = false): void {
  files.markSeen(path);
  openFile(path, { pin });
}

function fileIcon(path: string): string {
  if (/\.(png|jpe?g|gif|webp|svg|bmp|avif)$/i.test(path)) return "image";
  if (/\.pdf$/i.test(path)) return "picture_as_pdf";
  if (/\.(mp3|wav|m4a|ogg|flac|webm|opus)$/i.test(path)) return "audiotrack";
  if (/\.md$/i.test(path)) return "article";
  return "description";
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
  if (ev.dataTransfer) ev.dataTransfer.effectAllowed = "copy";
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
  if (!picked.length) return;
  uploading.value = true;
  try {
    let last: string | null = null;
    for (const f of picked) last = (await files.upload(f, uploadFolder.value)) ?? last;
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

const unseenCount = computed(() => files.unseen.length);
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
  <div class="column no-wrap full-height">
    <div class="q-pa-sm border-b">
      <div class="row items-center no-wrap">
        <q-btn-toggle
          v-model="view"
          :options="[
            { value: 'tree', label: 'Files' },
            { value: 'recent', slot: 'recent' },
          ]"
          dense
          no-caps
          unelevated
          size="sm"
          toggle-color="primary"
          class="col"
        >
          <template #recent>
            <span>Recent</span>
            <q-badge v-if="unseenCount" color="blue" class="q-ml-xs" :label="unseenCount" />
          </template>
        </q-btn-toggle>
        <q-btn flat dense round size="sm" icon="refresh" aria-label="Refresh" @click="refresh" />
      </div>
      <div class="row q-gutter-xs q-mt-xs items-center">
        <q-btn dense flat no-caps icon="note_add" label="New note" size="sm" @click="newNote()" />
        <q-btn dense flat no-caps icon="upload" label="Upload" size="sm" :loading="uploading" @click="pickUpload()" />
        <q-space />
        <span class="text-caption text-muted ellipsis-1" style="max-width: 110px" :title="folderForWrites">
          → {{ baseName(folderForWrites) }}
        </span>
      </div>
      <input ref="uploader" type="file" multiple hidden @change="onUploadPicked" />
    </div>

    <q-scroll-area class="col">
      <TreeList
        v-if="view === 'tree'"
        v-model:expanded="expanded"
        :nodes="treeNodes"
        :selected="selected"
        @select="onSelect"
        @lazy-load="onLazyLoad"
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
                  ? 'edit_note'
                  : node.kind === 'folder'
                    ? isOpen
                      ? 'folder_open'
                      : 'folder'
                    : fileIcon(node.path)
              "
              size="17px"
              class="q-mr-xs"
              :class="node.scratch ? 'text-primary' : node.kind === 'folder' ? 'folder-icon' : 'text-muted'"
            />
            <span class="col ellipsis-1" :class="{ 'text-weight-medium': node.scratch }">{{ node.label }}</span>
            <q-icon v-if="node.tags?.length" name="sell" size="12px" class="text-muted q-ml-xs">
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
    </q-scroll-area>
  </div>
</template>

<style scoped>
.folder-icon {
  color: #e0a43a;
}
</style>
