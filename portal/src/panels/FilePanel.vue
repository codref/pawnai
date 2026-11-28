<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Dialog } from "quasar";
import { api, HttpError, type Chat } from "@/api";
import { useFilesStore } from "@/stores/files";
import { useTabsStore } from "@/stores/tabs";
import { isSideChat, useChatsStore } from "@/stores/chats";
import { useAnnotationsStore } from "@/stores/annotations";
import { openChat, openDay, openSession } from "@/router";
import { vTarget, type Target } from "@/lib/targets";
import { toWorkspace } from "@/lib/workspace";
import { errorText, notifyError } from "@/lib/errors";
import { promptTags } from "@/lib/dialogs";
import { splitFrontmatter } from "@/lib/markdown";
import { fullTime, relativeTime } from "@/lib/time";
import MarkdownView from "@/components/MarkdownView.vue";
import CodeEditor from "@/components/CodeEditor.vue";
import AnnotationLayer from "@/components/AnnotationLayer.vue";

const props = defineProps<{ fileKey: string; tabId: string; edit?: boolean }>();
const files = useFilesStore();
const tabs = useTabsStore();
const chats = useChatsStore();

const IMAGE_RE = /\.(png|jpe?g|gif|webp|svg|bmp|avif)$/i;
const TEXT_RE = /\.(md|txt|json|ya?ml|csv|base|canvas)$/i;

const loading = ref(true);
const error = ref("");
const content = ref("");
const etag = ref("");
const writable = ref(false);
const lastModified = ref<string | null>(null);
const blobUrl = ref("");
const editing = ref(false);
const draft = ref("");
const saving = ref(false);
const remoteChanged = ref(false);
const tags = ref<string[]>([]);
const tagsSource = ref<"frontmatter" | "db" | "">("");
const savingTags = ref(false);

const isImage = computed(() => IMAGE_RE.test(props.fileKey));
const isText = computed(() => TEXT_RE.test(props.fileKey) || !/\.[a-z0-9]+$/i.test(props.fileKey));
const isMarkdown = computed(() => /\.md$/i.test(props.fileKey));
const dirty = computed(() => editing.value && draft.value !== content.value);
const parts = computed(() => splitFrontmatter(content.value));
const crumbs = computed(() => props.fileKey.split("/"));

function front(name: string): string {
  const hit = parts.value.frontmatter?.fields.find(([key]) => key === name);
  return (hit?.[1] ?? "").trim().replace(/^["']|["']$/g, "");
}

const transcriptSession = computed(() => (front("pawn") === "transcript" ? front("session_id") : ""));
const eventId = computed(() => (front("pawn") === "event" ? front("id") : ""));
const eventDay = computed(() => front("start").slice(0, 10));

async function load(): Promise<void> {
  loading.value = true;
  error.value = "";
  remoteChanged.value = false;
  try {
    if (isText.value) {
      const f = await api().readFile(props.fileKey);
      content.value = f.content;
      etag.value = f.etag;
      writable.value = f.writable;
      lastModified.value = f.last_modified;
      tags.value = f.tags ?? [];
    } else {
      const blob = await api().readFileBlob(props.fileKey);
      if (blobUrl.value) URL.revokeObjectURL(blobUrl.value);
      blobUrl.value = URL.createObjectURL(blob);
      writable.value = false;
      void loadTags();
    }
    files.markSeen(props.fileKey);
  } catch (e) {
    error.value = e instanceof HttpError && e.status === 404 ? "This file no longer exists." : errorText(e);
  } finally {
    loading.value = false;
  }
}

async function loadTags(): Promise<void> {
  try {
    const t = await api().getFileTags(props.fileKey);
    tags.value = t.tags;
    tagsSource.value = t.source;
  } catch {
    /* missing file: load() reports it */
  }
}

async function editTags(): Promise<void> {
  const next = await promptTags(tags.value);
  if (next === null) return;
  savingTags.value = true;
  try {
    const t = await api().setFileTags(props.fileKey, next);
    tags.value = t.tags;
    tagsSource.value = t.source;
    tabs.pin(props.tabId);
    void files.refreshFor([props.fileKey]);
    // Frontmatter edits change the file body (and its etag).
    if (t.source === "frontmatter" && !editing.value) await load();
  } catch (e) {
    notifyError(e, "Could not save tags");
  } finally {
    savingTags.value = false;
  }
}

function startEdit(): void {
  draft.value = content.value;
  editing.value = true;
  tabs.pin(props.tabId);
  tabs.clearEdit(props.tabId);
}

async function cancelEdit(): Promise<void> {
  if (dirty.value && !(await confirmDiscard())) return;
  editing.value = false;
  if (remoteChanged.value) await load();
}

function confirmDiscard(): Promise<boolean> {
  return new Promise((resolve) => {
    Dialog.create({
      title: "Discard changes?",
      message: `Your edits to ${props.fileKey} are not saved.`,
      cancel: { label: "Keep editing", flat: true },
      ok: { label: "Discard", color: "negative", flat: true },
    })
      .onOk(() => resolve(true))
      .onCancel(() => resolve(false))
      .onDismiss(() => resolve(false));
  });
}

async function save(force = false): Promise<void> {
  if (!editing.value || saving.value) return;
  saving.value = true;
  try {
    let tag = etag.value;
    if (force) tag = (await api().readFile(props.fileKey)).etag;
    const stat = await api().writeFile(props.fileKey, draft.value, tag);
    tabs.pin(props.tabId);
    etag.value = stat.etag;
    content.value = draft.value;
    lastModified.value = new Date().toISOString();
    remoteChanged.value = false;
    editing.value = false;
  } catch (e) {
    if (e instanceof HttpError && e.status === 409) onConflict();
    else notifyError(e, "Save failed");
  } finally {
    saving.value = false;
  }
}

/** Stale etag: someone (usually Pawn) wrote the file after we read it. */
function onConflict(): void {
  Dialog.create({
    title: "File changed on the server",
    message: `${props.fileKey} was changed after you opened it. Overwrite it with your version, or reload theirs and lose your edits?`,
    cancel: { label: "Keep editing", flat: true },
    ok: { label: "Overwrite", color: "negative", flat: true },
    options: {
      type: "radio",
      model: "overwrite",
      items: [
        { label: "Overwrite with my version", value: "overwrite" },
        { label: "Reload theirs (discard my edits)", value: "reload" },
      ],
    },
  }).onOk(async (choice: string) => {
    if (choice === "reload") {
      editing.value = false;
      await load();
    } else {
      await save(true);
    }
  });
}

// ── right-click layer: the body is a file target; side chats about it ────
const fileName = computed(() => crumbs.value[crumbs.value.length - 1] ?? props.fileKey);
const bodyTarget = computed<Target>(() => ({
  kind: "file",
  ref: props.fileKey,
  label: fileName.value,
  text: isText.value ? parts.value.body : undefined,
}));

// Annotations on the file body (text files only).
const annotations = useAnnotationsStore();
const bodyEl = ref<HTMLElement | null>(null);
const fileRefs = computed(() => [props.fileKey]);
const fileNotes = computed(() => annotations.forTargets("file", fileRefs.value));
function noteLabel(body: string, quote?: string): string {
  const one = (body || (quote ? `“${quote}”` : "")).replace(/\s+/g, " ").trim();
  return one.length > 60 ? one.slice(0, 59) + "…" : one || "Highlight";
}

const sideChats = ref<Chat[]>([]);
async function loadSideChats(): Promise<void> {
  try {
    sideChats.value = (await api().listChats({ origin_ref: props.fileKey })).filter(isSideChat);
  } catch {
    sideChats.value = [];
  }
}
watch(() => [props.fileKey, chats.revision], () => void loadSideChats(), { immediate: true });

function openSide(c: Chat, ev?: Event): void {
  const me = ev instanceof MouseEvent ? ev : null;
  openChat(c.id, { where: "float", at: me ? toWorkspace(me.clientX + 12, me.clientY + 8) : undefined });
}

// Live updates from /v1/vault/events.
watch(
  () => files.changed[props.fileKey],
  (n, old) => {
    if (!n || n === old || saving.value) return;
    if (editing.value) remoteChanged.value = true;
    else void load();
  },
);

watch(dirty, (d) => tabs.setStatus(props.tabId, { dirty: d }), { immediate: true });

function beforeUnload(ev: BeforeUnloadEvent): void {
  if (dirty.value) ev.preventDefault();
}
onMounted(async () => {
  window.addEventListener("beforeunload", beforeUnload);
  await load();
  if (props.edit && writable.value) startEdit();
});
onBeforeUnmount(() => {
  window.removeEventListener("beforeunload", beforeUnload);
  if (blobUrl.value) URL.revokeObjectURL(blobUrl.value);
});
</script>

<template>
  <div class="column no-wrap full-height">
    <div class="row items-center no-wrap q-px-md q-py-sm border-b">
      <q-icon :name="isImage ? 'image' : 'description'" size="20px" class="q-mr-sm text-muted" />
      <div class="col ellipsis-1">
        <span v-for="(c, i) in crumbs" :key="i">
          <span :class="i === crumbs.length - 1 ? 'text-weight-medium' : 'text-muted'">{{ c }}</span>
          <span v-if="i < crumbs.length - 1" class="text-muted q-mx-xs">/</span>
        </span>
      </div>
      <span v-if="lastModified && !editing" class="text-caption text-muted q-mr-sm">
        {{ relativeTime(lastModified) }}
        <q-tooltip>Modified {{ fullTime(lastModified) }}</q-tooltip>
      </span>
      <q-badge v-if="!loading && isText && !writable" outline color="grey" label="read-only" class="q-mr-sm">
        <q-tooltip>Outside Pawn/ and not marked “pawn: editable”</q-tooltip>
      </q-badge>
      <template v-if="editing">
        <span v-if="dirty" class="pawn-status warn q-mr-sm">unsaved</span>
        <q-btn flat dense no-caps label="Cancel" class="q-mr-xs" @click="cancelEdit" />
        <q-btn unelevated dense no-caps color="primary" icon="save" label="Save" :loading="saving" @click="save()" />
      </template>
      <q-btn
        v-else-if="isText && writable && !loading && !error"
        flat
        dense
        no-caps
        icon="edit"
        label="Edit"
        @click="startEdit"
      />
      <q-btn v-if="sideChats.length" flat dense no-caps size="sm" icon="question_answer" :label="String(sideChats.length)" class="q-ml-xs">
        <q-tooltip>Side chats about this file</q-tooltip>
        <q-menu auto-close>
          <q-list dense style="min-width: 220px">
            <q-item v-for="c in sideChats" :key="c.id" clickable @click="openSide(c, $event)">
              <q-item-section>
                <q-item-label class="ellipsis-1">{{ c.title || "Side chat" }}</q-item-label>
                <q-item-label v-if="c.origin?.quote" caption class="ellipsis-1">“{{ c.origin.quote }}”</q-item-label>
              </q-item-section>
            </q-item>
          </q-list>
        </q-menu>
      </q-btn>
      <q-btn
        v-if="fileNotes.length && !editing"
        flat
        dense
        no-caps
        size="sm"
        icon="sticky_note_2"
        :label="String(fileNotes.length)"
        class="q-ml-xs"
      >
        <q-tooltip>Annotations on this file</q-tooltip>
        <q-menu auto-close>
          <q-list dense style="min-width: 240px; max-width: 340px">
            <q-item v-for="a in fileNotes" :key="a.id" clickable @click="annotations.requestFocus(a.id)">
              <q-item-section>
                <q-item-label class="ellipsis-1">{{ noteLabel(a.body_md, a.anchor?.exact) }}</q-item-label>
                <q-item-label v-if="a.body_md && a.anchor" caption class="ellipsis-1">“{{ a.anchor.exact }}”</q-item-label>
              </q-item-section>
            </q-item>
          </q-list>
        </q-menu>
      </q-btn>
      <q-btn flat dense round icon="refresh" class="q-ml-xs" aria-label="Reload" :disable="editing" @click="load" />
    </div>

    <div v-if="!loading && !error" class="row items-center q-px-md q-py-xs border-b tag-row">
      <q-icon name="sell" size="14px" class="text-muted q-mr-xs" />
      <span v-for="t in tags" :key="t" class="tag-chip cursor-pointer" @click="editTags">#{{ t }}</span>
      <q-btn
        flat
        dense
        no-caps
        size="sm"
        :label="tags.length ? 'Edit' : 'Add tags'"
        :loading="savingTags"
        class="text-muted"
        @click="editTags"
      >
        <q-tooltip>
          {{
            tagsSource === "db" || (!writable && tagsSource !== "frontmatter")
              ? "Stored by Pawn (this file isn't editable)"
              : "Saved in the note's frontmatter"
          }}
        </q-tooltip>
      </q-btn>
    </div>

    <q-banner v-if="transcriptSession" dense class="pawn-banner">
      This note is the transcript for {{ transcriptSession }}.
      <template #action>
        <q-btn flat dense no-caps label="Open as session" @click="openSession(transcriptSession)" />
      </template>
    </q-banner>
    <q-banner v-else-if="eventId" dense class="pawn-banner">
      This note is a day event.
      <template #action>
        <q-btn
          flat
          dense
          no-caps
          label="Open in Day"
          @click="openDay({ date: eventDay || undefined, eventId })"
        />
      </template>
    </q-banner>

    <q-banner v-if="remoteChanged && editing" dense class="pawn-banner warn">
      This file changed on the server while you were editing. Saving will ask before overwriting.
    </q-banner>

    <div v-if="loading" class="col flex flex-center"><q-spinner size="32px" /></div>
    <div v-else-if="error" class="col flex flex-center column text-muted">
      <q-icon name="error_outline" size="40px" />
      <div class="q-mt-sm">{{ error }}</div>
    </div>
    <div v-else-if="editing" class="col q-pa-md editor-wrap">
      <CodeEditor v-model="draft" @save="save()" />
    </div>
    <div v-else class="col pawn-scroll">
      <div class="row no-wrap q-pa-lg file-row">
      <div ref="bodyEl" v-target="bodyTarget" class="col file-body">
        <img v-if="isImage" :src="blobUrl" :alt="fileKey" class="file-image" />
        <div v-else-if="!isText" class="text-muted">
          No preview for this file type.
          <a :href="blobUrl" :download="crumbs[crumbs.length - 1]">Download</a>
        </div>
        <template v-else-if="isMarkdown">
          <div v-if="parts.frontmatter && parts.frontmatter.fields.length" class="frontmatter q-mb-md">
            <div v-for="[k, v] in parts.frontmatter.fields" :key="k" class="row no-wrap">
              <span class="fm-key text-muted">{{ k }}</span>
              <span class="col ellipsis-1">{{ v }}</span>
            </div>
          </div>
          <MarkdownView :source="parts.body" :from-key="fileKey" />
        </template>
        <pre v-else class="plain" data-pawn-anchor-root>{{ content }}</pre>
      </div>
      <AnnotationLayer v-if="isText" kind="file" :refs="fileRefs" :root-for="() => bodyEl" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.file-row {
  max-width: 1100px;
  margin: 0 auto;
  justify-content: center;
}
.file-body {
  max-width: 860px;
  min-width: 0;
}
.tag-row {
  min-height: 30px;
}
.editor-wrap {
  min-height: 0;
}
.file-image {
  max-width: 100%;
  border-radius: 4px;
}
.frontmatter {
  font-size: 0.85rem;
  border: 1px solid var(--pawn-border);
  border-radius: 6px;
  padding: 6px 10px;
}
.fm-key {
  width: 120px;
  flex: none;
}
.plain {
  white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.85rem;
}
</style>
