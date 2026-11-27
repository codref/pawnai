<script setup lang="ts">
// A saved portion of a chat: view or edit its title, tags and Markdown, save
// it to the scratchpad, or jump back to the chat it came from.
import { computed, onMounted, ref, watch } from "vue";
import { api, HttpError, type Clip } from "@/api";
import { useChatsStore } from "@/stores/chats";
import { useTabsStore } from "@/stores/tabs";
import { closeTab, openChat, openFile } from "@/router";
import { confirmAction, promptTags, promptText } from "@/lib/dialogs";
import { errorText, notifyError } from "@/lib/errors";
import { relativeTime } from "@/lib/time";
import { vTarget, type Target } from "@/lib/targets";
import MarkdownView from "@/components/MarkdownView.vue";
import CodeEditor from "@/components/CodeEditor.vue";
import AnnotationLayer from "@/components/AnnotationLayer.vue";

const props = defineProps<{ id: string; tabId: string }>();
const chats = useChatsStore();
const tabs = useTabsStore();

const clip = ref<Clip | null>(null);
const loading = ref(true);
const error = ref("");
const editing = ref(false);
const draft = ref("");
const saving = ref(false);
const dirty = computed(() => editing.value && draft.value !== clip.value?.content_md);

const bodyEl = ref<HTMLElement | null>(null);
const clipRefs = computed(() => [props.id]);

const target = computed<Target | null>(() =>
  clip.value
    ? { kind: "clip", ref: clip.value.id, label: clip.value.title || "Clip", text: clip.value.content_md }
    : null,
);

async function load(): Promise<void> {
  loading.value = true;
  error.value = "";
  try {
    clip.value = await api().getClip(props.id);
    tabs.setStatus(props.tabId, { title: clip.value.title || "Clip" });
  } catch (e) {
    error.value = e instanceof HttpError && e.status === 404 ? "This clip no longer exists." : errorText(e);
  } finally {
    loading.value = false;
  }
}

async function patch(body: Partial<Pick<Clip, "title" | "tags" | "content_md">>): Promise<boolean> {
  if (!clip.value) return false;
  try {
    clip.value = await api().patchClip(clip.value.id, body);
    tabs.setStatus(props.tabId, { title: clip.value.title || "Clip" });
    void chats.refreshClips();
    if (body.tags) void chats.refreshTags();
    return true;
  } catch (e) {
    notifyError(e, "Could not update clip");
    return false;
  }
}

async function rename(): Promise<void> {
  const title = await promptText("Rename clip", clip.value?.title ?? "");
  if (title !== null) await patch({ title });
}

async function editTags(): Promise<void> {
  const tags = await promptTags(clip.value?.tags ?? []);
  if (tags !== null) await patch({ tags });
}

function startEdit(): void {
  draft.value = clip.value?.content_md ?? "";
  editing.value = true;
  tabs.pin(props.tabId);
}

async function saveEdit(): Promise<void> {
  saving.value = true;
  try {
    if (await patch({ content_md: draft.value })) editing.value = false;
  } finally {
    saving.value = false;
  }
}

async function cancelEdit(): Promise<void> {
  if (dirty.value && !(await confirmAction("Discard changes?", "Your edits to this clip are not saved.", "Discard")))
    return;
  editing.value = false;
}

async function saveToScratchpad(): Promise<void> {
  if (!clip.value) return;
  saving.value = true;
  try {
    const stat = await api().saveClip(clip.value.id);
    clip.value = { ...clip.value, vault_key: stat.key };
    void chats.refreshClips();
    openFile(stat.key, { pin: true });
  } catch (e) {
    notifyError(e, "Could not save clip");
  } finally {
    saving.value = false;
  }
}

async function remove(): Promise<void> {
  if (!clip.value) return;
  const ok = await confirmAction("Delete clip", `Delete “${clip.value.title || "Clip"}”? A note saved to the scratchpad stays.`);
  if (!ok) return;
  try {
    await api().deleteClip(clip.value.id);
    void chats.refreshClips();
    closeTab(props.tabId);
  } catch (e) {
    notifyError(e, "Could not delete clip");
  }
}

watch(dirty, (d) => tabs.setStatus(props.tabId, { dirty: d }), { immediate: true });
onMounted(load);
</script>

<template>
  <div class="column no-wrap full-height">
    <div v-if="loading" class="col flex flex-center"><q-spinner size="32px" /></div>
    <div v-else-if="error || !clip" class="col flex flex-center column text-muted">
      <q-icon name="error_outline" size="40px" />
      <div class="q-mt-sm">{{ error }}</div>
      <q-btn flat no-caps label="Close tab" class="q-mt-sm" @click="closeTab(tabId)" />
    </div>
    <template v-else>
      <div class="row items-center no-wrap q-px-md q-py-sm border-b">
        <q-icon name="content_cut" size="20px" class="q-mr-sm text-muted" />
        <div class="col" style="min-width: 0">
          <div class="text-subtitle1 ellipsis-1 cursor-pointer" @click="rename">
            {{ clip.title || "Clip" }}
            <q-tooltip>Rename</q-tooltip>
          </div>
          <div class="row items-center q-gutter-x-xs text-caption text-muted">
            <span v-for="t in clip.tags" :key="t" class="tag-chip cursor-pointer" @click="editTags">#{{ t }}</span>
            <q-btn v-if="!clip.tags.length" flat dense no-caps size="sm" icon="sell" label="Add tags" @click="editTags" />
            <span v-if="clip.created_at">· {{ relativeTime(clip.created_at) }}</span>
            <span v-if="clip.message_ids.length">· {{ clip.message_ids.length }} message(s)</span>
          </div>
        </div>
        <template v-if="editing">
          <span v-if="dirty" class="text-caption text-orange q-mr-sm">unsaved</span>
          <q-btn flat dense no-caps label="Cancel" class="q-mr-xs" @click="cancelEdit" />
          <q-btn unelevated dense no-caps color="primary" icon="save" label="Save" :loading="saving" @click="saveEdit" />
        </template>
        <template v-else>
          <q-btn v-if="clip.chat_id" flat dense round icon="forum" aria-label="Open source chat" @click="openChat(clip.chat_id)">
            <q-tooltip>Open the source chat</q-tooltip>
          </q-btn>
          <q-btn
            v-if="clip.vault_key"
            flat
            dense
            round
            icon="description"
            aria-label="Open saved note"
            @click="openFile(clip.vault_key, { pin: true })"
          >
            <q-tooltip>Open the saved note ({{ clip.vault_key }})</q-tooltip>
          </q-btn>
          <q-btn flat dense round icon="save_alt" aria-label="Save to scratchpad" :loading="saving" @click="saveToScratchpad">
            <q-tooltip>{{ clip.vault_key ? "Save again to the scratchpad" : "Save to the scratchpad" }}</q-tooltip>
          </q-btn>
          <q-btn flat dense round icon="edit" aria-label="Edit" @click="startEdit" />
          <q-btn flat dense round icon="delete" aria-label="Delete" @click="remove" />
        </template>
      </div>
      <div v-if="editing" class="col q-pa-md editor-wrap">
        <CodeEditor v-model="draft" @save="saveEdit" />
      </div>
      <q-scroll-area v-else class="col">
        <div class="row no-wrap q-pa-lg clip-row">
          <div ref="bodyEl" v-target="target" class="col clip-body">
            <MarkdownView :source="clip.content_md" />
          </div>
          <AnnotationLayer kind="clip" :refs="clipRefs" :root-for="() => bodyEl" />
        </div>
      </q-scroll-area>
    </template>
  </div>
</template>

<style scoped>
.clip-row {
  max-width: 1100px;
  margin: 0 auto;
  justify-content: center;
}
.clip-body {
  max-width: 860px;
  min-width: 0;
}
.editor-wrap {
  min-height: 0;
}
</style>
