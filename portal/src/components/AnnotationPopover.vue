<script setup lang="ts">
// Note editor for one annotation, opened by AnnotationLayer at the highlight
// (or the orphan marker): note text, colour, tags, the linked side chat, delete.
import { computed, ref, watch } from "vue";
import { api, type Annotation } from "@/api";
import { useAnnotationsStore } from "@/stores/annotations";
import { useChatsStore } from "@/stores/chats";
import { openChat } from "@/router";
import { askAboutAnnotation, deleteAnnotation } from "@/lib/actions";
import { promptTags } from "@/lib/dialogs";
import { ANNOTATION_COLORS, colorOf } from "@/lib/highlights";
import { toWorkspace } from "@/lib/workspace";

const props = defineProps<{
  modelValue: boolean;
  annotation: Annotation | null;
  /** Viewport point the menu opens below. */
  x: number;
  y: number;
  orphaned?: boolean;
  /** Chat of a message annotation (side chats nest under it). */
  chatId?: string;
}>();
const emit = defineEmits<{
  (e: "update:modelValue", v: boolean): void;
  (e: "reattach", id: string): void;
}>();

const store = useAnnotationsStore();
const chats = useChatsStore();
const anchorEl = ref<HTMLElement | null>(null);
const draft = ref("");
const sideTitle = ref("");
const sideMissing = ref(false);

const open = computed({
  get: () => props.modelValue,
  set: (v: boolean) => emit("update:modelValue", v),
});
const ann = computed(() => (props.annotation ? (store.byId[props.annotation.id] ?? props.annotation) : null));
const quote = computed(() => ann.value?.anchor?.exact?.trim() ?? "");

watch(
  () => [props.modelValue, props.annotation?.id] as const,
  async ([isOpen]) => {
    if (!isOpen || !ann.value) return;
    draft.value = ann.value.body_md;
    await loadSide();
  },
  { immediate: true },
);

async function loadSide(): Promise<void> {
  sideTitle.value = "";
  sideMissing.value = false;
  const id = ann.value?.side_chat_id;
  if (!id) return;
  const known = chats.byId.get(id);
  if (known) {
    sideTitle.value = known.title || "Side chat";
    return;
  }
  try {
    sideTitle.value = (await api().getChat(id)).title || "Side chat";
  } catch {
    sideMissing.value = true;
  }
}

async function saveDraft(): Promise<void> {
  const a = ann.value;
  if (!a || draft.value === a.body_md) return;
  await store.patch(a.id, { body_md: draft.value });
  void chats.refreshTags();
}

function onKey(ev: KeyboardEvent): void {
  if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
    ev.preventDefault();
    void saveDraft().then(() => (open.value = false));
  }
}

async function setColor(color: string): Promise<void> {
  if (ann.value) await store.patch(ann.value.id, { color });
}

async function editTags(): Promise<void> {
  const a = ann.value;
  if (!a) return;
  await saveDraft();
  const tags = await promptTags(a.tags);
  if (tags === null) return;
  await store.patch(a.id, { tags });
  void chats.refreshTags();
}

async function ask(ev: Event): Promise<void> {
  const a = ann.value;
  if (!a) return;
  const me = ev instanceof MouseEvent ? ev : null;
  await saveDraft();
  open.value = false;
  await askAboutAnnotation(a, { x: me?.clientX ?? props.x, y: me?.clientY ?? props.y }, props.chatId);
}

function openSide(ev: MouseEvent): void {
  const id = ann.value?.side_chat_id;
  if (!id) return;
  open.value = false;
  openChat(id, { where: "float", at: toWorkspace(ev.clientX + 12, ev.clientY + 8) });
}

async function unlink(): Promise<void> {
  if (ann.value) await store.patch(ann.value.id, { side_chat_id: null });
  sideMissing.value = false;
}

async function remove(): Promise<void> {
  const a = ann.value;
  if (!a) return;
  open.value = false;
  await deleteAnnotation(a.id);
}
</script>

<template>
  <div ref="anchorEl" class="ann-anchor" :style="{ left: `${x}px`, top: `${y}px` }">
    <q-menu
      v-model="open"
      no-parent-event
      :target="anchorEl ?? undefined"
      anchor="bottom left"
      self="top left"
      :offset="[0, 6]"
      @hide="saveDraft"
    >
      <div v-if="ann" class="ann-pop q-pa-sm" @keydown="onKey">
        <div class="row items-center no-wrap q-mb-xs">
          <span class="ann-swatch q-mr-sm" :class="`ann-c-${colorOf(ann.color)}`" />
          <span class="col text-caption ellipsis-2 ann-quote" :class="{ orphaned }">
            {{ quote ? `“${quote}”` : "Whole item" }}
          </span>
        </div>
        <q-banner v-if="orphaned" dense rounded class="ann-orphan-banner q-mb-xs text-caption">
          This text is no longer here. Select new text and Re-attach, or keep the note as is.
        </q-banner>
        <q-input
          v-model="draft"
          type="textarea"
          autogrow
          outlined
          dense
          autofocus
          placeholder="Add a note… (Ctrl+Enter to save)"
          class="ann-input"
          @blur="saveDraft"
        />
        <div class="row items-center q-mt-xs q-gutter-x-xs">
          <button
            v-for="c in ANNOTATION_COLORS"
            :key="c"
            type="button"
            class="ann-swatch ann-swatch-btn"
            :class="[`ann-c-${c}`, { selected: colorOf(ann.color) === c }]"
            :aria-label="`Colour ${c}`"
            @click="setColor(c)"
          />
          <q-space />
          <span v-for="t in ann.tags" :key="t" class="tag-chip cursor-pointer" @click="editTags">#{{ t }}</span>
          <q-btn flat dense no-caps size="sm" icon="sell" :label="ann.tags.length ? '' : 'Tags'" @click="editTags" />
        </div>
        <div v-if="ann.side_chat_id" class="row items-center no-wrap q-mt-xs ann-side">
          <q-icon name="forum" size="16px" class="text-muted q-mr-xs" />
          <template v-if="sideMissing">
            <span class="col text-caption text-muted">Side chat missing</span>
            <q-btn flat dense no-caps size="sm" label="Unlink" @click="unlink" />
          </template>
          <span v-else class="col text-caption ellipsis-1 cursor-pointer ann-side-link" @click="openSide">
            Side chat: {{ sideTitle || "…" }}
          </span>
        </div>
        <div class="row items-center q-mt-sm">
          <q-btn
            flat
            dense
            no-caps
            size="sm"
            icon="forum"
            :label="ann.side_chat_id && !sideMissing ? 'Open side chat' : 'Ask about this'"
            @click="ask"
          />
          <q-btn
            v-if="orphaned"
            flat
            dense
            no-caps
            size="sm"
            icon="link"
            label="Re-attach"
            @mousedown.prevent
            @click="emit('reattach', ann.id)"
          >
            <q-tooltip>Attach this note to the text selected now</q-tooltip>
          </q-btn>
          <q-space />
          <q-btn flat dense round size="sm" icon="delete_outline" aria-label="Delete annotation" @click="remove" />
        </div>
      </div>
    </q-menu>
  </div>
</template>

<style scoped>
.ann-anchor {
  position: fixed;
  width: 0;
  height: 0;
  pointer-events: none;
}
.ann-pop {
  width: 320px;
}
.ann-quote {
  color: var(--pawn-muted);
}
.ann-quote.orphaned {
  text-decoration: line-through;
}
.ann-orphan-banner {
  background: var(--pawn-chip);
}
.ann-swatch-btn {
  border: 2px solid transparent;
  cursor: pointer;
  padding: 0;
}
.ann-swatch-btn.selected {
  border-color: var(--pawn-muted);
}
.ann-side-link:hover {
  text-decoration: underline;
}
</style>
