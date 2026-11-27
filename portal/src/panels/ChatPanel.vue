<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { api, HttpError, type Chat, type ChatDetail, type MessageContext } from "@/api";
import { useAnnotationsStore } from "@/stores/annotations";
import { isSideChat, useChatsStore } from "@/stores/chats";
import { useComposerStore } from "@/stores/composer";
import { useFilesStore } from "@/stores/files";
import { useTabsStore } from "@/stores/tabs";
import { closeTab, openChat, openClip, openFile } from "@/router";
import { promptTags, promptText } from "@/lib/dialogs";
import { DND_FILE, hasType } from "@/lib/dnd";
import { errorText, isAbort } from "@/lib/errors";
import { vTarget, type Target } from "@/lib/targets";
import { toWorkspace } from "@/lib/workspace";
import MarkdownView from "@/components/MarkdownView.vue";
import AnnotationLayer from "@/components/AnnotationLayer.vue";

const props = defineProps<{ id: string; tabId: string }>();
const chats = useChatsStore();
const files = useFilesStore();
const tabs = useTabsStore();
const composer = useComposerStore();

interface UiMessage {
  key: string;
  id: string | null;
  role: string;
  content: string;
  context: MessageContext;
  pending?: boolean;
  progress?: string[];
  error?: string;
  note?: string;
}

let localSeq = 0;
const chat = ref<ChatDetail | null>(null);
const messages = ref<UiMessage[]>([]);
const loading = ref(true);
const loadError = ref("");
const draft = ref("");
/** Context notes live in the composer store so other panels can add to them. */
const contextNotes = computed<string[]>({
  get: () => composer.notesFor(props.id),
  set: (v) => {
    composer.clearNotes(props.id);
    v.forEach((p) => composer.addNote(props.id, p));
  },
});
const streaming = ref(false);
const dropping = ref(false);
const scroller = ref<HTMLElement | null>(null);
const input = ref<{ focus: () => void } | null>(null);
let abort: AbortController | null = null;

const modelOptions = computed(() =>
  (chats.models?.models ?? []).map((m) => ({ label: m.id, value: m.id })),
);

function toUi(m: ChatDetail["messages"][number]): UiMessage {
  return { key: m.id, id: m.id, role: m.role, content: m.content, context: m.context ?? {} };
}

async function load(): Promise<void> {
  loading.value = true;
  loadError.value = "";
  try {
    chat.value = await api().getChat(props.id);
    messages.value = chat.value.messages.map(toUi);
    tabs.setStatus(props.tabId, { title: chat.value.title || "New chat" });
    await scrollToBottom();
  } catch (e) {
    loadError.value = e instanceof HttpError && e.status === 404 ? "This chat no longer exists." : errorText(e);
  } finally {
    loading.value = false;
  }
}

/** Header fields only: never clobber messages mid-stream. */
async function refreshMeta(): Promise<void> {
  try {
    const fresh = await api().getChat(props.id);
    if (chat.value) Object.assign(chat.value, { ...fresh, messages: chat.value.messages });
    tabs.setStatus(props.tabId, { title: fresh.title || "New chat" });
  } catch {
    /* deleted or offline: the sidebar reports it */
  }
}

async function scrollToBottom(): Promise<void> {
  await nextTick();
  const el = scroller.value;
  if (el) el.scrollTop = el.scrollHeight;
}

function nearBottom(): boolean {
  const el = scroller.value;
  return !el || el.scrollHeight - el.scrollTop - el.clientHeight < 120;
}

async function send(): Promise<void> {
  const text = draft.value.trim();
  if (!text || streaming.value) return;
  const notes = [...contextNotes.value];
  draft.value = "";
  contextNotes.value = [];
  const user: UiMessage = {
    key: `local-${localSeq++}`,
    id: null,
    role: "user",
    content: text,
    context: notes.length ? { notes } : {},
  };
  const reply: UiMessage = {
    key: `local-${localSeq++}`,
    id: null,
    role: "assistant",
    content: "",
    context: {},
    pending: true,
    progress: [],
  };
  messages.value.push(user, reply);
  // Reactive proxies of the pushed rows, so handler writes re-render.
  const u = messages.value[messages.value.length - 2];
  const r = messages.value[messages.value.length - 1];
  await scrollToBottom();

  tabs.pin(props.tabId);
  streaming.value = true;
  tabs.setStatus(props.tabId, { busy: true });
  const ctl = new AbortController();
  abort = ctl;
  // The composer unlocks once the answer is stored; the stream may stay open
  // a little longer while the chat gets its automatic title and tags.
  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    r.pending = false;
    if (abort === ctl) {
      streaming.value = false;
      abort = null;
    }
    tabs.setStatus(props.tabId, { busy: false });
    if (!r.content && !r.error && !r.note) r.note = "No answer.";
    chats.touched();
    void scrollToBottom();
    input.value?.focus();
  };
  try {
    await api().sendMessage(
      props.id,
      {
        message: text,
        context: notes.map((path) => ({ path, inline: false })),
        model: chats.model || null,
      },
      {
        onMessage: (m) => {
          if (m.role === "user") u.id = m.id;
          else {
            r.id = m.id;
            release();
          }
        },
        onChat: (c) => {
          if (chat.value) {
            chat.value.title = c.title;
            chat.value.tags = c.tags;
          }
          tabs.setStatus(props.tabId, { title: c.title || "New chat" });
          chats.touched();
          void chats.refreshTags();
        },
        onProgress: (t) => {
          if (!t) return;
          const stick = nearBottom();
          r.progress = [...(r.progress ?? []), t];
          if (stick) void scrollToBottom();
        },
        onAnswer: (content) => {
          const stick = nearBottom();
          r.content = content;
          if (stick) void scrollToBottom();
        },
        onFiles: (paths) => {
          r.context = { ...r.context, written_files: paths };
          files.noteWritten(paths);
        },
        onJob: () => {
          r.note = "Queued as a background job; the answer will land in the jobs list.";
        },
        onError: (message) => {
          r.error = message;
        },
      },
      ctl.signal,
    );
  } catch (e) {
    if (!released) {
      if (isAbort(e)) r.note = "Stopped.";
      else r.error = errorText(e);
    }
  } finally {
    release();
    void refreshMeta();
  }
}

function stop(): void {
  abort?.abort();
}

function onKeydown(ev: KeyboardEvent): void {
  if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
    ev.preventDefault();
    void send();
  }
}

// ── header actions ───────────────────────────────────────────────────────
async function rename(): Promise<void> {
  if (!chat.value) return;
  const title = await promptText("Rename chat", chat.value.title);
  if (title !== null && (await chats.patchChat(props.id, { title }))) {
    chat.value.title = title;
    chat.value.title_auto = false;
    tabs.setStatus(props.tabId, { title });
  }
}

async function editTags(): Promise<void> {
  if (!chat.value) return;
  const tags = await promptTags(chat.value.tags);
  if (tags === null) return;
  const updated = await chats.patchChat(props.id, { tags });
  if (updated) {
    chat.value.tags = updated.tags;
    chat.value.tags_auto = false;
  }
}

async function togglePin(): Promise<void> {
  if (!chat.value) return;
  const updated = await chats.patchChat(props.id, { pinned: !chat.value.pinned });
  if (updated) chat.value.pinned = updated.pinned;
}

async function toggleArchive(): Promise<void> {
  if (!chat.value) return;
  const updated = await chats.patchChat(props.id, { archived: !chat.value.archived });
  if (updated) chat.value.archived = updated.archived;
}

// ── context notes (drag a file from the files sidebar) ───────────────────
function onDragOver(ev: DragEvent): void {
  if (!hasType(ev, DND_FILE)) return;
  ev.preventDefault();
  dropping.value = true;
}

function onDrop(ev: DragEvent): void {
  dropping.value = false;
  const path = ev.dataTransfer?.getData(DND_FILE);
  if (!path) return;
  ev.preventDefault();
  composer.addNote(props.id, path);
}

function openWritten(path: string): void {
  files.markSeen(path);
  openFile(path);
}

function baseName(path: string): string {
  return path.split("/").pop() ?? path;
}

function copy(text: string): void {
  void navigator.clipboard?.writeText(text).catch(() => {});
}

watch(
  () => chats.revision,
  () => void refreshMeta(),
);

// ── right-click layer: message targets, side chats, focus requests ──────
function excerpt(text: string, n = 48): string {
  const one = text.replace(/[*_`#>]+/g, "").replace(/\s+/g, " ").trim();
  return one.length > n ? one.slice(0, n - 1) + "…" : one;
}

// Annotations on every saved message of this chat (one batched load).
const annotations = useAnnotationsStore();
const annotatedIds = computed(() => messages.value.filter((m) => m.id && !m.pending).map((m) => m.id as string));
function messageRoot(id: string): HTMLElement | null {
  return scroller.value?.querySelector<HTMLElement>(`[data-mid="${CSS.escape(id)}"]`) ?? null;
}

function messageTarget(m: UiMessage): Target | null {
  if (!m.id || m.pending) return null;
  return {
    kind: "message",
    ref: m.id,
    chatId: props.id,
    role: m.role,
    label: `${m.role === "user" ? "You" : "Pawn"}: ${excerpt(m.content)}`,
    text: m.content,
  };
}

/** Unpromoted side chats of this chat, keyed by the message they came from. */
const sideByMessage = computed(() => {
  const out = new Map<string, Chat[]>();
  for (const c of chat.value?.side_chats ?? []) {
    if (!isSideChat(c) || !c.origin) continue;
    const ids =
      c.origin.kind === "messages" && Array.isArray((c.origin.anchor as { message_ids?: string[] })?.message_ids)
        ? (c.origin.anchor as { message_ids: string[] }).message_ids
        : [c.origin.ref];
    for (const id of ids) out.set(id, [...(out.get(id) ?? []), c]);
  }
  return out;
});

function openSide(c: Chat, ev?: Event): void {
  const me = ev instanceof MouseEvent ? ev : null;
  openChat(c.id, { where: "float", at: me ? toWorkspace(me.clientX + 12, me.clientY + 8) : undefined });
}

function sideTitle(c: Chat): string {
  return c.title || (c.origin?.quote ? `“${excerpt(c.origin.quote, 40)}”` : "Side chat");
}

// This chat as a side chat: where it came from.
const origin = computed(() => chat.value?.origin ?? null);
function openOrigin(): void {
  const o = origin.value;
  if (!o) return;
  if (o.kind === "message" || o.kind === "messages") {
    const src = o.chat_id || chat.value?.parent_chat_id;
    if (!src) return;
    openChat(src);
    chats.requestFocus(src, o.ref);
  } else if (o.kind === "file") openFile(o.ref);
  else if (o.kind === "clip") openClip(o.ref);
  else if (o.kind === "chat") openChat(o.ref);
  // Asked from an annotation: jump to its highlight too.
  if (o.annotation_id) annotations.requestFocus(o.annotation_id);
}

async function promote(): Promise<void> {
  const updated = await chats.patchChat(props.id, { promoted: true });
  if (updated && chat.value) chat.value.origin = updated.origin;
}

const flashId = ref("");
watch(
  () => chats.focusRequest,
  async (req) => {
    if (!req || req.chatId !== props.id) return;
    if (loading.value) await load();
    await nextTick();
    const el = scroller.value?.querySelector<HTMLElement>(`[data-mid="${req.messageId}"]`);
    if (!el) return;
    el.scrollIntoView({ block: "center", behavior: "smooth" });
    flashId.value = req.messageId;
    setTimeout(() => {
      if (flashId.value === req.messageId) flashId.value = "";
    }, 1600);
  },
);

// Quotes sent here from elsewhere ("Quote in chat") land in the draft.
watch(
  () => composer.quotes[props.id],
  (q) => {
    if (!q?.length) return;
    const block = composer
      .takeQuotes(props.id)
      .map((t) => t.trim().split("\n").map((l) => `> ${l}`).join("\n"))
      .join("\n\n");
    draft.value = draft.value.trim() ? `${draft.value.trimEnd()}\n\n${block}\n\n` : `${block}\n\n`;
    tabs.pin(props.tabId);
    void nextTick(() => input.value?.focus());
  },
  { immediate: true },
);

onMounted(async () => {
  await load();
  input.value?.focus();
});
onBeforeUnmount(() => abort?.abort());
</script>

<template>
  <div class="column no-wrap full-height">
    <div v-if="loading" class="col flex flex-center"><q-spinner size="32px" /></div>
    <div v-else-if="loadError || !chat" class="col flex flex-center column text-muted">
      <q-icon name="error_outline" size="40px" />
      <div class="q-mt-sm">{{ loadError }}</div>
      <q-btn flat no-caps label="Close tab" class="q-mt-sm" @click="closeTab(tabId)" />
    </div>

    <template v-else>
      <div class="row items-center no-wrap q-px-md q-py-sm border-b">
        <div class="col" style="min-width: 0">
          <div class="text-subtitle1 ellipsis-1 cursor-pointer" @click="rename">
            {{ chat.title || "New chat" }}
            <q-icon v-if="chat.title_auto && messages.length" name="auto_awesome" size="14px" class="text-muted q-ml-xs" />
            <q-tooltip>
              {{ chat.title_auto ? "Named automatically. Click to rename (keeps your title)." : "Rename" }}
            </q-tooltip>
          </div>
          <div class="row items-center q-gutter-x-xs">
            <q-chip
              v-for="t in chat.tags"
              :key="t"
              dense
              size="sm"
              icon="sell"
              :label="t"
              clickable
              class="q-ml-none"
              @click="editTags"
            />
            <q-btn v-if="!chat.tags.length" flat dense no-caps size="sm" icon="sell" label="Add tags" @click="editTags" />
            <q-badge v-if="chat.archived" outline color="grey" label="archived" />
          </div>
        </div>
        <q-btn flat dense round icon="push_pin" :color="chat.pinned ? 'primary' : undefined" @click="togglePin">
          <q-tooltip>{{ chat.pinned ? "Unpin" : "Pin" }}</q-tooltip>
        </q-btn>
        <q-btn flat dense round :icon="chat.archived ? 'unarchive' : 'archive'" @click="toggleArchive">
          <q-tooltip>{{ chat.archived ? "Unarchive" : "Archive" }}</q-tooltip>
        </q-btn>
      </div>

      <div v-if="origin" class="origin-bar row items-center no-wrap q-px-md q-py-xs border-b">
        <q-icon :name="origin.kind === 'file' ? 'find_in_page' : 'question_answer'" size="16px" class="text-muted q-mr-sm" />
        <span class="text-caption text-muted q-mr-xs">{{ origin.promoted ? "From" : "Asked about" }}</span>
        <span class="col ellipsis-1 text-caption origin-quote cursor-pointer" @click="openOrigin">
          {{ origin.quote ? `“${origin.quote}”` : origin.ref }}
          <q-tooltip>Open the source</q-tooltip>
        </span>
        <q-btn v-if="!origin.promoted" flat dense no-caps size="sm" icon="upgrade" label="Promote" @click="promote">
          <q-tooltip>Make this a normal chat (keeps the link to its source)</q-tooltip>
        </q-btn>
      </div>

      <div ref="scroller" class="col scroll q-px-md q-py-md">
        <div class="thread">
          <AnnotationLayer
            v-if="annotatedIds.length"
            kind="message"
            mode="inline"
            :refs="annotatedIds"
            :root-for="messageRoot"
            :chat-id="id"
          />
          <div v-if="!messages.length" class="text-center text-muted q-mt-xl">
            <q-icon name="forum" size="40px" />
            <div class="q-mt-sm">Ask Pawn anything. Drag files from the right sidebar to add them as context.</div>
          </div>

          <div
            v-for="m in messages"
            :key="m.key"
            v-target="messageTarget(m)"
            class="msg q-mb-md"
            :class="[`msg-${m.role}`, { flash: m.id && m.id === flashId }]"
            :data-mid="m.id ?? undefined"
          >
            <div v-if="m.role === 'user'" class="bubble-user">
              <div class="user-text" data-pawn-anchor-root>{{ m.content }}</div>
              <div v-if="m.context.notes?.length" class="q-mt-xs">
                <q-chip
                  v-for="p in m.context.notes"
                  :key="p"
                  dense
                  size="sm"
                  icon="attach_file"
                  :label="baseName(p)"
                  clickable
                  @click="openWritten(p)"
                />
              </div>
            </div>

            <div v-else class="bubble-assistant">
              <div v-if="m.pending && !m.content" class="progress text-muted">
                <div v-for="(p, i) in m.progress" :key="i" class="text-caption">
                  <q-icon name="chevron_right" size="14px" /> {{ p }}
                </div>
                <q-spinner-dots size="22px" class="q-mt-xs" />
              </div>
              <details v-else-if="m.progress?.length" class="text-caption text-muted q-mb-xs">
                <summary>{{ m.progress.length }} steps</summary>
                <div v-for="(p, i) in m.progress" :key="i">· {{ p }}</div>
              </details>

              <MarkdownView v-if="m.content" :source="m.content" />
              <q-banner v-if="m.error" dense rounded class="bg-red-1 text-negative q-mt-xs">
                <template #avatar><q-icon name="error_outline" /></template>
                {{ m.error }}
              </q-banner>
              <div v-if="m.note" class="text-caption text-muted q-mt-xs">{{ m.note }}</div>

              <div v-if="m.context.written_files?.length" class="q-mt-sm row items-center q-gutter-xs">
                <span class="text-caption text-muted">Wrote</span>
                <q-chip
                  v-for="p in m.context.written_files"
                  :key="p"
                  dense
                  size="sm"
                  icon="description"
                  clickable
                  color="blue-1"
                  text-color="primary"
                  @click="openWritten(p)"
                >
                  {{ baseName(p) }}
                  <span v-if="files.isUnseen(p)" class="unseen-dot q-ml-xs" />
                  <q-tooltip>{{ p }}</q-tooltip>
                </q-chip>
              </div>

              <div v-if="m.content && !m.pending" class="msg-actions">
                <q-btn flat dense round size="xs" icon="content_copy" aria-label="Copy" @click="copy(m.content)">
                  <q-tooltip>Copy Markdown</q-tooltip>
                </q-btn>
                <span class="text-caption text-muted q-ml-xs">Right-click or select text for more</span>
              </div>
            </div>
            <div v-if="m.id && sideByMessage.get(m.id)?.length" class="side-badges" :class="{ 'justify-end': m.role === 'user' }">
              <q-chip
                v-for="c in sideByMessage.get(m.id)"
                :key="c.id"
                dense
                size="sm"
                icon="question_answer"
                clickable
                class="side-chip"
                @click="openSide(c, $event)"
              >
                <span class="ellipsis-1" style="max-width: 220px">{{ sideTitle(c) }}</span>
                <q-tooltip>Side chat · click to open</q-tooltip>
              </q-chip>
            </div>
          </div>
        </div>
      </div>

      <div
        class="composer q-px-md q-pb-md q-pt-sm border-t"
        :class="{ 'drop-target': dropping }"
        @dragover="onDragOver"
        @dragleave="dropping = false"
        @drop="onDrop"
      >
        <div class="thread">
          <div v-if="contextNotes.length" class="q-mb-xs">
            <q-chip
              v-for="p in contextNotes"
              :key="p"
              dense
              size="sm"
              icon="attach_file"
              removable
              :label="baseName(p)"
              @remove="contextNotes = contextNotes.filter((x) => x !== p)"
            >
              <q-tooltip>{{ p }}</q-tooltip>
            </q-chip>
          </div>
          <q-input
            ref="input"
            v-model="draft"
            type="textarea"
            autogrow
            outlined
            dense
            placeholder="Message Pawn (Enter to send, Shift+Enter for a new line)"
            input-style="max-height: 200px"
            @keydown="onKeydown"
          >
            <template #after>
              <q-btn v-if="streaming" round unelevated color="negative" icon="stop" aria-label="Stop" @click="stop" />
              <q-btn
                v-else
                round
                unelevated
                color="primary"
                icon="send"
                aria-label="Send"
                :disable="!draft.trim()"
                @click="send"
              />
            </template>
          </q-input>
          <div class="row items-center q-mt-xs">
            <q-select
              :model-value="chats.model"
              :options="modelOptions"
              emit-value
              map-options
              dense
              borderless
              options-dense
              class="text-caption model-select"
              @update:model-value="chats.setModel"
            >
              <template #prepend><q-icon name="smart_toy" size="16px" /></template>
            </q-select>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.thread {
  position: relative;
  max-width: 820px;
  margin: 0 auto;
}
.bubble-user {
  margin-left: auto;
  max-width: 85%;
  width: fit-content;
  background: var(--pawn-bubble-user);
  border-radius: 12px;
  padding: 8px 12px;
}
.user-text {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.bubble-assistant {
  position: relative;
  padding: 2px 0;
}
.msg-actions {
  opacity: 0;
  transition: opacity 0.15s;
}
.msg:hover .msg-actions {
  opacity: 1;
}
details summary {
  cursor: pointer;
  user-select: none;
}
.progress {
  user-select: none;
}
.side-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
  margin-top: 2px;
}
.side-chip {
  background: var(--pawn-chip);
}
.origin-bar {
  background: var(--pawn-tabbar);
}
.origin-quote:hover {
  text-decoration: underline;
}
.msg {
  border-radius: 8px;
  transition: box-shadow 0.3s, background 0.3s;
}
.msg.flash {
  background: var(--pawn-drop);
  box-shadow: 0 0 0 4px var(--pawn-drop);
}
.model-select {
  min-width: 200px;
}
</style>
