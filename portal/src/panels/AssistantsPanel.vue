<script setup lang="ts">
// Voice assistants (Linux laptop, Android phone): presence, voice threads,
// live commands, phone-approved access and shared wake-word training.
// Server side: docs/ASSISTANTS.md.
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Notify, copyToClipboard } from "quasar";
import {
  api,
  type AssistantKind,
  type AssistantRegistration,
  type AssistantSummary,
  type AssistantsOverview,
} from "@/api";
import { confirmAction, promptText } from "@/lib/dialogs";
import { notifyError } from "@/lib/errors";
import { fullTime, relativeTime } from "@/lib/time";
import {
  CATEGORY_LABEL,
  PRESENCE_LABEL,
  clipTotal,
  commandSummary,
  kindIcon,
  linuxConfigSnippet,
  presenceFlags,
  presenceLevel,
  toolLabel,
} from "@/lib/assistants";
import { openChat } from "@/router";
import { useTabsStore } from "@/stores/tabs";
import StatusBadge from "@/components/StatusBadge.vue";

const props = defineProps<{ tabId: string }>();
const tabs = useTabsStore();

const WAKE = "__wake";
const data = ref<AssistantsOverview | null>(null);
const selected = ref("");
const cursor = ref(0);
const loading = ref(false);
let timer: ReturnType<typeof setInterval> | null = null;

const rows = computed(() => data.value?.assistants ?? []);
const fresh = computed(() => data.value?.presence_fresh_s ?? 600);
const current = computed<AssistantSummary | null>(
  () => rows.value.find((r) => r.device_id === selected.value) ?? null,
);
const wake = computed(() => data.value?.wake ?? null);

async function reload(quiet = false): Promise<void> {
  if (!quiet) loading.value = true;
  try {
    data.value = await api().assistantsOverview();
    if (!selected.value && rows.value.length) selected.value = rows.value[0].device_id;
    if (selected.value && selected.value !== WAKE && !current.value) selected.value = rows.value[0]?.device_id ?? "";
  } catch (e) {
    if (!quiet) notifyError(e, "Assistants");
  } finally {
    loading.value = false;
  }
}

function level(a: AssistantSummary) {
  return presenceLevel(a, fresh.value);
}

// ── Register ──────────────────────────────────────────────────────────────────

const registerOpen = ref(false);
const regName = ref("");
const regKind = ref<AssistantKind>("desktop");
const registered = ref<AssistantRegistration | null>(null);
const serverUrl = computed(() => window.location.origin);

function openRegister(): void {
  regName.value = "";
  regKind.value = "desktop";
  registered.value = null;
  registerOpen.value = true;
}

async function doRegister(): Promise<void> {
  try {
    registered.value = await api().registerAssistant(regName.value.trim(), regKind.value);
    await reload(true);
    selected.value = registered.value.device_id;
  } catch (e) {
    notifyError(e, "Register");
  }
}

async function copy(text: string): Promise<void> {
  try {
    await copyToClipboard(text);
    Notify.create({ message: "Copied", position: "bottom-right", timeout: 1200 });
  } catch (e) {
    notifyError(e, "Copy");
  }
}

// ── Detail actions ────────────────────────────────────────────────────────────

async function rename(a: AssistantSummary): Promise<void> {
  const name = await promptText("Rename assistant", a.name);
  if (!name || name === a.name) return;
  try {
    await api().renameAssistant(a.device_id, name);
    await reload(true);
  } catch (e) {
    notifyError(e, "Rename");
  }
}

async function revoke(a: AssistantSummary): Promise<void> {
  const ok = await confirmAction(
    `Revoke ${a.name}?`,
    "Its token stops working and the server deletes what it shared (events, commands). Wake clips are kept as training data.",
    "Revoke",
  );
  if (!ok) return;
  try {
    await api().revokeAssistant(a.device_id);
    selected.value = "";
    await reload(true);
  } catch (e) {
    notifyError(e, "Revoke");
  }
}

const TESTABLE = ["speak", "notify", "timer_set", "status_now", "location_now"];
const testOpen = ref(false);
const testTool = ref("speak");
const testText = ref("");
const testSeconds = ref(60);
const testRunning = ref(false);
const testTools = computed(() => (current.value?.capabilities ?? []).filter((t) => TESTABLE.includes(t)));

function openTest(): void {
  testTool.value = testTools.value[0] ?? "speak";
  testText.value = "Ciao, sono Pawn.";
  testSeconds.value = 60;
  testOpen.value = true;
}

async function runTest(): Promise<void> {
  const a = current.value;
  if (!a) return;
  const args: Record<string, string> = {};
  if (testTool.value === "speak" || testTool.value === "notify") args.text = testText.value;
  if (testTool.value === "timer_set") {
    args.seconds = String(testSeconds.value);
    if (testText.value) args.label = testText.value;
  }
  testRunning.value = true;
  try {
    const res = await api().runAssistantCommand(a.device_id, testTool.value, args);
    Notify.create({
      message: res.text,
      type: res.status === "ok" ? "positive" : "warning",
      position: "bottom-right",
    });
    testOpen.value = false;
    await reload(true);
  } catch (e) {
    notifyError(e, toolLabel(testTool.value));
  } finally {
    testRunning.value = false;
  }
}

// ── Wake models ───────────────────────────────────────────────────────────────

const fileEl = ref<HTMLInputElement | null>(null);
const uploading = ref(false);

async function onModelFile(ev: Event): Promise<void> {
  const input = ev.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  const notes = await promptText("Model notes", "", { message: `Publishing ${file.name}`, allowEmpty: true });
  if (notes === null) return;
  uploading.value = true;
  try {
    const model = await api().publishWakeModel(file, { notes });
    Notify.create({ message: `Published ${model.version}`, position: "bottom-right" });
    await reload(true);
  } catch (e) {
    notifyError(e, "Publish model");
  } finally {
    uploading.value = false;
  }
}

async function activate(version: string): Promise<void> {
  try {
    await api().activateWakeModel(version);
    await reload(true);
  } catch (e) {
    notifyError(e, "Activate model");
  }
}

// ── Keyboard ──────────────────────────────────────────────────────────────────

const ids = computed(() => [...rows.value.map((r) => r.device_id), WAKE]);

function onKey(ev: KeyboardEvent): void {
  const target = ev.target as HTMLElement | null;
  if (target && ["INPUT", "TEXTAREA"].includes(target.tagName)) return;
  if (ev.key === "j" || ev.key === "ArrowDown") cursor.value = Math.min(cursor.value + 1, ids.value.length - 1);
  else if (ev.key === "k" || ev.key === "ArrowUp") cursor.value = Math.max(cursor.value - 1, 0);
  else if (ev.key === "Enter") selected.value = ids.value[cursor.value] ?? "";
  else if (ev.key === "r") void reload();
  else return;
  ev.preventDefault();
}

watch(
  () => rows.value.length,
  (n) => tabs.setStatus(props.tabId, { title: n ? `Assistants (${n})` : "Assistants" }),
);

onMounted(() => {
  void reload();
  timer = setInterval(() => void reload(true), 15_000);
});
onBeforeUnmount(() => {
  if (timer) clearInterval(timer);
});
</script>

<template>
  <div class="assistants column no-wrap full-height" tabindex="0" @keydown="onKey">
    <div class="row items-center q-pa-xs q-gutter-xs border-b">
      <div class="col text-subtitle2 q-pl-sm">Voice assistants</div>
      <q-btn dense no-caps outline icon="add" label="Register" @click="openRegister" />
      <q-btn flat dense round icon="refresh" aria-label="Refresh" :loading="loading" @click="reload()" />
    </div>

    <div class="row col no-wrap body">
      <aside class="list col-auto pawn-scroll">
        <button
          v-for="(row, index) in rows"
          :key="row.device_id"
          type="button"
          class="entry"
          :class="{ picked: row.device_id === selected, cursor: index === cursor }"
          @click="selected = row.device_id"
        >
          <q-icon :name="kindIcon(row.kind)" size="22px" />
          <span class="col meta">
            <span class="ellipsis">{{ row.name }}</span>
            <span class="text-caption text-muted ellipsis">
              {{ PRESENCE_LABEL[level(row)] }} · seen {{ relativeTime(row.last_seen) }}
            </span>
          </span>
          <span class="dot" :class="level(row)" :title="PRESENCE_LABEL[level(row)]" />
        </button>
        <div v-if="!rows.length && !loading" class="q-pa-md text-muted">
          No assistant is registered yet. Register the laptop here, or the phone from the app (Settings → Pawn server).
        </div>
        <button
          type="button"
          class="entry"
          :class="{ picked: selected === WAKE, cursor: cursor === rows.length }"
          @click="selected = WAKE"
        >
          <q-icon name="graphic_eq" size="22px" />
          <span class="col meta">
            <span>Wake word</span>
            <span class="text-caption text-muted ellipsis">
              {{ wake?.model ? wake.model.version : "no model published" }} ·
              {{ clipTotal(wake?.clips ?? {}) }} clips
            </span>
          </span>
        </button>
      </aside>

      <!-- Wake word -->
      <section v-if="selected === WAKE && wake" class="detail col pawn-scroll q-pa-md">
        <div class="text-h6">Wake word</div>
        <p class="text-muted q-mb-md">
          Assistants upload the clip that triggered each wake, labelled by what happened next. Train on the
          laptop with <code>pawn-assistant wake-train pull</code>, then <code>wake-train publish</code>; every
          assistant downloads the active model.
        </p>
        <div class="text-subtitle2 q-mb-xs">Clips on the server</div>
        <div class="row q-gutter-xs q-mb-md">
          <q-chip v-for="(n, label) in wake.clips" :key="label" dense square>{{ label }} · {{ n }}</q-chip>
          <span v-if="!clipTotal(wake.clips)" class="text-muted">None yet.</span>
        </div>
        <div class="row items-center q-mb-xs">
          <div class="col text-subtitle2">Models</div>
          <q-btn dense no-caps outline icon="upload" label="Publish .onnx" :loading="uploading" @click="fileEl?.click()" />
          <input ref="fileEl" type="file" accept=".onnx" class="hidden" @change="onModelFile" />
        </div>
        <q-list bordered separator dense>
          <q-item v-for="m in wake.models" :key="m.version">
            <q-item-section>
              <q-item-label>
                {{ m.version }}
                <StatusBadge v-if="m.active" kind="ok" icon="check" label="active" />
              </q-item-label>
              <q-item-label caption>
                {{ fullTime(m.created_at) }} · {{ Math.round(m.size / 1024) }} KB · sha {{ m.sha256.slice(0, 10) }}
                <template v-if="m.thresholds.wake"> · wake {{ m.thresholds.wake }}</template>
              </q-item-label>
              <q-item-label v-if="m.notes" caption>{{ m.notes }}</q-item-label>
            </q-item-section>
            <q-item-section side>
              <q-btn v-if="!m.active" flat dense no-caps label="Activate" @click="activate(m.version)" />
            </q-item-section>
          </q-item>
          <q-item v-if="!wake.models.length"><q-item-section class="text-muted">No model published.</q-item-section></q-item>
        </q-list>
      </section>

      <!-- One assistant -->
      <section v-else-if="current" class="detail col pawn-scroll q-pa-md">
        <div class="row items-center q-gutter-sm q-mb-sm">
          <q-icon :name="kindIcon(current.kind)" size="28px" />
          <div class="text-h6">{{ current.name }}</div>
          <q-btn flat dense round icon="edit" aria-label="Rename" @click="rename(current)" />
          <q-space />
          <q-btn dense no-caps outline icon="campaign" label="Send test" :disable="!testTools.length" @click="openTest" />
          <q-btn dense no-caps flat color="negative" icon="delete" label="Revoke" @click="revoke(current)" />
        </div>

        <div class="facts q-mb-md">
          <span class="text-muted">Kind</span>
          <span>{{ current.kind }}<template v-if="current.platform"> · {{ current.platform }}</template></span>
          <span class="text-muted">Presence</span>
          <span>
            <span class="dot inline" :class="level(current)" /> {{ PRESENCE_LABEL[level(current)] }}
            <template v-if="current.presence_at"> · reported {{ relativeTime(current.presence_at) }}</template>
            <q-chip v-for="f in presenceFlags(current)" :key="f" dense square class="q-ml-xs">{{ f }}</q-chip>
          </span>
          <span class="text-muted">Last wake</span>
          <span>{{ current.presence.last_wake_at ? relativeTime(current.presence.last_wake_at) : "—" }}</span>
          <span class="text-muted">Commands via</span>
          <span>
            {{ current.kind === "desktop" ? "command stream" : current.unifiedpush ? "UnifiedPush" : "no push channel yet" }}
          </span>
          <span class="text-muted">Registered</span>
          <span>{{ fullTime(current.created_at) }} · id <code>{{ current.device_id }}</code></span>
        </div>

        <div class="text-subtitle2 q-mb-xs">Can run</div>
        <div class="row q-gutter-xs q-mb-md">
          <q-chip v-for="t in current.capabilities" :key="t" dense square>{{ toolLabel(t) }}</q-chip>
          <span v-if="!current.capabilities.length" class="text-muted">Nothing (turned on in the assistant's settings).</span>
        </div>

        <template v-if="current.kind === 'phone'">
          <div class="text-subtitle2 q-mb-xs">Shares with pawn-server</div>
          <div class="row q-gutter-xs q-mb-md">
            <q-chip v-for="c in current.enabled_categories" :key="c" dense square>{{ CATEGORY_LABEL[c] ?? c }}</q-chip>
            <span v-if="!current.enabled_categories.length" class="text-muted">Nothing. Toggles live on the phone.</span>
          </div>
        </template>

        <div class="text-subtitle2 q-mb-xs">Voice conversation</div>
        <div v-if="current.thread" class="thread q-mb-md">
          <div class="text-caption text-muted">
            {{ current.thread.turns }} turns · last {{ relativeTime(current.thread.last_turn_at) }} ·
            access {{ current.thread.access }} · {{ current.thread.lang }}
          </div>
          <div v-if="current.thread.last_user" class="q-mt-xs"><b>You:</b> {{ current.thread.last_user }}</div>
          <div v-if="current.thread.last_answer"><b>Pawn:</b> {{ current.thread.last_answer }}</div>
          <q-btn
            v-if="current.thread.portal_chat_id"
            class="q-mt-xs"
            flat
            dense
            no-caps
            icon="chat_bubble_outline"
            label="Open in chat"
            @click="openChat(current.thread.portal_chat_id!)"
          />
        </div>
        <div v-else class="text-muted q-mb-md">
          None active. A conversation started elsewhere can be taken over by saying "continua qui".
        </div>

        <template v-if="current.access_requests.length">
          <div class="text-subtitle2 q-mb-xs">Full-access requests</div>
          <q-list bordered separator dense class="q-mb-md">
            <q-item v-for="r in current.access_requests" :key="r.id">
              <q-item-section>
                <q-item-label>
                  {{ r.requester_device_id === current.device_id ? `asked ${r.approver_name ?? "the phone"}` : "approval for another assistant" }}
                  · {{ r.minutes }} min
                </q-item-label>
                <q-item-label caption>
                  {{ relativeTime(r.created_at) }}
                  <template v-if="r.access_until && r.status === 'approved'"> · until {{ fullTime(r.access_until) }}</template>
                  <template v-if="r.message"> · {{ r.message }}</template>
                </q-item-label>
              </q-item-section>
              <q-item-section side><span class="status" :class="r.status">{{ r.status }}</span></q-item-section>
            </q-item>
          </q-list>
        </template>

        <div class="text-subtitle2 q-mb-xs">Recent commands</div>
        <q-list bordered separator dense class="q-mb-md">
          <q-item v-for="c in current.commands" :key="c.id">
            <q-item-section>
              <q-item-label>
                {{ toolLabel(c.tool) }}
                <q-icon v-if="c.action" name="fingerprint" size="14px" class="q-ml-xs"><q-tooltip>Confirmed on the phone</q-tooltip></q-icon>
                <span class="text-muted"> {{ commandSummary(c) }}</span>
              </q-item-label>
              <q-item-label caption>{{ relativeTime(c.created_at) }}</q-item-label>
            </q-item-section>
            <q-item-section side><span class="status" :class="c.status">{{ c.status }}</span></q-item-section>
          </q-item>
          <q-item v-if="!current.commands.length"><q-item-section class="text-muted">No commands yet.</q-item-section></q-item>
        </q-list>

        <div class="text-subtitle2 q-mb-xs">Wake clips uploaded</div>
        <div class="row q-gutter-xs">
          <q-chip v-for="(n, label) in current.wake_clips" :key="label" dense square>{{ label }} · {{ n }}</q-chip>
          <span v-if="!clipTotal(current.wake_clips)" class="text-muted">None (opt-in on the assistant).</span>
        </div>
      </section>

      <section v-else class="detail col flex flex-center text-muted">Select an assistant</section>
    </div>

    <!-- Register -->
    <q-dialog v-model="registerOpen">
      <q-card style="min-width: 420px; max-width: 640px">
        <q-card-section class="text-h6">Register an assistant</q-card-section>
        <q-card-section v-if="!registered" class="q-gutter-sm">
          <q-input v-model="regName" dense outlined autofocus label="Name" placeholder="laptop" />
          <q-btn-toggle
            v-model="regKind"
            dense
            no-caps
            toggle-color="primary"
            :options="[
              { label: 'Laptop (Linux)', value: 'desktop' },
              { label: 'Phone', value: 'phone' },
            ]"
          />
          <div class="text-caption text-muted">
            The phone usually registers itself from the app (Settings → Pawn server → Register phone).
          </div>
        </q-card-section>
        <q-card-section v-else class="q-gutter-sm">
          <div>
            Registered <b>{{ registered.name }}</b>. The token is shown only now.
          </div>
          <q-input :model-value="registered.device_token" dense outlined readonly label="Device token">
            <template #append>
              <q-btn flat dense round icon="content_copy" aria-label="Copy token" @click="copy(registered.device_token)" />
            </template>
          </q-input>
          <template v-if="registered.kind === 'desktop'">
            <div class="text-caption text-muted">On the laptop:</div>
            <pre class="snippet">{{ linuxConfigSnippet(serverUrl, registered.device_id, registered.device_token) }}</pre>
            <q-btn
              flat
              dense
              no-caps
              icon="content_copy"
              label="Copy snippet"
              @click="copy(linuxConfigSnippet(serverUrl, registered.device_id, registered.device_token))"
            />
          </template>
        </q-card-section>
        <q-card-actions align="right">
          <q-btn v-if="!registered" flat no-caps label="Cancel" v-close-popup />
          <q-btn v-if="!registered" flat no-caps color="primary" label="Register" @click="doRegister" />
          <q-btn v-else flat no-caps color="primary" label="Done" v-close-popup />
        </q-card-actions>
      </q-card>
    </q-dialog>

    <!-- Send test -->
    <q-dialog v-model="testOpen">
      <q-card style="min-width: 380px">
        <q-card-section class="text-h6">Send to {{ current?.name }}</q-card-section>
        <q-card-section class="q-gutter-sm">
          <q-select
            v-model="testTool"
            dense
            outlined
            emit-value
            map-options
            label="Tool"
            :options="testTools.map((t) => ({ label: toolLabel(t), value: t }))"
          />
          <q-input
            v-if="['speak', 'notify', 'timer_set'].includes(testTool)"
            v-model="testText"
            dense
            outlined
            :label="testTool === 'timer_set' ? 'Label (optional)' : 'Text'"
          />
          <q-input v-if="testTool === 'timer_set'" v-model.number="testSeconds" dense outlined type="number" min="1" label="Seconds" />
          <div class="text-caption text-muted">Waits up to 30 s for the assistant to answer.</div>
        </q-card-section>
        <q-card-actions align="right">
          <q-btn flat no-caps label="Cancel" v-close-popup />
          <q-btn flat no-caps color="primary" label="Send" :loading="testRunning" @click="runTest" />
        </q-card-actions>
      </q-card>
    </q-dialog>
  </div>
</template>

<style scoped>
.body {
  min-height: 0;
}
.list {
  width: 300px;
  border-right: 1px solid var(--pawn-border);
}
.entry {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  text-align: left;
  border: 0;
  border-bottom: 1px solid var(--pawn-border);
  background: transparent;
  color: inherit;
  padding: 10px 8px;
  cursor: pointer;
}
.entry:hover,
.entry.cursor {
  background: var(--pawn-hover);
}
.entry.picked {
  background: var(--pawn-active);
}
.meta {
  min-width: 0;
  display: flex;
  flex-direction: column;
}
.detail {
  min-width: 0;
}
.dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  flex: none;
  background: var(--pawn-border);
}
.dot.inline {
  display: inline-block;
  vertical-align: middle;
}
.dot.active {
  background: #2e9d5b;
}
.dot.idle {
  background: #c9a227;
}
.dot.away {
  background: #8a8f98;
}
.facts {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 4px 16px;
  align-items: center;
}
.thread {
  border-left: 3px solid var(--pawn-border);
  padding-left: 10px;
}
.status {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.status.ok,
.status.approved {
  color: #2e9d5b;
}
.status.denied,
.status.error,
.status.expired {
  color: #c0392b;
}
.snippet {
  background: var(--pawn-hover);
  padding: 8px;
  border-radius: 4px;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-all;
  margin: 0;
}
.hidden {
  display: none;
}
@media (max-width: 800px) {
  .list {
    width: 200px;
  }
}
</style>
