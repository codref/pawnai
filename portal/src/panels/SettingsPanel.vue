<script setup lang="ts">
// The Settings tool tab: a curated subset of pawnai.yaml (the server's
// settings_registry), saved as overrides in {state_dir}/settings.override.yaml.
// pawnai.yaml itself is never edited; Reset drops the override.
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { api, HttpError, type SettingField, type SettingsPayload, type SettingsStatus } from "@/api";
import { useChatsStore } from "@/stores/chats";
import { useTabsStore } from "@/stores/tabs";
import { errorText, notifyError } from "@/lib/errors";
import { fullTime } from "@/lib/time";
import { loadString, saveString } from "@/lib/storage";
import { MODE_OPTIONS, SKIN_OPTIONS, mode, setMode, setSkin, skin, type ThemeMode } from "@/lib/theme";
import { ICONS, settingsIcon } from "@/lib/icons";
import type { SegmentOption } from "@/lib/segment";
import SegmentedControl from "@/components/SegmentedControl.vue";
import SchedulerActivity from "@/components/settings/SchedulerActivity.vue";
import ConfigView from "@/components/settings/ConfigView.vue";

const props = defineProps<{ tabId: string }>();
const chats = useChatsStore();
const tabs = useTabsStore();

const data = ref<SettingsPayload | null>(null);
const loading = ref(true);
const loadError = ref("");
const group = ref(loadString("settings.group", "models"));
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- one model per input type
type Draft = any;
const drafts = ref<Record<string, Draft>>({});
const errors = ref<Record<string, string>>({});
const saving = ref<Set<string>>(new Set());
const status = ref<SettingsStatus | null>(null);
const statusLoading = ref(false);
const diagLoading = ref(false);
const diagRef = ref<{ refresh(): Promise<void> } | null>(null);
const query = ref("");
const searchInput = ref<{ focus(): void } | null>(null);
const body = ref<HTMLElement | null>(null);
const flashPath = ref("");

/** Pages that are not a settings_registry group. */
const LOCAL_PAGES = new Set(["appearance", "status", "activity", "config"]);

watch(group, (g) => {
  saveString("settings.group", g);
  if (g === "status" && !status.value) void loadStatus();
});

const SOURCE: Record<string, { label: string; color: string; tip: string }> = {
  default: { label: "default", color: "grey-6", tip: "Built-in default" },
  yaml: { label: "yaml", color: "blue-grey", tip: "From pawnai.yaml" },
  env: { label: "env", color: "brown", tip: "From a PAWN_… environment variable" },
  override: { label: "portal", color: "primary", tip: "Saved here (settings.override.yaml)" },
};
const APPLY: Record<string, string> = {
  live: "",
  new_chats: "Applies to new chats",
  restart: "Needs a server restart",
};

/** One-line description under each group's title. */
const GROUP_HINTS: Record<string, string> = {
  appearance: "Skin and light/dark mode for this browser.",
  models: "The background model and the chat model catalog.",
  agent: "Harness limits and behaviour of the sallm agent.",
  coworker: "Item extraction, goals and the morning/weekly loop.",
  capture: "Browser capture targets and research enrichment.",
  vault: "Obsidian vault paths, transcript push and vision.",
  scheduler: "Durable schedules: polling, limits and timezone.",
  matrix: "The optional Matrix chatbot worker.",
  portal: "Portal indexing, auto-naming and workspace.",
  emergence: "Observation layer, subjects, night shift and provocations.",
  status: "Server version, database migration and queues.",
  activity: "Schedules, fires, the coworker crons and background runs.",
  config: "The running configuration, credentials masked.",
};

interface NavEntry {
  id: string;
  label: string;
}
const navSections = computed<{ title: string; entries: NavEntry[] }[]>(() => [
  { title: "This browser", entries: [{ id: "appearance", label: "Appearance" }] },
  ...(data.value
    ? [
        { title: "Server", entries: data.value.groups.map((g) => ({ id: g.id, label: g.label })) },
        {
          title: "Diagnostics",
          entries: [
            { id: "status", label: "Status" },
            { id: "activity", label: "Scheduler activity" },
            { id: "config", label: "Configuration" },
          ],
        },
      ]
    : []),
]);
const navFlat = computed(() => navSections.value.flatMap((s) => s.entries));

const terms = computed(() => query.value.toLowerCase().split(/\s+/).filter(Boolean));
const searching = computed(() => terms.value.length > 0);
/** What the body shows: a nav page, or the search results across groups. */
const view = computed(() => (searching.value ? "search" : group.value));
const groupLabel = (id: string) => data.value?.groups.find((g) => g.id === id)?.label ?? id;

function matches(f: SettingField): boolean {
  const hay = [f.label, f.path, f.help, groupLabel(f.group), ...(f.choices ?? [])].join(" ").toLowerCase();
  return terms.value.every((t) => hay.includes(t));
}
const matched = computed(() => (searching.value ? (data.value?.fields ?? []).filter(matches) : []));
const matchCounts = computed(() => {
  const out = new Map<string, number>();
  for (const f of matched.value) out.set(f.group, (out.get(f.group) ?? 0) + 1);
  return out;
});

const current = computed(() =>
  searching.value
    ? { id: "search", label: `Search: ${query.value.trim()}` }
    : (navFlat.value.find((e) => e.id === group.value) ?? { id: group.value, label: group.value }),
);

const modeIcons: Record<ThemeMode, string> = { auto: ICONS.modeAuto, dark: ICONS.modeDark, light: ICONS.modeLight };
const modeOptions: SegmentOption<ThemeMode>[] = MODE_OPTIONS.map((o) => ({ value: o.id, label: o.label, icon: modeIcons[o.id] }));

function pick(id: string): void {
  query.value = "";
  group.value = id;
}

/** Jump to one setting's field (from search results or a diagnostics link). */
async function openSetting(path: string): Promise<void> {
  const f = data.value?.fields.find((x) => x.path === path);
  if (!f) return;
  pick(f.group);
  await nextTick();
  const el = body.value?.querySelector<HTMLElement>(`[data-path="${CSS.escape(path)}"]`);
  el?.scrollIntoView({ block: "center", behavior: "smooth" });
  flashPath.value = path;
  setTimeout(() => {
    if (flashPath.value === path) flashPath.value = "";
  }, 1600);
}

function refreshView(): void {
  if (view.value === "status") void loadStatus();
  else void diagRef.value?.refresh();
}

/** "/" focuses the settings search unless the user is typing somewhere. */
function onPanelKey(ev: KeyboardEvent): void {
  if (ev.key !== "/" || ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const t = ev.target as HTMLElement | null;
  if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
  ev.preventDefault();
  searchInput.value?.focus();
}

function onNavKey(ev: KeyboardEvent): void {
  if (ev.key !== "ArrowDown" && ev.key !== "ArrowUp") return;
  ev.preventDefault();
  const list = navFlat.value;
  const i = list.findIndex((e) => e.id === group.value);
  const next = list[(i + (ev.key === "ArrowDown" ? 1 : -1) + list.length) % list.length];
  if (next) pick(next.id);
  const host = ev.currentTarget as HTMLElement;
  requestAnimationFrame(() => host.querySelector<HTMLElement>(".nav-item.active")?.focus());
}

const fields = computed(() =>
  searching.value ? matched.value : (data.value?.fields ?? []).filter((f) => f.group === group.value),
);
const modelOptions = computed(() => (data.value?.catalog.models ?? []).map((m) => m.id));
const zones = (() => {
  try {
    return (Intl as unknown as { supportedValuesOf(k: string): string[] }).supportedValuesOf("timeZone");
  } catch {
    return ["UTC"];
  }
})();
const tzFilter = ref<string[]>(zones);

function filterZones(val: string, update: (fn: () => void) => void): void {
  update(() => {
    const q = val.toLowerCase();
    tzFilter.value = q ? zones.filter((z) => z.toLowerCase().includes(q)) : zones;
  });
}

function initialDraft(f: SettingField): unknown {
  if (f.secret) return "";
  if (f.type === "list") return Array.isArray(f.value) ? [...f.value] : [];
  return f.value ?? (f.type === "bool" ? false : "");
}

function setData(payload: SettingsPayload): void {
  data.value = payload;
  const next: Record<string, Draft> = {};
  for (const f of payload.fields) next[f.path] = initialDraft(f);
  drafts.value = next;
}

function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(a ?? "") === JSON.stringify(b ?? "");
}

function dirty(f: SettingField): boolean {
  const d = drafts.value[f.path];
  if (f.secret) return Boolean(d);
  return !same(d, initialDraft(f));
}

const dirtyGroups = computed(() => {
  const out = new Set<string>();
  for (const f of data.value?.fields ?? []) if (dirty(f)) out.add(f.group);
  return out;
});
watch(
  () => dirtyGroups.value.size > 0,
  (d) => tabs.setStatus(props.tabId, { dirty: d, title: "Settings" }),
  { immediate: true },
);

async function load(): Promise<void> {
  loading.value = true;
  loadError.value = "";
  try {
    setData(await api().getSettings());
    if (!LOCAL_PAGES.has(group.value) && !data.value?.groups.some((g) => g.id === group.value)) {
      group.value = data.value?.groups[0]?.id ?? "models";
    }
    if (group.value === "status") void loadStatus();
  } catch (e) {
    loadError.value = errorText(e);
  } finally {
    loading.value = false;
  }
}

async function loadStatus(): Promise<void> {
  statusLoading.value = true;
  try {
    status.value = await api().settingsStatus();
  } catch (e) {
    notifyError(e, "Could not load status");
  } finally {
    statusLoading.value = false;
  }
}

function fieldErrors(e: unknown): Record<string, string> | null {
  if (!(e instanceof HttpError) || e.status !== 422) return null;
  try {
    const parsed = JSON.parse(e.detail) as { errors?: Record<string, string> };
    return parsed.errors ?? null;
  } catch {
    return null;
  }
}

function coerce(f: SettingField, v: unknown): unknown {
  if (f.type === "int") return v === "" || v === null ? v : Math.trunc(Number(v));
  if (f.type === "float") return v === "" || v === null ? v : Number(v);
  return v;
}

function busy(path: string, on: boolean): void {
  const next = new Set(saving.value);
  if (on) next.add(path);
  else next.delete(path);
  saving.value = next;
}

/** Keep drafts the user is still editing when a save of another field returns. */
function merge(payload: SettingsPayload, keep: string[]): void {
  const kept = Object.fromEntries(keep.map((p) => [p, drafts.value[p]]));
  setData(payload);
  drafts.value = { ...drafts.value, ...kept };
}

function otherDirty(path: string): string[] {
  return (data.value?.fields ?? []).filter((f) => f.path !== path && dirty(f)).map((f) => f.path);
}

async function save(f: SettingField): Promise<void> {
  busy(f.path, true);
  const { [f.path]: _drop, ...rest } = errors.value;
  errors.value = rest;
  try {
    const payload = await api().patchSettings({ [f.path]: coerce(f, drafts.value[f.path]) });
    merge(payload, otherDirty(f.path));
    if (f.path === "agent.default") void chats.loadModels();
  } catch (e) {
    const errs = fieldErrors(e);
    if (errs) errors.value = { ...errors.value, ...errs };
    else notifyError(e, `Could not save ${f.label}`);
  } finally {
    busy(f.path, false);
  }
}

async function reset(f: SettingField): Promise<void> {
  busy(f.path, true);
  try {
    merge(await api().resetSetting(f.path), otherDirty(f.path));
    const { [f.path]: _drop, ...rest } = errors.value;
    errors.value = rest;
    if (f.path === "agent.default") void chats.loadModels();
  } catch (e) {
    notifyError(e, `Could not reset ${f.label}`);
  } finally {
    busy(f.path, false);
  }
}

function revert(f: SettingField): void {
  drafts.value = { ...drafts.value, [f.path]: initialDraft(f) };
  const { [f.path]: _drop, ...rest } = errors.value;
  errors.value = rest;
}

function shown(v: unknown): string {
  if (v === null || v === undefined || v === "") return "(empty)";
  if (Array.isArray(v)) return v.length ? v.join(", ") : "(empty)";
  return String(v);
}

function nextFire(iso: string): string {
  return fullTime(iso.replace(/([+-]\d\d:\d\d)$/, "$1")) || iso;
}

function onMode(value: ThemeMode | null): void {
  if (value) setMode(value);
}

onMounted(load);
</script>

<template>
  <div class="column no-wrap full-height settings-panel" @keydown="onPanelKey">
    <q-banner v-if="data?.restart_required.length" dense class="pawn-banner warn">
      <template #avatar><q-icon name="restart_alt" /></template>
      Restart pawn-server to apply:
      {{ data.restart_required.map((p) => data!.fields.find((f) => f.path === p)?.label ?? p).join(", ") }}.
    </q-banner>
    <div class="col row no-wrap" style="min-height: 0">
      <nav class="settings-nav border-r pawn-scroll" aria-label="Settings sections">
        <q-input
          ref="searchInput"
          v-model="query"
          dense
          outlined
          clearable
          placeholder="Search settings  /"
          class="nav-search"
          aria-label="Search settings"
          @keydown.esc="query = ''"
          @keydown.enter="matched.length === 1 && openSetting(matched[0].path)"
        >
          <template #prepend><q-icon :name="ICONS.search" size="18px" /></template>
        </q-input>
        <div class="nav-list" @keydown="onNavKey">
        <template v-for="sec in navSections" :key="sec.title">
          <div class="nav-heading">{{ sec.title }}</div>
          <button
            v-for="e in sec.entries"
            :key="e.id"
            type="button"
            class="nav-item"
            :class="{ active: !searching && group === e.id, dim: searching && !matchCounts.has(e.id) }"
            :aria-current="!searching && group === e.id ? 'page' : undefined"
            :tabindex="group === e.id ? 0 : -1"
            @click="pick(e.id)"
          >
            <q-icon :name="settingsIcon(e.id)" class="nav-icon" />
            <span class="nav-label">{{ e.label }}</span>
            <span v-if="searching && matchCounts.has(e.id)" class="count">{{ matchCounts.get(e.id) }}</span>
            <span v-if="dirtyGroups.has(e.id)" class="dot">
              <q-tooltip>Unsaved changes</q-tooltip>
            </span>
          </button>
        </template>
        </div>
      </nav>

      <div ref="body" class="col scroll q-pa-md settings-body">
        <q-input
          v-model="query"
          dense
          outlined
          clearable
          placeholder="Search settings"
          class="nav-compact q-mb-sm"
          aria-label="Search settings"
          @keydown.esc="query = ''"
        >
          <template #prepend><q-icon :name="ICONS.search" size="18px" /></template>
        </q-input>
        <q-select
          :model-value="group"
          :options="navFlat"
          option-value="id"
          option-label="label"
          emit-value
          map-options
          dense
          outlined
          options-dense
          class="nav-compact q-mb-md"
          @update:model-value="(v: string) => pick(v)"
        >
          <template #prepend><q-icon :name="settingsIcon(group)" /></template>
        </q-select>
        <div class="form">
          <header class="page-head">
            <span class="page-icon"><q-icon :name="settingsIcon(current.id)" /></span>
            <div class="col">
              <div class="page-title">{{ current.label }}</div>
              <div v-if="GROUP_HINTS[current.id]" class="page-hint">{{ GROUP_HINTS[current.id] }}</div>
            </div>
            <q-btn
              v-if="view === 'status' || view === 'activity' || view === 'config'"
              flat
              dense
              round
              :icon="ICONS.refresh"
              aria-label="Refresh"
              :loading="view === 'status' ? statusLoading : diagLoading"
              @click="refreshView"
            />
          </header>
          <!-- ── appearance (this browser only) ── -->
          <template v-if="view === 'appearance'">
            <div class="sub-label">Skin</div>
            <div class="skin-grid q-mb-lg">
              <button
                v-for="opt in SKIN_OPTIONS"
                :key="opt.id"
                type="button"
                class="skin-choice"
                :class="{ active: skin === opt.id }"
                :aria-pressed="skin === opt.id"
                @click="setSkin(opt.id)"
              >
                <span class="preview" :class="`preview-${opt.id}`" aria-hidden="true">
                  <span class="pv-head" />
                  <span class="pv-side">
                    <span class="pv-line" /><span class="pv-line short" /><span class="pv-line active" />
                    <span class="pv-line" />
                  </span>
                  <span class="pv-main">
                    <span class="pv-tab" />
                    <span class="pv-bubble" />
                    <span class="pv-bubble right" />
                  </span>
                </span>
                <span class="skin-name">
                  <q-icon v-if="skin === opt.id" name="check_circle" size="16px" class="skin-check" />
                  {{ opt.label }}
                </span>
              </button>
            </div>
            <div class="sub-label">Mode</div>
            <SegmentedControl :model-value="mode" :options="modeOptions" aria-label="Color mode" @update:model-value="onMode" />
            <div class="text-caption text-muted q-mt-sm">
              Saved in this browser. The header button cycles mode.
            </div>
          </template>
          <div v-else-if="loading" class="flex flex-center q-pa-lg"><q-spinner size="32px" /></div>
          <div v-else-if="loadError || !data" class="flex flex-center column text-muted q-pa-lg">
            <q-icon name="error_outline" size="40px" />
            <div class="q-mt-sm">{{ loadError || "Settings are unavailable." }}</div>
            <q-btn flat no-caps label="Retry" class="q-mt-sm" @click="load" />
          </div>
          <template v-else>
            <!-- ── status ── -->
            <template v-if="view === 'status'">
              <q-markup-table v-if="status" flat bordered dense separator="horizontal" class="q-mb-md">
                <tbody>
                  <tr><td class="text-muted">Version</td><td>{{ status.version ?? "?" }}</td></tr>
                  <tr><td class="text-muted">Config file</td><td class="mono">{{ status.config_file ?? "(none: env and defaults)" }}</td></tr>
                  <tr><td class="text-muted">Overrides</td><td class="mono">{{ status.override_file }}</td></tr>
                  <tr><td class="text-muted">State dir</td><td class="mono">{{ status.state_dir }}</td></tr>
                  <tr>
                    <td class="text-muted">Database</td>
                    <td>
                      <q-icon
                        :name="status.db.reachable ? 'check_circle' : 'error'"
                        :color="status.db.reachable ? 'positive' : 'negative'"
                        class="q-mr-xs"
                      />
                      <template v-if="status.db.reachable">
                        migration {{ status.db.revision ?? "?" }}
                        <q-badge
                          v-if="status.db.head && status.db.revision !== status.db.head"
                          color="warning"
                          :label="`head is ${status.db.head}: run alembic upgrade head`"
                          class="q-ml-xs"
                        />
                        <span v-else-if="status.db.head" class="text-muted">(head)</span>
                      </template>
                      <span v-else>{{ status.db.error }}</span>
                    </td>
                  </tr>
                </tbody>
              </q-markup-table>
              <div class="sub-label">Queues</div>
              <div v-if="status?.queues_error" class="text-caption text-muted">{{ status.queues_error }}</div>
              <q-markup-table v-else-if="status?.queues.length" flat bordered dense>
                <thead>
                  <tr><th class="text-left">Name</th><th class="text-left">Topic</th><th>Pending</th><th>Leases</th><th>Dead</th><th>State</th></tr>
                </thead>
                <tbody>
                  <tr v-for="qs in status.queues" :key="qs.name + qs.topic">
                    <td>{{ qs.name }}</td>
                    <td class="mono">{{ qs.topic }}</td>
                    <td class="text-center">{{ qs.messages }}</td>
                    <td class="text-center">{{ qs.leases }}</td>
                    <td class="text-center">{{ qs.dead_letters }}</td>
                    <td class="text-center">{{ qs.paused ? "paused" : "running" }}</td>
                  </tr>
                </tbody>
              </q-markup-table>
              <div v-else-if="status" class="text-caption text-muted">No queues configured.</div>
            </template>

            <!-- ── diagnostics: scheduler activity / configuration ── -->
            <SchedulerActivity
              v-else-if="view === 'activity'"
              ref="diagRef"
              @loading="(on: boolean) => (diagLoading = on)"
              @open-setting="openSetting"
            />
            <ConfigView
              v-else-if="view === 'config'"
              ref="diagRef"
              @loading="(on: boolean) => (diagLoading = on)"
              @open-setting="openSetting"
            />

            <!-- ── a settings group ── -->
            <template v-else>
              <div v-if="view === 'search' && !fields.length" class="flex flex-center column text-muted q-pa-lg">
                <q-icon :name="ICONS.search" size="32px" />
                <div class="q-mt-sm">No setting matches “{{ query.trim() }}”.</div>
                <div class="text-caption">
                  Not every pawnai.yaml key is editable here. Search the
                  <a href="#" class="link" @click.prevent="pick('config')">running configuration</a> instead.
                </div>
              </div>
              <div v-if="view === 'models' && data.catalog.models.length" class="q-mb-md">
                <div class="sub-label">Model catalog</div>
                <q-markup-table flat bordered dense>
                  <thead>
                    <tr><th class="text-left">Id</th><th class="text-left">Provider</th><th>Vision</th></tr>
                  </thead>
                  <tbody>
                    <tr v-for="m in data.catalog.models" :key="m.id">
                      <td class="mono">
                        {{ m.id }}
                        <q-badge v-if="m.id === data.catalog.default" outline color="blue-grey" label="yaml default" class="q-ml-xs" />
                        <q-badge v-if="m.id === data.catalog.background" color="primary" label="background" class="q-ml-xs" />
                      </td>
                      <td>{{ m.provider }}</td>
                      <td class="text-center"><q-icon v-if="m.vision" name="visibility" /></td>
                    </tr>
                  </tbody>
                </q-markup-table>
                <div class="text-caption text-muted q-mt-xs">Providers and models are edited in pawnai.yaml.</div>
              </div>

              <div
                v-for="f in fields"
                :key="f.path"
                class="field"
                :class="{ 'field-dirty': dirty(f), 'field-flash': flashPath === f.path }"
                :data-path="f.path"
              >
                <div class="row items-center no-wrap q-mb-xs field-head">
                  <button
                    v-if="view === 'search'"
                    type="button"
                    class="group-chip"
                    :title="`Open ${groupLabel(f.group)}`"
                    @click="openSetting(f.path)"
                  >
                    <q-icon :name="settingsIcon(f.group)" size="14px" />
                    {{ groupLabel(f.group) }}
                  </button>
                  <div class="field-label">{{ f.label }}</div>
                  <q-badge :color="SOURCE[f.source]?.color" :label="SOURCE[f.source]?.label ?? f.source" class="q-ml-sm" outline>
                    <q-tooltip>
                      {{ SOURCE[f.source]?.tip }}
                      <template v-if="f.yaml_value !== null && f.source !== 'yaml'"> · yaml: {{ shown(f.yaml_value) }}</template>
                    </q-tooltip>
                  </q-badge>
                  <q-badge v-if="f.pending_restart" color="warning" label="restart pending" class="q-ml-xs">
                    <q-tooltip>Running with {{ shown(f.running) }} until the server restarts</q-tooltip>
                  </q-badge>
                  <span v-if="APPLY[f.apply]" class="text-caption text-muted q-ml-sm">{{ APPLY[f.apply] }}</span>
                  <q-space />
                  <span class="text-caption text-muted mono field-path">{{ f.path }}</span>
                </div>

                <div class="row items-start no-wrap q-gutter-x-sm">
                  <div class="col">
                    <q-toggle
                      v-if="f.type === 'bool'"
                      v-model="drafts[f.path]"
                      dense
                      :label="drafts[f.path] ? 'On' : 'Off'"
                    />
                    <q-input
                      v-else-if="f.type === 'int' || f.type === 'float'"
                      v-model.number="drafts[f.path]"
                      type="number"
                      dense
                      outlined
                      :min="f.min ?? undefined"
                      :error="Boolean(errors[f.path])"
                      :error-message="errors[f.path]"
                      style="max-width: 200px"
                      @keydown.enter="save(f)"
                    />
                    <q-select
                      v-else-if="f.type === 'enum'"
                      v-model="drafts[f.path]"
                      dense
                      outlined
                      options-dense
                      :options="f.choices ?? []"
                      :error="Boolean(errors[f.path])"
                      :error-message="errors[f.path]"
                      style="max-width: 320px"
                    />
                    <q-select
                      v-else-if="f.type === 'model'"
                      v-model="drafts[f.path]"
                      dense
                      outlined
                      options-dense
                      :clearable="f.nullable"
                      :options="modelOptions"
                      :placeholder="f.nullable ? 'Background model' : undefined"
                      :error="Boolean(errors[f.path])"
                      :error-message="errors[f.path]"
                      style="max-width: 420px"
                      @clear="drafts[f.path] = ''"
                    />
                    <q-select
                      v-else-if="f.type === 'tz'"
                      v-model="drafts[f.path]"
                      dense
                      outlined
                      use-input
                      fill-input
                      hide-selected
                      input-debounce="0"
                      options-dense
                      :options="tzFilter"
                      :error="Boolean(errors[f.path])"
                      :error-message="errors[f.path]"
                      style="max-width: 320px"
                      @filter="filterZones"
                    />
                    <q-select
                      v-else-if="f.type === 'list'"
                      v-model="drafts[f.path]"
                      dense
                      outlined
                      multiple
                      use-chips
                      use-input
                      hide-dropdown-icon
                      new-value-mode="add-unique"
                      input-debounce="0"
                      placeholder="Type and press Enter"
                      :error="Boolean(errors[f.path])"
                      :error-message="errors[f.path]"
                    />
                    <q-input
                      v-else-if="f.type === 'text'"
                      v-model="drafts[f.path]"
                      type="textarea"
                      autogrow
                      dense
                      outlined
                      :error="Boolean(errors[f.path])"
                      :error-message="errors[f.path]"
                    />
                    <q-input
                      v-else
                      v-model="drafts[f.path]"
                      dense
                      outlined
                      :type="f.secret ? 'password' : 'text'"
                      :class="{ mono: f.type === 'cron' }"
                      :placeholder="f.secret ? (f.set ? '•••••• (set — type to replace)' : 'not set') : undefined"
                      :error="Boolean(errors[f.path])"
                      :error-message="errors[f.path]"
                      :style="f.type === 'cron' ? 'max-width: 260px' : undefined"
                      @keydown.enter="save(f)"
                    />
                  </div>
                  <template v-if="dirty(f)">
                    <q-btn flat dense no-caps label="Revert" @click="revert(f)" />
                    <q-btn
                      unelevated
                      dense
                      no-caps
                      color="primary"
                      icon="save"
                      label="Save"
                      :loading="saving.has(f.path)"
                      @click="save(f)"
                    />
                  </template>
                  <q-btn
                    v-else-if="f.source === 'override'"
                    flat
                    dense
                    no-caps
                    icon="undo"
                    label="Reset"
                    :loading="saving.has(f.path)"
                    @click="reset(f)"
                  >
                    <q-tooltip>Drop the portal override ({{ f.yaml_value !== null ? `yaml: ${shown(f.yaml_value)}` : "back to env / default" }})</q-tooltip>
                  </q-btn>
                </div>
                <div v-if="f.help" class="text-caption text-muted">{{ f.help }}</div>
                <div v-if="f.type === 'cron' && f.next?.length && !dirty(f)" class="text-caption text-muted">
                  Next: {{ f.next.map(nextFire).join(" · ") }}
                </div>
              </div>
              <div v-if="fields.length" class="text-caption text-muted q-mt-lg">
                Saved to <span class="mono">{{ data.override_file }}</span>. pawnai.yaml
                <template v-if="data.config_file">(<span class="mono">{{ data.config_file }}</span>)</template> is never
                edited.
              </div>
            </template>
          </template>
          </div>
        </div>
      </div>
  </div>
</template>

<style scoped>
.border-r {
  border-right: 1px solid var(--pawn-border);
}
.settings-panel {
  container-type: inline-size;
}
/* ── nav ── */
.settings-nav {
  flex: none;
  width: 210px;
  padding: 8px 8px 16px;
  display: flex;
  flex-direction: column;
  gap: 1px;
}
.nav-search {
  margin: 0 2px 6px;
}
.nav-list {
  display: flex;
  flex-direction: column;
  gap: 1px;
}
.nav-item.dim {
  opacity: 0.45;
}
.count {
  flex: none;
  min-width: 18px;
  padding: 0 5px;
  border-radius: 9px;
  font-size: 0.7rem;
  line-height: 16px;
  text-align: center;
  background: var(--pawn-active);
  color: var(--q-primary);
}
.group-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-right: 8px;
  padding: 1px 8px;
  border: 1px solid var(--pawn-border);
  border-radius: 10px;
  background: transparent;
  color: var(--pawn-muted);
  font: inherit;
  font-size: 0.72rem;
  cursor: pointer;
}
.group-chip:hover {
  color: var(--q-primary);
  border-color: var(--q-primary);
}
.field-flash {
  animation: field-flash 1.6s ease-out;
}
@keyframes field-flash {
  0%,
  40% {
    border-color: var(--q-primary);
    box-shadow: 0 0 0 2px var(--q-primary);
  }
  100% {
    box-shadow: none;
  }
}
.link {
  color: var(--q-primary);
  text-decoration: none;
}
.link:hover {
  text-decoration: underline;
}
.nav-heading {
  font-size: 0.66rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--pawn-muted);
  padding: 12px 10px 4px;
}
.nav-heading:first-child {
  padding-top: 4px;
}
.nav-item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  height: 32px;
  padding: 0 10px;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: inherit;
  font: inherit;
  font-size: 0.86rem;
  text-align: left;
  cursor: pointer;
}
.nav-item:hover {
  background: var(--pawn-hover);
}
.nav-item.active {
  background: var(--pawn-active);
  color: var(--q-primary);
  font-weight: 500;
}
.nav-item.active::before {
  content: "";
  position: absolute;
  left: 0;
  top: 7px;
  bottom: 7px;
  width: 3px;
  border-radius: 2px;
  background: var(--q-primary);
}
.nav-item:focus-visible {
  outline: 2px solid var(--q-primary);
  outline-offset: -2px;
}
.nav-icon {
  flex: none;
  width: 20px;
  font-size: 18px;
  color: var(--pawn-muted);
}
.nav-item.active .nav-icon {
  color: inherit;
}
.nav-label {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.nav-compact {
  display: none;
}
@container (max-width: 640px) {
  .settings-nav {
    display: none;
  }
  .nav-compact {
    display: block;
  }
}
/* ── body ── */
.form {
  max-width: 760px;
}
.page-head {
  display: flex;
  align-items: center;
  gap: 12px;
  padding-bottom: 14px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--pawn-border);
}
.page-icon {
  flex: none;
  width: 36px;
  height: 36px;
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  background: var(--pawn-active);
  color: var(--q-primary);
}
.page-title {
  font-size: 1.1rem;
  font-weight: 600;
  line-height: 1.3;
}
.page-hint {
  font-size: 0.8rem;
  color: var(--pawn-muted);
}
.sub-label {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--pawn-muted);
  margin: 4px 0 8px;
}
.field {
  border: 1px solid var(--pawn-border);
  border-radius: 8px;
  padding: 12px 14px 10px;
  margin-bottom: 12px;
  transition: border-color 0.12s;
}
.field-dirty {
  border-color: var(--q-warning);
}
.field-label {
  font-weight: 500;
}
.field-path {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
  margin-left: 8px;
}
.mono {
  font-family: var(--pawn-mono);
  font-size: 12px;
}
.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--q-warning);
  display: inline-block;
  flex: none;
}
/* ── appearance ── */
.skin-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 12px;
  max-width: 520px;
}
.skin-choice {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
  border: 1px solid var(--pawn-border);
  border-radius: 10px;
  background: transparent;
  color: inherit;
  font: inherit;
  cursor: pointer;
  text-align: left;
  transition:
    border-color 0.12s,
    box-shadow 0.12s;
}
.skin-choice:hover {
  border-color: var(--q-primary);
}
.skin-choice.active {
  border-color: var(--q-primary);
  box-shadow: 0 0 0 1px var(--q-primary);
}
.skin-name {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
  padding: 0 2px;
}
.skin-check {
  color: var(--q-primary);
}
/* Mini mock of the portal: header, sidebar, tab strip, two bubbles. */
.preview {
  position: relative;
  display: grid;
  grid-template-columns: 30% 1fr;
  grid-template-rows: 12px 1fr;
  height: 92px;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--pawn-border);
}
.pv-head {
  grid-column: 1 / -1;
}
.pv-side {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 6px;
}
.pv-line {
  height: 4px;
  border-radius: 2px;
  width: 85%;
}
.pv-line.short {
  width: 55%;
}
.pv-main {
  position: relative;
  padding: 6px 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.pv-tab {
  width: 40%;
  height: 6px;
  border-radius: 2px 2px 0 0;
}
.pv-bubble {
  height: 14px;
  width: 70%;
  border-radius: 4px;
}
.pv-bubble.right {
  align-self: flex-end;
  width: 55%;
}
.preview-default {
  background: #f5f6f8;
}
.preview-default .pv-head {
  background: #1976d2;
}
.preview-default .pv-side {
  background: #ffffff;
  border-right: 1px solid rgba(0, 0, 0, 0.08);
}
.preview-default .pv-line {
  background: rgba(0, 0, 0, 0.14);
}
.preview-default .pv-line.active {
  background: rgba(25, 118, 210, 0.55);
}
.preview-default .pv-tab {
  background: #ffffff;
  box-shadow: inset 0 2px 0 #1976d2;
}
.preview-default .pv-bubble {
  background: rgba(0, 0, 0, 0.07);
}
.preview-default .pv-bubble.right {
  background: #e8f0fe;
}
.preview-cyberpunk {
  background-color: #070b10;
  background-image:
    linear-gradient(rgba(92, 225, 255, 0.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(92, 225, 255, 0.08) 1px, transparent 1px);
  background-size: 9px 9px;
}
.preview-cyberpunk .pv-head {
  background: #070b10;
  border-bottom: 1px solid #5ce1ff;
  box-shadow: 0 0 6px rgba(92, 225, 255, 0.6);
}
.preview-cyberpunk .pv-side {
  border-right: 1px solid rgba(92, 225, 255, 0.35);
}
.preview-cyberpunk .pv-line {
  background: rgba(92, 225, 255, 0.3);
}
.preview-cyberpunk .pv-line.active {
  background: #5ce1ff;
  box-shadow: 0 0 4px #5ce1ff;
}
.preview-cyberpunk .pv-tab {
  background: #0d141c;
  box-shadow: inset 0 2px 0 #5ce1ff;
}
.preview-cyberpunk .pv-bubble {
  background: rgba(92, 225, 255, 0.1);
  border: 1px solid rgba(92, 225, 255, 0.3);
}
.preview-cyberpunk .pv-bubble.right {
  background: rgba(255, 77, 154, 0.14);
  border-color: rgba(255, 77, 154, 0.55);
}
</style>
