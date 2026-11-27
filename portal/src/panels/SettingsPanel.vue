<script setup lang="ts">
// The Settings tool tab: a curated subset of pawnai.yaml (the server's
// settings_registry), saved as overrides in {state_dir}/settings.override.yaml.
// pawnai.yaml itself is never edited; Reset drops the override.
import { computed, onMounted, ref, watch } from "vue";
import { api, HttpError, type SettingField, type SettingsPayload, type SettingsStatus } from "@/api";
import { useChatsStore } from "@/stores/chats";
import { useTabsStore } from "@/stores/tabs";
import { errorText, notifyError } from "@/lib/errors";
import { fullTime } from "@/lib/time";
import { loadString, saveString } from "@/lib/storage";

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

const fields = computed(() => (data.value?.fields ?? []).filter((f) => f.group === group.value));
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
    if (group.value !== "status" && !data.value?.groups.some((g) => g.id === group.value)) {
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

onMounted(load);
</script>

<template>
  <div class="column no-wrap full-height">
    <div v-if="loading" class="col flex flex-center"><q-spinner size="32px" /></div>
    <div v-else-if="loadError || !data" class="col flex flex-center column text-muted">
      <q-icon name="error_outline" size="40px" />
      <div class="q-mt-sm">{{ loadError || "Settings are unavailable." }}</div>
      <q-btn flat no-caps label="Retry" class="q-mt-sm" @click="load" />
    </div>
    <template v-else>
      <q-banner v-if="data.restart_required.length" dense class="bg-orange-1 text-orange-10 border-b">
        <template #avatar><q-icon name="restart_alt" /></template>
        Restart pawn-server to apply:
        {{ data.restart_required.map((p) => data!.fields.find((f) => f.path === p)?.label ?? p).join(", ") }}.
      </q-banner>
      <div class="col row no-wrap" style="min-height: 0">
        <q-tabs v-model="group" vertical dense no-caps inline-label active-color="primary" class="tabs border-r">
          <q-tab v-for="g in data.groups" :key="g.id" :name="g.id">
            <div class="row items-center no-wrap">
              {{ g.label }}
              <span v-if="dirtyGroups.has(g.id)" class="dot q-ml-xs" />
            </div>
          </q-tab>
          <q-separator class="q-my-xs" />
          <q-tab name="status" icon="monitor_heart" label="Status" />
        </q-tabs>

        <div class="col scroll q-pa-md">
          <div class="form">
            <!-- ── status ── -->
            <template v-if="group === 'status'">
              <div class="row items-center q-mb-sm">
                <div class="text-subtitle1">Server status</div>
                <q-space />
                <q-btn flat dense round icon="refresh" aria-label="Refresh" :loading="statusLoading" @click="loadStatus" />
              </div>
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
                          color="orange"
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
              <div class="text-subtitle2 q-mb-xs">Queues</div>
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

            <!-- ── a settings group ── -->
            <template v-else>
              <div v-if="group === 'models' && data.catalog.models.length" class="q-mb-md">
                <div class="text-subtitle2 q-mb-xs">Model catalog</div>
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

              <div v-for="f in fields" :key="f.path" class="field q-mb-md">
                <div class="row items-center no-wrap q-mb-xs">
                  <div class="text-weight-medium">{{ f.label }}</div>
                  <q-badge :color="SOURCE[f.source]?.color" :label="SOURCE[f.source]?.label ?? f.source" class="q-ml-sm" outline>
                    <q-tooltip>
                      {{ SOURCE[f.source]?.tip }}
                      <template v-if="f.yaml_value !== null && f.source !== 'yaml'"> · yaml: {{ shown(f.yaml_value) }}</template>
                    </q-tooltip>
                  </q-badge>
                  <q-badge v-if="f.pending_restart" color="orange" label="restart pending" class="q-ml-xs">
                    <q-tooltip>Running with {{ shown(f.running) }} until the server restarts</q-tooltip>
                  </q-badge>
                  <span v-if="APPLY[f.apply]" class="text-caption text-muted q-ml-sm">{{ APPLY[f.apply] }}</span>
                  <q-space />
                  <span class="text-caption text-muted mono">{{ f.path }}</span>
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
              <div class="text-caption text-muted q-mt-lg">
                Saved to <span class="mono">{{ data.override_file }}</span>. pawnai.yaml
                <template v-if="data.config_file">(<span class="mono">{{ data.config_file }}</span>)</template> is never
                edited.
              </div>
            </template>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.tabs {
  min-width: 170px;
}
.border-r {
  border-right: 1px solid var(--pawn-border);
}
.form {
  max-width: 760px;
}
.field {
  border-bottom: 1px solid var(--pawn-border);
  padding-bottom: 12px;
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
}
.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--q-warning);
  display: inline-block;
}
</style>
