<script setup lang="ts">
// Diagnostics → Configuration: the running AgentConfig flattened to
// path = value, credentials masked by the server. Paths saved as portal
// overrides are marked; paths the Settings tab can edit link to their field.
import { computed, onMounted, ref } from "vue";
import { api, type ConfigDiagnostics } from "@/api";
import { notifyError } from "@/lib/errors";
import { ICONS } from "@/lib/icons";

const emit = defineEmits<{
  (e: "loading", on: boolean): void;
  (e: "open-setting", path: string): void;
}>();

const data = ref<ConfigDiagnostics | null>(null);
const filter = ref("");
const loading = ref(false);

async function refresh(): Promise<void> {
  loading.value = true;
  emit("loading", true);
  try {
    data.value = await api().settingsConfig();
  } catch (e) {
    notifyError(e, "Could not load the configuration");
  } finally {
    loading.value = false;
    emit("loading", false);
  }
}
defineExpose({ refresh });
onMounted(refresh);

interface Row {
  path: string;
  section: string;
  value: unknown;
}

function flatten(obj: unknown, prefix: string, out: Row[]): void {
  if (obj && typeof obj === "object" && !Array.isArray(obj) && Object.keys(obj).length) {
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) flatten(v, prefix ? `${prefix}.${k}` : k, out);
    return;
  }
  out.push({ path: prefix, section: prefix.split(".")[0], value: obj });
}

const rows = computed<Row[]>(() => {
  const out: Row[] = [];
  if (data.value) flatten(data.value.config, "", out);
  return out;
});

const overridden = computed(() => {
  const out: Row[] = [];
  if (data.value) flatten(data.value.overrides, "", out);
  return new Set(out.map((r) => r.path));
});

function shown(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "string") return v === "" ? '""' : v;
  return JSON.stringify(v);
}

const filtered = computed(() => {
  const terms = filter.value.toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return rows.value;
  return rows.value.filter((r) => {
    const hay = `${r.path} ${shown(r.value)}`.toLowerCase();
    return terms.every((t) => hay.includes(t));
  });
});

const sections = computed(() => {
  const map = new Map<string, Row[]>();
  for (const r of filtered.value) {
    const list = map.get(r.section) ?? [];
    list.push(r);
    map.set(r.section, list);
  }
  return [...map.entries()];
});
</script>

<template>
  <div v-if="!data" class="flex flex-center q-pa-lg"><q-spinner size="32px" /></div>
  <div v-else>
    <q-input v-model="filter" dense outlined clearable placeholder="Filter by path or value" class="q-mb-md" @keydown.esc="filter = ''">
      <template #prepend><q-icon :name="ICONS.search" /></template>
      <template #append>
        <span class="text-caption text-muted">{{ filtered.length }} / {{ rows.length }}</span>
      </template>
    </q-input>
    <div v-if="!sections.length" class="text-caption text-muted">Nothing matches.</div>
    <div v-for="[section, list] in sections" :key="section" class="section q-mb-md">
      <div class="section-head mono">{{ section }}</div>
      <div v-for="r in list" :key="r.path" class="row-kv">
        <span class="mono path">{{ r.path.slice(section.length + 1) || r.path }}</span>
        <span class="mono value" :class="{ masked: shown(r.value).includes('••••••') }">{{ shown(r.value) }}</span>
        <span class="tags">
          <q-badge v-if="overridden.has(r.path)" outline color="primary" label="portal" />
          <q-btn
            v-if="data.editable[r.path]"
            flat
            dense
            round
            size="sm"
            icon="edit"
            :aria-label="`Edit ${r.path}`"
            @click="emit('open-setting', r.path)"
          >
            <q-tooltip>Edit in Settings</q-tooltip>
          </q-btn>
        </span>
      </div>
    </div>
    <div class="text-caption text-muted q-mt-md">
      The running config (yaml, env and portal overrides; credentials masked).
      <template v-if="data.config_file">Loaded from <span class="mono">{{ data.config_file }}</span>.</template>
      Values marked as needing a restart may differ from what is saved.
    </div>
  </div>
</template>

<style scoped>
.section {
  border: 1px solid var(--pawn-border);
  border-radius: 8px;
  overflow: hidden;
}
.section-head {
  padding: 6px 12px;
  font-weight: 600;
  background: var(--pawn-hover);
  border-bottom: 1px solid var(--pawn-border);
}
.row-kv {
  display: grid;
  grid-template-columns: minmax(140px, 38%) 1fr auto;
  gap: 12px;
  align-items: center;
  padding: 3px 12px;
  min-height: 28px;
}
.row-kv + .row-kv {
  border-top: 1px solid var(--pawn-border);
}
.row-kv:hover {
  background: var(--pawn-hover);
}
.path {
  color: var(--pawn-muted);
  overflow-wrap: anywhere;
}
.value {
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}
.masked {
  color: var(--pawn-muted);
}
.tags {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.mono {
  font-family: var(--pawn-mono);
  font-size: 12px;
}
</style>
