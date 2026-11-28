<script setup lang="ts">
// Diagnostics → Scheduler activity: the durable scheduler and the coworker
// crons (briefing, weekly, emergence night shift). Rows come from the DB, so
// they cover every process; the loop heartbeats only cover this server
// process (a --scheduler-only worker elsewhere shows as "not in this process").
import { computed, onMounted, onUnmounted, ref } from "vue";
import { api, type SchedulerDiagnostics, type WorkerBeat } from "@/api";
import { notifyError } from "@/lib/errors";
import { fullTime } from "@/lib/time";
import StatusBadge, { type StatusKind } from "@/components/StatusBadge.vue";

const emit = defineEmits<{
  (e: "loading", on: boolean): void;
  (e: "open-setting", path: string): void;
}>();

const data = ref<SchedulerDiagnostics | null>(null);
const loading = ref(false);
const now = ref(Date.now());
let timer: ReturnType<typeof setInterval> | undefined;

async function refresh(): Promise<void> {
  if (loading.value) return;
  loading.value = true;
  emit("loading", true);
  try {
    data.value = await api().settingsScheduler();
    now.value = Date.now();
  } catch (e) {
    notifyError(e, "Could not load scheduler activity");
  } finally {
    loading.value = false;
    emit("loading", false);
  }
}
defineExpose({ refresh });

onMounted(() => {
  void refresh();
  timer = setInterval(() => {
    now.value = Date.now();
    if (document.visibilityState === "visible") void refresh();
  }, 15_000);
});
onUnmounted(() => clearInterval(timer));

function parse(iso: string | null | undefined): number {
  if (!iso) return NaN;
  return Date.parse(/[zZ]$|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z");
}

/** "in 5m" / "3h ago" / "now". */
function rel(iso: string | null | undefined): string {
  const t = parse(iso);
  if (Number.isNaN(t)) return "—";
  const diff = Math.round((t - now.value) / 1000);
  const s = Math.abs(diff);
  if (s < 45) return "now";
  const unit = s < 3600 ? `${Math.round(s / 60)}m` : s < 86_400 ? `${Math.round(s / 3600)}h` : `${Math.round(s / 86_400)}d`;
  return diff > 0 ? `in ${unit}` : `${unit} ago`;
}

function duration(s: number | null): string {
  if (s === null || s === undefined) return "—";
  if (s < 60) return `${s.toFixed(1)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

const KIND: Record<string, StatusKind> = {
  completed: "ok",
  done: "ok",
  active: "ok",
  failed: "danger",
  error: "danger",
  running: "info",
  claimed: "info",
  pending: "neutral",
  paused: "warn",
  proposed: "warn",
  cancelled: "neutral",
};
const kind = (status: string | null | undefined): StatusKind => KIND[String(status ?? "")] ?? "neutral";

interface LoopCard {
  id: string;
  label: string;
  beat: WorkerBeat | undefined;
  interval: number;
  state: { kind: StatusKind; label: string };
}

function loopState(beat: WorkerBeat | undefined, interval: number): { kind: StatusKind; label: string } {
  if (!beat) return { kind: "neutral", label: "not in this process" };
  if (!beat.running) return { kind: "danger", label: "stopped" };
  if (beat.error) return { kind: "danger", label: "last tick failed" };
  const last = parse(beat.last_tick_at ?? beat.started_at);
  // A tick can legitimately run long (a scheduled turn, the night shift), so
  // only call it stalled after a generous multiple of the poll interval.
  if (!Number.isNaN(last) && now.value - last > Math.max(interval * 4, 120) * 1000) {
    return { kind: "warn", label: "no tick lately" };
  }
  return { kind: "ok", label: "running" };
}

const loops = computed<LoopCard[]>(() => {
  const d = data.value;
  if (!d) return [];
  const sched = d.workers.scheduler;
  const cow = d.workers.coworker;
  const schedInterval = Number(sched?.poll_interval_seconds ?? d.config.poll_interval_seconds ?? 30);
  const cowInterval = Number(cow?.poll_interval_seconds ?? 30);
  return [
    { id: "scheduler", label: "Scheduler loop", beat: sched, interval: schedInterval, state: loopState(sched, schedInterval) },
    { id: "coworker", label: "Coworker loop", beat: cow, interval: cowInterval, state: loopState(cow, cowInterval) },
  ];
});

const counts = computed(() => Object.entries(data.value?.fire_counts_24h ?? {}).sort((a, b) => b[1] - a[1]));

function scheduleWhen(s: SchedulerDiagnostics["schedules"][number]): string {
  if (s.schedule_kind === "cron") return s.cron_expression ?? "";
  if (s.schedule_kind === "interval") return `every ${duration(s.interval_seconds)}`;
  return s.run_at ? `once ${fullTime(s.run_at)}` : "once";
}

/** The running coworker loop holds its own next fire; it differs when the cron changed since start. */
function loopDrift(c: SchedulerDiagnostics["crons"][number]): boolean {
  if (!c.enabled || !c.running_next || !c.next.length) return false;
  return Math.abs(parse(c.running_next) - parse(c.next[0])) > 60_000;
}

function shortResult(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}

const configRows = computed(() => Object.entries(data.value?.config ?? {}));
</script>

<template>
  <div v-if="!data" class="flex flex-center q-pa-lg"><q-spinner size="32px" /></div>
  <div v-else class="activity">
    <q-banner v-if="data.db_error" dense class="pawn-banner warn q-mb-md">
      <template #avatar><q-icon name="error_outline" /></template>
      Database unavailable: {{ data.db_error }}
    </q-banner>

    <!-- ── loops ── -->
    <div class="loops q-mb-md">
      <div v-for="l in loops" :key="l.id" class="loop">
        <div class="row items-center no-wrap q-mb-xs">
          <div class="loop-title">{{ l.label }}</div>
          <q-space />
          <StatusBadge :kind="l.state.kind" :label="l.state.label" :busy="l.state.kind === 'ok'" />
        </div>
        <template v-if="l.beat">
          <div class="kv">
            <span>Last tick</span>
            <span :title="fullTime(l.beat.last_tick_at)">{{ rel(l.beat.last_tick_at) }}</span>
          </div>
          <div class="kv"><span>Started</span><span :title="fullTime(l.beat.started_at)">{{ rel(l.beat.started_at) }}</span></div>
          <div class="kv"><span>Ticks</span><span>{{ l.beat.ticks }} · every {{ l.interval }}s</span></div>
          <div v-if="l.id === 'scheduler'" class="kv">
            <span>Fired</span>
            <span>
              {{ l.beat.fired ?? 0 }}
              <template v-if="l.beat.last_fire_at">· last {{ rel(String(l.beat.last_fire_at)) }}</template>
            </span>
          </div>
          <div v-if="l.beat.error" class="text-negative text-caption q-mt-xs ellipsis-2-lines" :title="String(l.beat.error)">
            {{ l.beat.error }}
          </div>
        </template>
        <div v-else class="text-caption text-muted">
          <template v-if="l.id === 'scheduler'">
            {{ data.config.enabled ? "Enabled, but not running in this process (a separate --scheduler-only worker, or this one runs a single other worker)." : "Disabled (agent_scheduler.enabled)." }}
          </template>
          <template v-else>Not running here: the coworker loop starts with pawn-server serve when coworker.enabled is on.</template>
        </div>
      </div>
    </div>

    <div v-if="counts.length" class="row items-center q-gutter-xs q-mb-md">
      <span class="text-caption text-muted q-mr-xs">Fires, last 24h:</span>
      <StatusBadge v-for="[st, n] in counts" :key="st" :kind="kind(st)" :label="`${n} ${st}`" />
    </div>

    <!-- ── built-in crons ── -->
    <div class="sub-label">Built-in crons</div>
    <q-markup-table flat bordered dense class="q-mb-lg">
      <thead>
        <tr><th class="text-left">Job</th><th class="text-left">Cron</th><th class="text-left">Next</th><th class="text-left">Last run</th></tr>
      </thead>
      <tbody>
        <tr v-for="c in data.crons" :key="c.id" :class="{ 'text-muted': !c.enabled }">
          <td>
            {{ c.label }}
            <div v-if="!c.enabled" class="text-caption">
              off — <a href="#" class="link" @click.prevent="emit('open-setting', c.gate)">{{ c.gate }}</a>
            </div>
          </td>
          <td>
            <a href="#" class="link mono" @click.prevent="emit('open-setting', c.path)">{{ c.cron || "(none)" }}</a>
            <div class="text-caption text-muted">{{ c.timezone }}</div>
          </td>
          <td>
            <template v-if="c.next.length">
              <span :title="c.next.map(fullTime).join('\n')">{{ rel(c.next[0]) }}</span>
              <div class="text-caption text-muted">{{ fullTime(c.next[0]) }}</div>
            </template>
            <span v-else>—</span>
            <div v-if="loopDrift(c)" class="text-caption text-warning">
              Running loop fires {{ rel(c.running_next) }}: restart to apply.
            </div>
          </td>
          <td>
            <template v-if="c.last">
              <StatusBadge :kind="c.last.ok ? 'ok' : 'danger'" :label="c.last.ok ? 'ok' : 'failed'" />
              <span class="q-ml-xs" :title="fullTime(c.last.at)">{{ rel(c.last.at) }}</span>
              <div v-if="shortResult(c.last.result)" class="text-caption text-muted ellipsis result" :title="shortResult(c.last.result)">
                {{ shortResult(c.last.result) }}
              </div>
            </template>
            <template v-else-if="c.id === 'night' && data.last_night_run">
              <StatusBadge :kind="kind(data.last_night_run.status)" :label="data.last_night_run.status" />
              <span class="q-ml-xs" :title="fullTime(data.last_night_run.created_at)">{{ rel(data.last_night_run.created_at) }}</span>
            </template>
            <span v-else class="text-muted">since start: none</span>
          </td>
        </tr>
      </tbody>
    </q-markup-table>

    <!-- ── durable schedules ── -->
    <div class="sub-label">Schedules</div>
    <q-markup-table v-if="data.schedules.length" flat bordered dense class="q-mb-lg">
      <thead>
        <tr><th class="text-left">Name</th><th class="text-left">When</th><th class="text-left">Status</th><th class="text-left">Next</th><th class="text-left">Last</th></tr>
      </thead>
      <tbody>
        <tr v-for="s in data.schedules" :key="s.id">
          <td>
            <div>{{ s.name }}</div>
            <div class="text-caption text-muted ellipsis result" :title="s.prompt">{{ s.prompt }}</div>
          </td>
          <td>
            <span :class="{ mono: s.schedule_kind === 'cron' }">{{ scheduleWhen(s) }}</span>
            <div class="text-caption text-muted">{{ s.timezone }}<template v-if="s.model"> · {{ s.model }}</template></div>
          </td>
          <td><StatusBadge :kind="kind(s.status)" :label="s.status" /></td>
          <td :title="fullTime(s.next_run_at)">{{ s.status === "active" ? rel(s.next_run_at) : "—" }}</td>
          <td :title="fullTime(s.last_run_at)">{{ rel(s.last_run_at) }}</td>
        </tr>
      </tbody>
    </q-markup-table>
    <div v-else class="text-caption text-muted q-mb-lg">No schedules. The agent proposes them with schedule_propose.</div>

    <template v-if="data.proposals.length">
      <div class="sub-label">Pending proposals</div>
      <q-markup-table flat bordered dense class="q-mb-xs">
        <tbody>
          <tr v-for="p in data.proposals" :key="p.id">
            <td style="width: 1%"><StatusBadge kind="warn" :label="p.action" /></td>
            <td>
              {{ String(p.proposed_payload.name ?? p.schedule_id ?? p.id) }}
              <div v-if="p.rationale" class="text-caption text-muted">{{ p.rationale }}</div>
            </td>
            <td class="mono text-caption">{{ p.id.slice(0, 8) }}</td>
            <td :title="fullTime(p.created_at)">{{ rel(p.created_at) }}</td>
          </tr>
        </tbody>
      </q-markup-table>
      <div class="text-caption text-muted q-mb-lg">Approve with <span class="mono">pawn-server schedules approve &lt;id&gt;</span> or from the Inbox.</div>
    </template>

    <!-- ── fires ── -->
    <div class="sub-label">Recent fires</div>
    <q-markup-table v-if="data.fires.length" flat bordered dense class="q-mb-lg">
      <thead>
        <tr><th class="text-left">Due</th><th class="text-left">Schedule</th><th class="text-left">Status</th><th class="text-right">Took</th></tr>
      </thead>
      <tbody>
        <tr v-for="f in data.fires" :key="f.id">
          <td :title="fullTime(f.scheduled_for)">{{ rel(f.scheduled_for) }}</td>
          <td>
            {{ f.schedule_name ?? f.schedule_id }}
            <div v-if="f.error" class="text-caption text-negative ellipsis result" :title="f.error">{{ f.error }}</div>
          </td>
          <td>
            <StatusBadge :kind="kind(f.status)" :label="f.status" :busy="f.status === 'running'" />
            <span v-if="f.attempt > 1" class="text-caption text-muted q-ml-xs">try {{ f.attempt }}</span>
          </td>
          <td class="text-right">{{ duration(f.duration_s) }}</td>
        </tr>
      </tbody>
    </q-markup-table>
    <div v-else class="text-caption text-muted q-mb-lg">No fires yet.</div>

    <!-- ── background runs ── -->
    <div class="sub-label">Background runs</div>
    <q-markup-table v-if="data.runs.length" flat bordered dense class="q-mb-lg">
      <thead>
        <tr><th class="text-left">When</th><th class="text-left">Source</th><th class="text-left">What</th><th class="text-left">Status</th><th class="text-right">Took</th></tr>
      </thead>
      <tbody>
        <tr v-for="r in data.runs" :key="r.id">
          <td :title="fullTime(r.created_at)">{{ rel(r.created_at) }}</td>
          <td>{{ r.source }}</td>
          <td>
            <span class="mono">{{ r.command }}</span>
            <span v-if="r.session_id" class="text-caption text-muted q-ml-xs">{{ r.session_id }}</span>
            <div v-if="r.error" class="text-caption text-negative ellipsis result" :title="r.error">{{ r.error }}</div>
            <div v-else-if="r.prompt" class="text-caption text-muted ellipsis result" :title="r.prompt">{{ r.prompt }}</div>
          </td>
          <td><StatusBadge :kind="kind(r.status)" :label="r.status" :busy="r.status === 'running'" /></td>
          <td class="text-right">{{ duration(r.duration_s) }}</td>
        </tr>
      </tbody>
    </q-markup-table>
    <div v-else class="text-caption text-muted q-mb-lg">No background runs recorded.</div>

    <!-- ── scheduler config ── -->
    <div class="row items-center">
      <div class="sub-label">Scheduler configuration</div>
      <q-space />
      <a href="#" class="link text-caption" @click.prevent="emit('open-setting', 'agent_scheduler.enabled')">Edit in Settings</a>
    </div>
    <q-markup-table flat bordered dense>
      <tbody>
        <tr v-for="[k, v] in configRows" :key="k">
          <td class="mono text-muted">agent_scheduler.{{ k }}</td>
          <td class="mono">{{ typeof v === "object" ? JSON.stringify(v) : String(v) }}</td>
        </tr>
      </tbody>
    </q-markup-table>
    <div class="text-caption text-muted q-mt-xs">Refreshes every 15 seconds while shown.</div>
  </div>
</template>

<style scoped>
.loops {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
}
.loop {
  border: 1px solid var(--pawn-border);
  border-radius: 8px;
  padding: 10px 12px;
}
.loop-title {
  font-weight: 500;
}
.kv {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 0.8rem;
}
.kv > span:first-child {
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
.mono {
  font-family: var(--pawn-mono);
  font-size: 12px;
}
.result {
  max-width: 340px;
}
.link {
  color: var(--q-primary);
  text-decoration: none;
}
.link:hover {
  text-decoration: underline;
}
td {
  vertical-align: top;
}
</style>
