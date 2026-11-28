<script setup lang="ts">
// The Inbox tool tab: coworker items (filters, grouping, triage, bulk delete),
// idea notes (Keep / Goal / Task / Drop, quick capture) and research capture
// atoms (File / Ignore). Every item card is a right-click target, so Ask about
// this, annotations and Copy work on items like anywhere else.
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { api, type CaptureAtom, type CaptureCollection, type Idea, type InboxItem, type ItemAction } from "@/api";
import { useInboxStore, type InboxSection } from "@/stores/inbox";
import { useTabsStore } from "@/stores/tabs";
import { openBrainmap, openFile, openSession } from "@/router";
import { confirmAction, promptText } from "@/lib/dialogs";
import { fullTime, relativeTime } from "@/lib/time";
import { vTarget, type LocalAction, type Target } from "@/lib/targets";
import {
  ITEM_KINDS,
  groupByDay,
  groupBySource,
  kindTone,
  kindLabel,
  verbsFor,
  type Group,
  type ItemsScope,
} from "@/lib/inbox";
import AnnotationLayer from "@/components/AnnotationLayer.vue";
import StatusBadge from "@/components/StatusBadge.vue";
import ItemCard from "@/components/ItemCard.vue";
import SegmentedControl from "@/components/SegmentedControl.vue";
import type { SegmentOption } from "@/lib/segment";

const props = defineProps<{ tabId: string }>();
const inbox = useInboxStore();
const tabs = useTabsStore();

const SECTIONS: { id: InboxSection; label: string; icon: string }[] = [
  { id: "items", label: "Items", icon: "checklist" },
  { id: "ideas", label: "Ideas", icon: "lightbulb" },
  { id: "captures", label: "Captures", icon: "travel_explore" },
];
const SCOPES: { id: ItemsScope; label: string }[] = [
  { id: "open", label: "Open" },
  { id: "closed", label: "Closed" },
  { id: "all", label: "All" },
];

const sectionOptions = computed<SegmentOption<InboxSection>[]>(() =>
  SECTIONS.map((s) => ({ value: s.id, label: s.label, icon: s.icon, count: sectionCount(s.id) })),
);
const scopeOptions: SegmentOption<ItemsScope>[] = SCOPES.map((s) => ({ value: s.id, label: s.label }));

const search = ref(inbox.q);
const selected = ref<Set<string>>(new Set());
const scroller = ref<HTMLElement | null>(null);

onMounted(() => inbox.attach());
onBeforeUnmount(() => inbox.detach());

watch(
  () => inbox.total,
  (n) => tabs.setStatus(props.tabId, { title: n ? `Inbox (${n})` : "Inbox" }),
  { immediate: true },
);

// ── items ────────────────────────────────────────────────────────────────

const groups = computed<Group<InboxItem>[]>(() =>
  inbox.groupBy === "session" ? groupBySource(inbox.items) : groupByDay(inbox.items, (i) => i.created_at),
);
const itemIds = computed(() => inbox.items.map((i) => i.id));
const selectedIds = computed(() => itemIds.value.filter((id) => selected.value.has(id)));
const allSelected = computed(() => inbox.items.length > 0 && selectedIds.value.length === inbox.items.length);

function toggle(id: string, on: boolean): void {
  const next = new Set(selected.value);
  if (on) next.add(id);
  else next.delete(id);
  selected.value = next;
}

function toggleAll(on: boolean): void {
  selected.value = on ? new Set(itemIds.value) : new Set();
}

watch(itemIds, (ids) => {
  const keep = new Set(ids);
  if ([...selected.value].some((id) => !keep.has(id))) {
    selected.value = new Set([...selected.value].filter((id) => keep.has(id)));
  }
});

function applySearch(): void {
  if (search.value !== inbox.q) inbox.setQuery({ q: search.value });
}

function isoDay(offsetDays: number): string {
  const d = new Date();
  d.setDate(d.getDate() + offsetDays);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

async function run(item: InboxItem, action: ItemAction, arg?: string): Promise<void> {
  if (action === "delete") {
    const ok = await confirmAction("Delete item", `Delete “${excerpt(item.text, 80)}”? Its fingerprint is suppressed.`);
    if (!ok) return;
  }
  if (action === "todo" && arg === undefined) {
    const project = await promptText("Add TODO", "", {
      message: "Optional project for the TODO note. Leave empty for none.",
      label: "Project",
      allowEmpty: true,
    });
    if (project === null) return;
    arg = project || undefined;
  }
  await inbox.act(item, action, arg);
}

async function deleteSelected(): Promise<void> {
  const ids = selectedIds.value;
  if (!ids.length) return;
  if (!(await confirmAction("Delete items", `Delete ${ids.length} selected item(s)?`))) return;
  await inbox.deleteMany(ids);
  selected.value = new Set();
}

async function deleteAllOpen(): Promise<void> {
  const what = inbox.kind ? `open ${kindLabel(inbox.kind)} items` : "open items";
  if (!(await confirmAction("Delete all", `Delete all ${what}${inbox.q ? ` matching “${inbox.q}”` : ""}?`, "Delete all")))
    return;
  await inbox.deleteAllOpen();
}

function excerpt(text: string, n = 60): string {
  const one = text.replace(/\s+/g, " ").trim();
  return one.length > n ? one.slice(0, n - 1) + "…" : one;
}

function itemLocal(item: InboxItem): LocalAction[] {
  const out: LocalAction[] = verbsFor(item)
    .filter((v) => v.action !== "later")
    .map((v) => ({ label: v.label, icon: v.icon, danger: v.danger, run: () => run(item, v.action) }));
  if (!isProposal(item)) {
    out.splice(3, 0, { label: "Later (tomorrow)", icon: "snooze", run: () => run(item, "later", isoDay(1)) });
  }
  if (item.source_kind === "session" && item.source_ref) {
    const sessionId = item.source_ref;
    out.push({
      label: "Open session",
      icon: "graphic_eq",
      run: () => {
        openSession(sessionId, {}, "items");
      },
    });
  }
  if (item.note_key) out.push({ label: "Open note", icon: "description", run: () => openFile(item.note_key!, { pin: true }) });
  out.push({ label: "Show on brainmap", icon: "hub", run: () => openBrainmap("items", `item:${item.id}`) });
  if (item.source_ref) {
    out.push({
      label: "Show items from this source",
      icon: "filter_alt",
      run: () => inbox.setQuery({ sourceRef: item.source_ref, scope: "all" }),
    });
  }
  return out;
}

function isProposal(item: InboxItem): boolean {
  return verbsFor(item)[0]?.action === "approve";
}

function itemTarget(item: InboxItem): Target {
  return { kind: "item", ref: item.id, label: excerpt(item.text, 40), text: item.text, local: itemLocal(item) };
}

function itemRoot(id: string): HTMLElement | null {
  return scroller.value?.querySelector<HTMLElement>(`[data-item="${CSS.escape(id)}"]`) ?? null;
}

// ── ideas ────────────────────────────────────────────────────────────────

const ideaLine = ref("");
async function capture(): Promise<void> {
  const line = ideaLine.value.trim();
  if (line && (await inbox.captureIdea(line))) ideaLine.value = "";
}

async function ideaGoal(idea: Idea): Promise<void> {
  const name = await promptText("Park as a goal thread", idea.title, {
    message: "Adds a Parked thread to Goals.md that links this idea.",
    label: "Thread name",
  });
  if (name) await inbox.ideaAction(idea, "goal", name);
}

async function ideaTask(idea: Idea): Promise<void> {
  const written = await inbox.ideaAction(idea, "task");
  if (written) openFile(written);
}

// ── captures ─────────────────────────────────────────────────────────────

const filing = ref<CaptureAtom | null>(null);
const fileForm = ref({ collection: "", entity: "", tags: "" });
const collections = ref<CaptureCollection[]>([]);
const collectionOptions = computed(() => collections.value.map((c) => c.id));

async function openFiling(c: CaptureAtom): Promise<void> {
  filing.value = c;
  fileForm.value = { collection: c.collection, entity: c.entity, tags: "" };
  if (!collections.value.length) {
    try {
      collections.value = await api().listCaptureCollections();
    } catch {
      collections.value = [];
    }
  }
}

async function submitFiling(): Promise<void> {
  const c = filing.value;
  if (!c) return;
  const ok = await inbox.fileCapture({
    path: c.path,
    collection: fileForm.value.collection.trim(),
    entity: fileForm.value.entity.trim(),
    tags: fileForm.value.tags
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean),
  });
  if (ok) filing.value = null;
}

async function ignoreCapture(c: CaptureAtom): Promise<void> {
  await inbox.fileCapture({ path: c.path, ignore: true });
}

function captureType(c: CaptureAtom): string {
  const t = c.type || c.snippet_kind || "capture";
  return t.replace(/_/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function fileTarget(path: string, label: string, text: string): Target {
  return { kind: "file", ref: path, label, text };
}

const sectionCount = (s: InboxSection) =>
  s === "items" ? inbox.counts.items : s === "ideas" ? inbox.counts.ideas : inbox.counts.captures;
</script>

<template>
  <div class="column no-wrap full-height inbox">
    <div class="row items-center no-wrap q-px-md q-py-sm border-b q-gutter-x-xs">
      <SegmentedControl
        :model-value="inbox.section"
        :options="sectionOptions"
        aria-label="Inbox section"
        @update:model-value="inbox.setSection"
      />
      <q-space />
      <q-btn
        flat
        dense
        round
        icon="refresh"
        aria-label="Refresh"
        :loading="inbox.loadingItems || inbox.loadingVault"
        @click="inbox.section === 'items' ? inbox.loadItems() : inbox.loadVault()"
      />
    </div>

    <!-- ── Items ───────────────────────────────────────────────────────── -->
    <template v-if="inbox.section === 'items'">
      <div class="q-px-md q-pt-sm q-pb-xs column q-gutter-y-xs border-b">
        <div class="row items-center no-wrap q-gutter-x-xs">
          <SegmentedControl
            :model-value="inbox.scope"
            :options="scopeOptions"
            size="sm"
            aria-label="Item status"
            @update:model-value="(v: ItemsScope) => inbox.setQuery({ scope: v })"
          />
          <q-input
            v-model="search"
            dense
            outlined
            clearable
            debounce="350"
            placeholder="Search items…"
            class="col"
            @update:model-value="applySearch"
            @keydown.enter="applySearch"
          >
            <template #prepend><q-icon name="search" size="18px" /></template>
          </q-input>
          <q-select
            :model-value="inbox.groupBy"
            dense
            outlined
            emit-value
            map-options
            options-dense
            style="min-width: 120px"
            :options="[
              { label: 'By date', value: 'day' },
              { label: 'By source', value: 'session' },
            ]"
            @update:model-value="(v) => inbox.setQuery({ groupBy: v })"
          />
        </div>
        <div class="row items-center q-gutter-xs">
          <button
            type="button"
            class="pawn-pill"
            :class="{ active: inbox.kind === '' }"
            :aria-pressed="inbox.kind === ''"
            @click="inbox.setQuery({ kind: '' })"
          >
            All kinds
          </button>
          <button
            v-for="k in ITEM_KINDS"
            :key="k"
            type="button"
            class="pawn-pill kind-pill"
            :class="{ active: inbox.kind === k }"
            :style="{ '--kind': kindTone(k) }"
            :aria-pressed="inbox.kind === k"
            @click="inbox.setQuery({ kind: inbox.kind === k ? '' : k })"
          >
            <span class="kind-dot" />{{ kindLabel(k) }}
          </button>
        </div>
        <div v-if="inbox.sourceRef" class="row items-center text-caption">
          <q-icon name="filter_alt" size="16px" class="q-mr-xs" />
          Filtered: {{ inbox.sourceRef }}
          <q-btn flat dense no-caps size="sm" label="Clear" class="q-ml-xs" @click="inbox.setQuery({ sourceRef: '' })" />
        </div>
        <div class="row items-center text-caption text-muted">
          <q-checkbox
            dense
            size="sm"
            :model-value="allSelected"
            :disable="!inbox.items.length"
            @update:model-value="toggleAll"
          />
          <span class="q-ml-xs">
            {{ selectedIds.length ? `${selectedIds.length} selected` : `${inbox.items.length} of ${inbox.itemsTotal}` }}
          </span>
          <q-btn
            v-if="selectedIds.length"
            flat
            dense
            no-caps
            size="sm"
            color="negative"
            icon="delete"
            label="Delete selected"
            class="q-ml-sm"
            @click="deleteSelected"
          />
          <q-space />
          <q-btn
            v-if="inbox.scope === 'open' && inbox.itemsTotal"
            flat
            dense
            no-caps
            size="sm"
            icon="delete_sweep"
            label="Delete all open"
            @click="deleteAllOpen"
          />
        </div>
      </div>

      <div ref="scroller" class="col scroll q-px-md q-py-sm">
        <div class="list">
          <AnnotationLayer
            v-if="itemIds.length"
            kind="item"
            mode="inline"
            :refs="itemIds"
            :root-for="itemRoot"
          />
          <div v-if="!inbox.items.length && !inbox.loadingItems" class="text-center text-muted q-mt-xl">
            <q-icon name="inbox" size="40px" />
            <div class="q-mt-sm">Nothing here. Items appear after diarized sessions when the coworker is on.</div>
          </div>
          <template v-for="g in groups" :key="g.key">
            <div class="group-label text-caption text-muted q-mt-sm q-mb-xs">{{ g.label }} · {{ g.rows.length }}</div>
            <ItemCard
              v-for="item in g.rows"
              :key="item.id"
              :item="item"
              :selected="selected.has(item.id)"
              :busy="inbox.busy.has(item.id)"
              :source-link="inbox.groupBy !== 'session' && Boolean(item.source_ref)"
              :target="itemTarget(item)"
              @toggle="(on) => toggle(item.id, on)"
              @action="(action, arg) => run(item, action, arg)"
              @filter-source="inbox.setQuery({ sourceRef: item.source_ref, scope: 'all' })"
            />
          </template>
          <div v-if="inbox.items.length < inbox.itemsTotal" class="text-center q-my-md">
            <q-btn flat no-caps label="Load more" :loading="inbox.loadingItems" @click="inbox.loadItems(true)" />
          </div>
        </div>
      </div>
    </template>

    <!-- ── Ideas ───────────────────────────────────────────────────────── -->
    <template v-else-if="inbox.section === 'ideas'">
      <div class="q-px-md q-py-sm border-b">
        <q-input v-model="ideaLine" dense outlined placeholder="Capture an idea (one line)…" @keydown.enter="capture">
          <template #prepend><q-icon name="lightbulb" size="18px" /></template>
          <template #append>
            <q-btn flat dense round icon="add" aria-label="Save idea" :disable="!ideaLine.trim()" @click="capture" />
          </template>
        </q-input>
      </div>
      <div class="col scroll q-px-md q-py-sm">
        <div v-if="!inbox.ideas.length && !inbox.loadingVault" class="text-center text-muted q-mt-xl">
          <q-icon name="lightbulb" size="40px" />
          <div class="q-mt-sm">No ideas waiting. Capture one above or use /idea in chat.</div>
        </div>
        <template v-for="g in groupByDay(inbox.ideas, (i) => i.created)" :key="g.key">
          <div class="group-label text-caption text-muted q-mt-sm q-mb-xs">{{ g.label }} · {{ g.rows.length }}</div>
          <div
            v-for="idea in g.rows"
            :key="idea.path"
            v-target="fileTarget(idea.path, idea.title, idea.line)"
            class="card pawn-tile q-mb-sm"
            style="--tile-tone: var(--pawn-warn)"
          >
            <div class="text-weight-medium">{{ idea.title }}</div>
            <div v-if="idea.line && idea.line !== idea.title" class="text-body2 q-mt-xs pre-wrap">{{ idea.line }}</div>
            <div class="row items-center q-mt-xs q-gutter-x-xs actions">
              <q-btn flat dense round size="sm" icon="description" aria-label="Open note" @click="openFile(idea.path)" />
              <q-space />
              <q-btn
                flat
                dense
                no-caps
                size="sm"
                icon="bookmark"
                label="Keep"
                :loading="inbox.busy.has(idea.path)"
                @click="inbox.ideaAction(idea, 'keep')"
              >
                <q-tooltip>Keep for later (status: later)</q-tooltip>
              </q-btn>
              <q-btn flat dense no-caps size="sm" icon="flag" label="Goal" @click="ideaGoal(idea)">
                <q-tooltip>Park a thread in Goals.md</q-tooltip>
              </q-btn>
              <q-btn flat dense no-caps size="sm" icon="add_task" label="Task" @click="ideaTask(idea)">
                <q-tooltip>Write a TaskNotes task that links this idea</q-tooltip>
              </q-btn>
              <q-btn flat dense no-caps size="sm" icon="close" label="Drop" @click="inbox.ideaAction(idea, 'drop')" />
            </div>
          </div>
        </template>
      </div>
    </template>

    <!-- ── Captures ────────────────────────────────────────────────────── -->
    <template v-else>
      <div class="col scroll q-px-md q-py-sm">
        <div v-if="!inbox.captures.length && !inbox.loadingVault" class="text-center text-muted q-mt-xl">
          <q-icon name="travel_explore" size="40px" />
          <div class="q-mt-sm">No research captures waiting. Send snippets from the browser extension's Research target.</div>
        </div>
        <template v-for="g in groupByDay(inbox.captures, (c) => c.created)" :key="g.key">
          <div class="group-label text-caption text-muted q-mt-sm q-mb-xs">{{ g.label }} · {{ g.rows.length }}</div>
          <div
            v-for="c in g.rows"
            :key="c.path"
            v-target="fileTarget(c.path, c.title, c.caption || c.hint)"
            class="card pawn-tile q-mb-sm"
            style="--tile-tone: var(--pawn-info)"
          >
            <div class="row items-center q-gutter-x-xs text-caption">
              <StatusBadge kind="info" :label="captureType(c)" />
              <StatusBadge v-if="c.status === 'proposed'" kind="ok" label="proposed" />
              <span v-if="c.collection" class="text-muted">→ {{ c.collection }}<template v-if="c.entity"> / {{ c.entity }}</template></span>
              <q-space />
              <span class="age text-muted" :title="fullTime(c.created)">{{ relativeTime(c.created) }}</span>
            </div>
            <div class="text-weight-medium q-mt-xs">{{ c.title }}</div>
            <div v-if="c.caption" class="text-body2 q-mt-xs">{{ c.caption }}</div>
            <div v-if="c.hint" class="text-caption text-muted q-mt-xs">Hint: {{ c.hint }}</div>
            <div class="row items-center q-mt-xs q-gutter-x-xs actions">
              <q-btn flat dense round size="sm" icon="description" aria-label="Open note" @click="openFile(c.path)" />
              <q-btn
                v-if="c.source_url"
                flat
                dense
                round
                size="sm"
                icon="open_in_new"
                aria-label="Open source"
                :href="c.source_url"
                target="_blank"
                rel="noopener noreferrer"
              >
                <q-tooltip>{{ c.source_url }}</q-tooltip>
              </q-btn>
              <q-space />
              <q-btn
                flat
                dense
                no-caps
                size="sm"
                icon="inventory_2"
                label="File"
                :loading="inbox.busy.has(c.path)"
                @click="openFiling(c)"
              />
              <q-btn flat dense no-caps size="sm" icon="visibility_off" label="Ignore" @click="ignoreCapture(c)" />
            </div>
          </div>
        </template>
      </div>
    </template>

    <q-dialog :model-value="Boolean(filing)" @update:model-value="(v) => !v && (filing = null)">
      <q-card style="min-width: 380px">
        <q-card-section class="text-subtitle1">File “{{ filing?.title }}”</q-card-section>
        <q-card-section class="q-gutter-y-sm q-pt-none">
          <q-select
            v-model="fileForm.collection"
            dense
            outlined
            use-input
            hide-selected
            fill-input
            new-value-mode="add-unique"
            input-debounce="0"
            label="Collection"
            :options="collectionOptions"
            @input-value="(v: string) => (fileForm.collection = v)"
          />
          <q-input v-model="fileForm.entity" dense outlined label="Entity (optional)" />
          <q-input v-model="fileForm.tags" dense outlined label="Tags (comma-separated)" />
        </q-card-section>
        <q-card-actions align="right">
          <q-btn v-close-popup flat no-caps label="Cancel" />
          <q-btn
            unelevated
            no-caps
            color="primary"
            label="File"
            :loading="filing ? inbox.busy.has(filing.path) : false"
            @click="submitFiling"
          />
        </q-card-actions>
      </q-card>
    </q-dialog>
  </div>
</template>

<style scoped>
.list {
  position: relative;
  max-width: 920px;
  margin: 0 auto;
  padding-right: 18px;
}
.card {
  border: 1px solid var(--pawn-border);
  border-radius: 8px;
  padding: 8px 10px;
  background: var(--pawn-bg);
}
.card.closed {
  opacity: 0.7;
}
.card.interrupt {
  border-left: 3px solid var(--q-negative);
}
.item-text {
  white-space: pre-wrap;
  word-break: break-word;
}
.item-quote {
  font-style: italic;
  border-left: 2px solid var(--pawn-border);
  padding-left: 8px;
}
.pre-wrap {
  white-space: pre-wrap;
}
.group-label {
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.actions :deep(.q-btn) {
  opacity: 0.85;
}
</style>
