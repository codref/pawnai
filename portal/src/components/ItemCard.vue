<script setup lang="ts">
// Coworker item card shared by the Inbox and, later, the session and Day agenda.
import type { InboxItem, ItemAction } from "@/api";
import { openBrainmap, openFile, openSession } from "@/router";
import { relativeTime, fullTime } from "@/lib/time";
import { vTarget, type Target } from "@/lib/targets";
import { isOpen, isProposal, kindLabel, kindTone, sourceLabel, verbsFor, type Verb } from "@/lib/inbox";
import StatusBadge from "@/components/StatusBadge.vue";

const props = withDefaults(
  defineProps<{
    item: InboxItem;
    selected?: boolean;
    checkbox?: boolean;
    sourceLink?: boolean;
    /** Hide "Open session" when the card is already inside that session. */
    sessionLink?: boolean;
    busy?: boolean;
    target?: Target | null;
  }>(),
  { selected: false, checkbox: true, sourceLink: false, sessionLink: true, busy: false, target: null },
);

const emit = defineEmits<{
  toggle: [on: boolean];
  action: [action: ItemAction, arg?: string];
  filterSource: [];
}>();

function isoDay(offsetDays: number): string {
  const d = new Date();
  d.setDate(d.getDate() + offsetDays);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function inlineVerbs(item: InboxItem): Verb[] {
  return verbsFor(item).filter((v) => v.action !== "delete" && v.action !== "later");
}

function act(action: ItemAction, arg?: string): void {
  emit("action", action, arg);
}
</script>

<template>
  <div
    v-target="target"
    :data-item="item.id"
    class="item-card pawn-tile row no-wrap q-mb-sm"
    :class="{ closed: !isOpen(item), interrupt: item.interrupt }"
    :style="{ '--tile-tone': item.interrupt ? 'var(--pawn-danger)' : kindTone(item.kind) }"
  >
    <q-checkbox
      v-if="checkbox"
      dense
      size="sm"
      class="q-mr-sm self-start"
      :model-value="selected"
      @update:model-value="(v: boolean) => emit('toggle', v)"
    />
    <div class="col" style="min-width: 0">
      <div class="row items-center q-gutter-x-xs text-caption">
        <span class="pawn-status" :style="{ '--pawn-status': kindTone(item.kind) }">{{ kindLabel(item.kind) }}</span>
        <StatusBadge v-if="item.interrupt" kind="danger" label="interrupt" />
        <span v-if="item.status !== 'new'" class="text-muted">
          {{ item.status }}<template v-if="item.snooze_until"> until {{ item.snooze_until.slice(0, 10) }}</template>
        </span>
        <span v-if="item.owner" class="text-muted">· {{ item.owner }}</span>
        <span v-if="item.due" class="text-muted">· due {{ item.due }}</span>
        <q-space />
        <span class="age text-muted" :title="fullTime(item.created_at)">{{ relativeTime(item.created_at) }}</span>
      </div>
      <div class="item-text q-mt-xs" data-pawn-anchor-root>{{ item.text }}</div>
      <div v-if="item.quote" class="item-quote text-caption q-mt-xs">“{{ item.quote }}”</div>
      <div v-if="item.reason" class="text-caption text-muted q-mt-xs">{{ item.reason }}</div>
      <div class="row items-center q-mt-xs q-gutter-x-xs actions">
        <q-btn
          v-if="sessionLink && item.source_kind === 'session' && item.source_ref"
          flat
          dense
          no-caps
          size="sm"
          icon="graphic_eq"
          label="Open session"
          @click="openSession(item.source_ref, {}, 'items')"
        />
        <q-btn
          v-if="sourceLink && item.source_ref"
          flat
          dense
          no-caps
          size="sm"
          icon="link"
          class="text-muted"
          :label="sourceLabel(item)"
          @click="emit('filterSource')"
        >
          <q-tooltip>Show every item from this source</q-tooltip>
        </q-btn>
        <q-space />
        <template v-if="isOpen(item)">
          <q-btn
            v-for="v in inlineVerbs(item)"
            :key="v.action"
            flat
            dense
            no-caps
            size="sm"
            :icon="v.icon"
            :label="v.label"
            :loading="busy"
            @click="act(v.action)"
          />
          <q-btn v-if="!isProposal(item)" flat dense no-caps size="sm" icon="snooze" label="Later">
            <q-menu anchor="bottom right" self="top right">
              <q-list dense style="min-width: 160px">
                <q-item v-close-popup clickable @click="act('later', isoDay(1))">
                  <q-item-section>Tomorrow</q-item-section>
                </q-item>
                <q-item v-close-popup clickable @click="act('later', isoDay(7))">
                  <q-item-section>Next week</q-item-section>
                </q-item>
                <q-separator />
                <q-date
                  minimal
                  flat
                  mask="YYYY-MM-DD"
                  :model-value="isoDay(1)"
                  :options="(d: string) => d >= isoDay(1).replace(/-/g, '/')"
                  @update:model-value="(d: string) => act('later', d)"
                />
              </q-list>
            </q-menu>
          </q-btn>
        </template>
        <q-btn
          flat
          dense
          round
          size="sm"
          icon="hub"
          aria-label="Show on brainmap"
          @click="openBrainmap('items', `item:${item.id}`)"
        >
          <q-tooltip>Show on the brainmap</q-tooltip>
        </q-btn>
        <q-btn
          v-if="item.note_key"
          flat
          dense
          round
          size="sm"
          icon="description"
          aria-label="Open note"
          @click="openFile(item.note_key, { pin: true })"
        >
          <q-tooltip>Open {{ item.note_key }}</q-tooltip>
        </q-btn>
        <q-btn flat dense round size="sm" icon="delete" aria-label="Delete" @click="act('delete')" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.item-card {
  border: 1px solid var(--pawn-border);
  border-radius: 8px;
  padding: 8px 10px;
  background: var(--pawn-bg);
}
.item-card.closed {
  opacity: 0.7;
}
.item-card.interrupt {
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
.actions :deep(.q-btn) {
  min-height: 24px;
}
</style>
