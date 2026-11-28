<script setup lang="ts">
// Empty workspace: today's remaining events, sessions that still need a name,
// and a shortcut into the day-plan chat.
import { computed, onMounted, ref } from "vue";
import { api, type PlannerEvent, type SessionSummary } from "@/api";
import { clockOf, dayHeading, minutesOf, todayISO } from "@/lib/planner";
import { openChat, openDay, openSession, openSessions } from "@/router";
import { isSideChat, useChatsStore } from "@/stores/chats";

const chats = useChatsStore();
const today = todayISO();
const events = ref<PlannerEvent[]>([]);
const needs = ref<SessionSummary[]>([]);
const needsTotal = ref(0);

const upcoming = computed(() => {
  const now = Date.now();
  return events.value
    .filter((event) => (event.start || event.day || "").slice(0, 10) === today)
    .filter((event) => {
      const end = event.end || "";
      if (!end) return true;
      const stamp = end.length <= 10 ? `${end.slice(0, 10)}T23:59:59` : end;
      const when = new Date(stamp).getTime();
      return Number.isNaN(when) || when >= now;
    })
    .sort((a, b) => (a.start || "").localeCompare(b.start || ""))
    .slice(0, 3);
});

function whenOf(event: PlannerEvent): string {
  const minutes = minutesOf(event.start);
  return minutes == null ? "All day" : clockOf(minutes);
}

async function load(): Promise<void> {
  try {
    const day = await api().plannerDay(today);
    events.value = day.events ?? [];
  } catch {
    events.value = [];
  }
  try {
    const page = await api().listSessions({ needs_speakers: true, limit: 3 });
    needs.value = page.data ?? [];
    needsTotal.value = page.facets?.needs_speakers ?? needs.value.length;
  } catch {
    needs.value = [];
    needsTotal.value = 0;
  }
}

async function planDay(): Promise<void> {
  try {
    const rows = await api().listChats({ origin_ref: today });
    const existing = rows
      .filter((chat) => isSideChat(chat) && chat.origin?.kind === "day")
      .sort((a, b) => (b.updated_at ?? "").localeCompare(a.updated_at ?? ""));
    if (existing[0]) {
      openChat(existing[0].id);
      return;
    }
  } catch {
    /* start a new side chat */
  }
  const chat = await chats.createChat({
    title: `Plan ${today}`,
    origin: { kind: "day", ref: today, quote: dayHeading(today) },
  });
  if (chat) openChat(chat.id);
}

onMounted(() => void load());
</script>

<template>
  <div class="home column q-pa-lg">
    <div class="text-h6">Today</div>
    <div class="text-caption text-muted q-mb-md">{{ dayHeading(today) }}</div>

    <q-card flat bordered class="card">
      <q-card-section class="q-pb-xs text-subtitle2">Next events</q-card-section>
      <q-list dense>
        <q-item
          v-for="event in upcoming"
          :key="event.id"
          clickable
          @click="openDay({ date: today, eventId: event.id })"
        >
          <q-item-section side class="clock">{{ whenOf(event) }}</q-item-section>
          <q-item-section>{{ event.title }}</q-item-section>
        </q-item>
        <q-item v-if="!upcoming.length">
          <q-item-section class="text-muted">Nothing else scheduled today.</q-item-section>
        </q-item>
      </q-list>
      <q-card-actions>
        <q-btn flat no-caps icon="calendar_today" label="Open day" @click="openDay({ date: today })" />
        <q-btn flat no-caps icon="auto_awesome" label="Plan my day" @click="planDay" />
      </q-card-actions>
    </q-card>

    <q-card flat bordered class="card q-mt-md">
      <q-card-section class="q-pb-xs text-subtitle2">
        Needs training
        <span v-if="needsTotal" class="text-caption text-muted">· {{ needsTotal }}</span>
      </q-card-section>
      <q-list dense>
        <q-item v-for="session in needs" :key="session.session_id" clickable @click="openSession(session.session_id)">
          <q-item-section avatar><q-icon name="graphic_eq" /></q-item-section>
          <q-item-section>{{ session.title || session.session_id }}</q-item-section>
        </q-item>
        <q-item v-if="!needs.length">
          <q-item-section class="text-muted">Every session has a speaker name.</q-item-section>
        </q-item>
      </q-list>
      <q-card-actions v-if="needsTotal">
        <q-btn flat no-caps label="Open sessions" @click="openSessions()" />
      </q-card-actions>
    </q-card>
  </div>
</template>

<style scoped>
.home {
  max-width: 36rem;
  margin: 0 auto;
  width: 100%;
}
.card {
  background: var(--pawn-bg);
}
.clock {
  font-variant-numeric: tabular-nums;
  min-width: 4.5rem;
}
</style>
