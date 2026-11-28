// Sessions browse state plus the header badge. `needsSpeakers` is the
// unfiltered facet, so a search or date range does not change the badge.
import { defineStore } from "pinia";
import { ref } from "vue";
import { api, type SessionFacets, type SessionSummary } from "@/api";
import { errorText } from "@/lib/errors";
import { useFilesStore } from "./files";

export type SessionScope = "all" | "needs";
export type SessionRange = "all" | "today" | "week" | "month";
export type SessionSort = "recent" | "oldest" | "duration";

const PAGE = 40;

function dayStamp(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

function shiftDays(days: number): string {
  const date = new Date();
  date.setDate(date.getDate() + days);
  return dayStamp(date);
}

export const useSessionsStore = defineStore("sessions", () => {
  const needsSpeakers = ref(0);
  const q = ref("");
  const scope = ref<SessionScope>("all");
  const range = ref<SessionRange>("all");
  const speaker = ref("");
  const sort = ref<SessionSort>("recent");
  const rows = ref<SessionSummary[]>([]);
  const facets = ref<SessionFacets>({ speakers: [], needs_speakers: 0, tags: [], total: 0 });
  const loading = ref(false);
  const error = ref("");
  const done = ref(false);
  const selected = ref<string[]>([]);
  /** Session tab that should focus the speakers rail on the next open. */
  const focusRail = ref("");
  let stop: (() => void) | null = null;
  let seq = 0;
  let browsing = false;

  function dates(): { from?: string; to?: string } {
    const today = dayStamp(new Date());
    if (range.value === "today") return { from: today, to: today };
    if (range.value === "week") return { from: shiftDays(-6), to: today };
    if (range.value === "month") return { from: shiftDays(-29), to: today };
    return {};
  }

  async function refreshCount(): Promise<void> {
    try {
      const page = await api().listSessions({ limit: 1 });
      needsSpeakers.value = page.facets.needs_speakers ?? 0;
    } catch {
      /* the badge stays at the last known count */
    }
  }

  async function fetchPage(reset: boolean): Promise<void> {
    const token = ++seq;
    loading.value = true;
    error.value = "";
    browsing = true;
    const offset = reset ? 0 : rows.value.length;
    try {
      const page = await api().listSessions({
        q: q.value.trim() || undefined,
        limit: PAGE,
        offset,
        needs_speakers: scope.value === "needs" ? true : undefined,
        speaker: speaker.value || undefined,
        sort: sort.value,
        ...dates(),
      });
      if (token !== seq) return;
      rows.value = reset ? page.data : rows.value.concat(page.data);
      facets.value = page.facets;
      needsSpeakers.value = page.facets.needs_speakers ?? needsSpeakers.value;
      const total = page.facets.total ?? rows.value.length;
      done.value = page.data.length < PAGE || rows.value.length >= total;
    } catch (e) {
      if (token !== seq) return;
      error.value = errorText(e);
    } finally {
      if (token === seq) loading.value = false;
    }
  }

  function reload(): Promise<void> {
    selected.value = [];
    return fetchPage(true);
  }

  function more(): Promise<void> {
    if (loading.value || done.value) return Promise.resolve();
    return fetchPage(false);
  }

  function setSpeaker(name: string): void {
    speaker.value = speaker.value === name ? "" : name;
    void reload();
  }

  function toggleSelected(id: string, on?: boolean): void {
    const has = selected.value.includes(id);
    const next = on ?? !has;
    selected.value = next ? (has ? selected.value : selected.value.concat(id)) : selected.value.filter((item) => item !== id);
  }

  function selectRange(ids: string[]): void {
    selected.value = ids;
  }

  function clearSelected(): void {
    selected.value = [];
  }

  function watch(): void {
    if (stop) return;
    const files = useFilesStore();
    const prefix = `${files.agentRoot || "Pawn"}/Transcripts/`;
    stop = files.onVaultChange(prefix, () => {
      void refreshCount();
      if (browsing) void fetchPage(true);
    });
    void refreshCount();
  }

  function stopWatch(): void {
    stop?.();
    stop = null;
  }

  return {
    needsSpeakers,
    q,
    scope,
    range,
    speaker,
    sort,
    rows,
    facets,
    loading,
    error,
    done,
    selected,
    focusRail,
    refreshCount,
    reload,
    more,
    setSpeaker,
    toggleSelected,
    selectRange,
    clearSelected,
    watch,
    stopWatch,
  };
});
