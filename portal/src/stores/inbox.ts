// Inbox state: open counts for the header badge, coworker items (filters,
// paging, triage), idea notes and research capture atoms. Vault writes under
// the Inbox folders (lib/inbox.ts touchesInbox) refresh what is loaded.
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { Notify } from "quasar";
import {
  api,
  type CaptureAtom,
  type CaptureFileBody,
  type Idea,
  type IdeaActionKind,
  type InboxCounts,
  type InboxItem,
  type ItemAction,
} from "@/api";
import { loadString, saveString } from "@/lib/storage";
import { notifyError } from "@/lib/errors";
import { PAGE_SIZE, statusesForScope, touchesInbox, type ItemsGroupBy, type ItemsScope } from "@/lib/inbox";
import { useFilesStore } from "./files";

export type InboxSection = "items" | "ideas" | "captures";

function toast(message: string): void {
  Notify.create({ message, position: "bottom-right", timeout: 2200 });
}

export const useInboxStore = defineStore("inbox", () => {
  const counts = ref<InboxCounts>({ items: 0, ideas: 0, captures: 0 });
  const total = computed(() => counts.value.items + counts.value.ideas + counts.value.captures);

  const section = ref<InboxSection>((loadString("inbox.section") as InboxSection) || "items");
  const scope = ref<ItemsScope>((loadString("inbox.scope") as ItemsScope) || "open");
  const groupBy = ref<ItemsGroupBy>((loadString("inbox.groupBy") as ItemsGroupBy) || "day");
  const kind = ref("");
  const q = ref("");
  const sourceRef = ref("");

  const items = ref<InboxItem[]>([]);
  const itemsTotal = ref(0);
  const loadingItems = ref(false);
  const busy = ref<Set<string>>(new Set());
  const ideas = ref<Idea[]>([]);
  const captures = ref<CaptureAtom[]>([]);
  const loadingVault = ref(false);
  /** Panels open: only then do vault events reload the lists. */
  let viewers = 0;
  let unwatch: (() => void) | null = null;
  let refreshTimer: ReturnType<typeof setTimeout> | null = null;

  async function refreshCounts(): Promise<void> {
    try {
      counts.value = await api().inboxCounts();
    } catch {
      /* the badge is best effort */
    }
  }

  function setSection(s: InboxSection): void {
    section.value = s;
    saveString("inbox.section", s);
  }

  function setQuery(p: { scope?: ItemsScope; kind?: string; q?: string; sourceRef?: string; groupBy?: ItemsGroupBy }): void {
    if (p.scope !== undefined) {
      scope.value = p.scope;
      saveString("inbox.scope", p.scope);
    }
    if (p.groupBy !== undefined) {
      groupBy.value = p.groupBy;
      saveString("inbox.groupBy", p.groupBy);
    }
    if (p.kind !== undefined) kind.value = p.kind;
    if (p.q !== undefined) q.value = p.q;
    if (p.sourceRef !== undefined) sourceRef.value = p.sourceRef;
    if (p.scope !== undefined || p.kind !== undefined || p.q !== undefined || p.sourceRef !== undefined) {
      void loadItems();
    }
  }

  async function loadItems(more = false): Promise<void> {
    loadingItems.value = true;
    try {
      const page = await api().listItems({
        statuses: statusesForScope(scope.value),
        kind: kind.value || undefined,
        q: q.value.trim() || undefined,
        source_ref: sourceRef.value || undefined,
        limit: PAGE_SIZE,
        offset: more ? items.value.length : 0,
      });
      items.value = more ? [...items.value, ...page.items] : page.items;
      itemsTotal.value = page.total;
    } catch (e) {
      notifyError(e, "Could not load items");
    } finally {
      loadingItems.value = false;
    }
  }

  async function loadVault(): Promise<void> {
    loadingVault.value = true;
    try {
      const [i, c] = await Promise.all([api().listIdeas("inbox"), api().listCaptureAtoms("inbox")]);
      ideas.value = i;
      captures.value = c;
    } catch (e) {
      notifyError(e, "Could not load the inbox");
    } finally {
      loadingVault.value = false;
    }
  }

  function refreshAll(): Promise<unknown> {
    return Promise.all([refreshCounts(), viewers ? loadItems() : null, viewers ? loadVault() : null]);
  }

  /** Start watching vault events (call once at app start). */
  function watch(): void {
    if (unwatch) return;
    unwatch = useFilesStore().onKeys(touchesInbox, () => {
      if (refreshTimer) clearTimeout(refreshTimer);
      refreshTimer = setTimeout(() => void refreshAll(), 400);
    });
  }

  function attach(): void {
    viewers += 1;
    if (viewers === 1) void refreshAll();
  }

  function detach(): void {
    viewers = Math.max(0, viewers - 1);
  }

  function mark(id: string, on: boolean): void {
    const next = new Set(busy.value);
    if (on) next.add(id);
    else next.delete(id);
    busy.value = next;
  }

  /** Triage one item; the row is replaced, or dropped when it leaves the scope. */
  async function act(item: InboxItem, action: ItemAction, arg?: string): Promise<boolean> {
    mark(item.id, true);
    try {
      const res = await api().itemAction(item.id, action, arg);
      toast(res.receipt);
      const stays =
        scope.value === "all" ||
        (scope.value === "open" ? ["new", "notified", "snoozed"] : ["filed", "task", "dismissed"]).includes(
          res.item.status,
        );
      items.value = stays
        ? items.value.map((it) => (it.id === item.id ? res.item : it))
        : items.value.filter((it) => it.id !== item.id);
      if (!stays) itemsTotal.value = Math.max(0, itemsTotal.value - 1);
      void refreshCounts();
      return true;
    } catch (e) {
      notifyError(e, `Could not ${action} item`);
      return false;
    } finally {
      mark(item.id, false);
    }
  }

  async function deleteMany(ids: string[]): Promise<void> {
    try {
      const res = await api().deleteItems({ ids });
      toast(`Deleted ${res.deleted} item(s).`);
    } catch (e) {
      notifyError(e, "Could not delete items");
    }
    await Promise.all([loadItems(), refreshCounts()]);
  }

  async function deleteAllOpen(): Promise<void> {
    try {
      const res = await api().deleteItems({ all_open: true, kind: kind.value || undefined, q: q.value || undefined });
      toast(`Deleted ${res.deleted} item(s).`);
    } catch (e) {
      notifyError(e, "Could not delete items");
    }
    await Promise.all([loadItems(), refreshCounts()]);
  }

  async function ideaAction(idea: Idea, action: IdeaActionKind, title?: string): Promise<string | null> {
    mark(idea.path, true);
    try {
      const res = await api().ideaAction(idea.path, action, title);
      toast(res.receipt);
      ideas.value = ideas.value.filter((i) => i.path !== idea.path);
      void refreshCounts();
      return res.written;
    } catch (e) {
      notifyError(e, "Could not update the idea");
      return null;
    } finally {
      mark(idea.path, false);
    }
  }

  async function captureIdea(line: string): Promise<boolean> {
    try {
      toast(await api().captureIdea(line));
      await Promise.all([loadVault(), refreshCounts()]);
      return true;
    } catch (e) {
      notifyError(e, "Could not capture the idea");
      return false;
    }
  }

  async function fileCapture(body: CaptureFileBody): Promise<boolean> {
    mark(body.path, true);
    try {
      await api().fileCapture(body);
      toast(body.ignore ? "Ignored." : "Filed.");
      captures.value = captures.value.filter((c) => c.path !== body.path);
      void refreshCounts();
      return true;
    } catch (e) {
      notifyError(e, body.ignore ? "Could not ignore the capture" : "Could not file the capture");
      return false;
    } finally {
      mark(body.path, false);
    }
  }

  return {
    counts,
    total,
    section,
    scope,
    groupBy,
    kind,
    q,
    sourceRef,
    items,
    itemsTotal,
    loadingItems,
    busy,
    ideas,
    captures,
    loadingVault,
    refreshCounts,
    setSection,
    setQuery,
    loadItems,
    loadVault,
    watch,
    attach,
    detach,
    act,
    deleteMany,
    deleteAllOpen,
    ideaAction,
    captureIdea,
    fileCapture,
  };
});
