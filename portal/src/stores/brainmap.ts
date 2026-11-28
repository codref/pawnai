// One store per brainmap board: the graph from the server, the saved layout,
// filters, and every human verdict (tie, cut, confirm, snip, lasso, accept,
// burn). Mutations are optimistic and roll back with a toast on failure, so
// the board feels like paper and string, not like a form.
import { defineStore } from "pinia";
import { computed, ref, shallowRef } from "vue";
import { Notify } from "quasar";
import {
  api,
  type BrainEdge,
  type BrainGraph,
  type BrainKind,
  type BrainLayout,
  type BrainNode,
  type BrainPos,
  type BrainViewport,
  type EmergenceStatus,
  type ItemAction,
} from "@/api";
import { notifyError } from "@/lib/errors";
import { touchesInbox } from "@/lib/inbox";
import { useFilesStore } from "@/stores/files";
import { KIND_FILTERS } from "@/lib/brainmap/boards";
import {
  boardCounts,
  emptyGraph,
  filterGraph,
  interrogationQueue,
  mergeGraph,
  newSince,
  removeNode,
  replaceNode,
  withRemoveVerb,
} from "@/lib/brainmap/graph";
import { loadJson, saveJson } from "@/lib/storage";

const OPEN_ITEM = new Set(["new", "notified", "snoozed"]);

function toast(message: string): void {
  Notify.create({ message, position: "bottom-right", timeout: 2400, classes: "brainmap-toast" });
}

/** Vault writes that can change a board: items, threads, hunches, goals. */
export function touchesBoard(key: string): boolean {
  return (
    touchesInbox(key) ||
    key.includes("/Threads/") ||
    key.includes("/Questions/") ||
    key.includes("/Decisions/") ||
    key.includes("/Hypotheses/")
  );
}

interface SavedFilters {
  hidden: BrainKind[];
  showSuggestions: boolean;
  showDone: boolean;
}

export function useBrainmapStore(board: string) {
  return defineStore(`brainmap:${board}`, () => {
    const graph = shallowRef<BrainGraph>(emptyGraph(board));
    const loading = ref(false);
    const loaded = ref(false);
    const positions = shallowRef<Record<string, BrainPos>>({});
    const viewport = ref<BrainViewport | null>(null);
    const saved = loadJson<SavedFilters>(`brainmap.${board}.filters`, {
      hidden: [],
      showSuggestions: true,
      showDone: false,
    });
    const hidden = ref<Set<BrainKind>>(new Set(saved.hidden));
    const showSuggestions = ref(saved.showSuggestions);
    const showDone = ref(saved.showDone);
    /** Ids animating out (burn) or in (accept flash). */
    const burning = ref<Set<string>>(new Set());
    const flashing = ref<Set<string>>(new Set());
    const busy = ref<Set<string>>(new Set());

    /** Cards the user selected (shift-click, lasso): what the board chat asks about. */
    const selection = ref<Set<string>>(new Set());
    const meta = computed(() => graph.value.meta ?? null);
    /** User boards (not the built-in Items board) hold pins and note cards. */
    const custom = computed(() => board !== "items");
    const newIds = computed(() => newSince(graph.value));

    const kinds = computed(() => new Set(KIND_FILTERS.map((k) => k.kind).filter((k) => !hidden.value.has(k))));
    const visible = computed(() =>
      filterGraph(graph.value, {
        kinds: kinds.value,
        showSuggestions: showSuggestions.value,
        showDone: showDone.value,
      }),
    );
    const counts = computed(() => boardCounts(visible.value));
    const queue = computed(() => interrogationQueue(visible.value));
    const nodeById = computed(() => new Map(graph.value.nodes.map((n) => [n.id, n])));

    function persistFilters(): void {
      saveJson(`brainmap.${board}.filters`, {
        hidden: [...hidden.value],
        showSuggestions: showSuggestions.value,
        showDone: showDone.value,
      });
    }

    function toggleKind(kind: BrainKind): void {
      const next = new Set(hidden.value);
      if (next.has(kind)) next.delete(kind);
      else next.add(kind);
      hidden.value = next;
      persistFilters();
    }

    function setShow(which: "suggestions" | "done", on: boolean): void {
      if (which === "suggestions") showSuggestions.value = on;
      else showDone.value = on;
      persistFilters();
    }

    function mark(set: typeof busy, id: string, on: boolean): void {
      const next = new Set(set.value);
      if (on) next.add(id);
      else next.delete(id);
      set.value = next;
    }

    async function load(focus = ""): Promise<void> {
      loading.value = true;
      try {
        const [raw, layout] = await Promise.all([
          api().brainmap(board, focus ? { focus } : {}),
          loaded.value ? Promise.resolve(null) : api().getBrainmapLayout(board).catch(() => null),
        ]);
        const g = board === "items" ? raw : withRemoveVerb(raw);
        // Keep spilled evidence the user pulled in, if it still exists server side.
        const spilled = graph.value.nodes.filter((n) => n.kind === "scrap");
        graph.value = spilled.length
          ? mergeGraph(g, {
              nodes: spilled,
              edges: graph.value.edges.filter((e) => spilled.some((s) => s.id === e.from || s.id === e.to)),
            })
          : g;
        if (layout) {
          positions.value = layout.positions || {};
          viewport.value = layout.viewport;
        }
        loaded.value = true;
      } catch (e) {
        notifyError(e, "Could not load the board");
      } finally {
        loading.value = false;
      }
    }

    // ── layout ──────────────────────────────────────────────────────────
    let saveTimer: ReturnType<typeof setTimeout> | undefined;
    function saveLayout(next: Record<string, BrainPos>, view: BrainViewport | null): void {
      positions.value = next;
      viewport.value = view;
      clearTimeout(saveTimer);
      saveTimer = setTimeout(() => {
        const body: BrainLayout = { positions: positions.value, viewport: viewport.value };
        api()
          .putBrainmapLayout(board, body)
          .catch((e) => notifyError(e, "Could not save the board layout"));
      }, 1500);
    }

    // ── strings ─────────────────────────────────────────────────────────
    async function tie(from: string, to: string, role: string): Promise<BrainEdge | null> {
      const temp: BrainEdge = { id: `pending:${from}->${to}`, from, to, role, by: "user", cuttable: false };
      const before = graph.value;
      graph.value = { ...before, edges: [...before.edges, temp] };
      try {
        const res = await api().brainmapTie(board, from, to, role);
        graph.value = {
          ...graph.value,
          edges: graph.value.edges.map((e) => (e.id === temp.id ? res.edge : e)),
          suggestions: graph.value.suggestions.filter((s) => !(s.from === from && s.to === to)),
        };
        toast(res.receipt || "Tied. Pawn will learn from it.");
        return res.edge;
      } catch (e) {
        graph.value = { ...graph.value, edges: graph.value.edges.filter((x) => x.id !== temp.id) };
        notifyError(e, "Could not tie the string");
        return null;
      }
    }

    async function cut(edge: BrainEdge): Promise<boolean> {
      const before = graph.value;
      graph.value = { ...before, edges: before.edges.filter((e) => e.id !== edge.id) };
      try {
        const res = await api().brainmapCut(board, edge.id);
        toast(res.receipt || "Cut. Pawn will not tie these again.");
        return true;
      } catch (e) {
        graph.value = { ...graph.value, edges: [...graph.value.edges, edge] };
        notifyError(e, "Could not cut the string");
        return false;
      }
    }

    async function verdict(id: string, v: "confirm" | "snip"): Promise<void> {
      const sugg = graph.value.suggestions.find((s) => s.id === id);
      if (!sugg) return;
      graph.value = { ...graph.value, suggestions: graph.value.suggestions.filter((s) => s.id !== id) };
      try {
        const res = await api().brainmapSuggestion(board, id, v);
        if (res.edge) graph.value = { ...graph.value, edges: [...graph.value.edges, res.edge] };
        toast(res.receipt || (v === "confirm" ? "Tied." : "Snipped."));
        if (res.reload) void load();
      } catch (e) {
        graph.value = { ...graph.value, suggestions: [...graph.value.suggestions, sugg] };
        notifyError(e, v === "confirm" ? "Could not tie the string" : "Could not snip the string");
      }
    }

    async function lasso(nodeIds: string[], title?: string, into?: string): Promise<BrainNode | null> {
      try {
        const res = await api().brainmapLasso(board, { node_ids: nodeIds, title, into });
        graph.value = mergeGraph(into ? graph.value : { ...graph.value, nodes: [...graph.value.nodes, res.subject] }, {
          edges: res.edges,
        });
        flash(res.subject.id);
        toast(into ? `Filed ${nodeIds.length} under “${res.subject.title}”.` : `New thread “${res.subject.title}”.`);
        return res.subject;
      } catch (e) {
        notifyError(e, "Could not make the thread");
        return null;
      }
    }

    async function spill(node: BrainNode): Promise<number> {
      const subjectId = node.meta.subject_id;
      if (!subjectId) return 0;
      mark(busy, node.id, true);
      try {
        const res = await api().brainmapSpill(board, subjectId);
        const before = graph.value.nodes.length;
        graph.value = mergeGraph(graph.value, res);
        const added = graph.value.nodes.length - before;
        toast(added ? `Pulled ${added} scrap(s) of evidence.` : "Nothing more on this thread.");
        return added;
      } catch (e) {
        notifyError(e, "Could not pull the thread");
        return 0;
      } finally {
        mark(busy, node.id, false);
      }
    }

    // ── cards ───────────────────────────────────────────────────────────
    function flash(id: string): void {
      mark(flashing, id, true);
      setTimeout(() => mark(flashing, id, false), 900);
    }

    function burn(id: string): Promise<void> {
      mark(burning, id, true);
      return new Promise((resolve) =>
        setTimeout(() => {
          graph.value = removeNode(graph.value, id);
          mark(burning, id, false);
          resolve();
        }, 750),
      );
    }

    async function itemAction(node: BrainNode, action: ItemAction, arg?: string): Promise<void> {
      const itemId = node.meta.item_id;
      if (!itemId) return;
      mark(busy, node.id, true);
      try {
        const res = await api().itemAction(itemId, action, arg);
        toast(res.receipt);
        if (action === "delete" || !OPEN_ITEM.has(res.item.status)) {
          if (action === "delete" || action === "ignore" || action === "reject") await burn(node.id);
          else {
            flash(node.id);
            graph.value = replaceNode(graph.value, { ...node, status: res.item.status });
          }
        }
      } catch (e) {
        notifyError(e, "Could not update the item");
      } finally {
        mark(busy, node.id, false);
      }
    }

    async function subjectStatus(node: BrainNode, status: EmergenceStatus): Promise<void> {
      const subjectId = node.meta.subject_id;
      if (!subjectId) return;
      mark(busy, node.id, true);
      try {
        const res = await api().emergenceStatus(subjectId, status);
        if (status === "dismissed") {
          toast(`Burned “${node.title}”. Pawn will learn from it.`);
          await burn(node.id);
        } else {
          flash(node.id);
          graph.value = replaceNode(graph.value, {
            ...node,
            status: res.subject.status,
            proposed: res.subject.status === "proposed",
            actions: node.actions.filter((a) => a !== "accept"),
          });
          toast(status === "active" ? `Accepted “${node.title}”.` : `Marked ${status}.`);
        }
      } catch (e) {
        notifyError(e, "Could not update the card");
      } finally {
        mark(busy, node.id, false);
      }
    }

    // ── user boards: selection, membership, proposals ──────────────────────
    function setSelection(ids: Iterable<string>): void {
      selection.value = new Set(ids);
    }

    function toggleSelect(id: string): void {
      const next = new Set(selection.value);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      selection.value = next;
    }

    /** Accept or burn a ghost card (a board proposal). */
    async function decide(node: BrainNode, v: "accept" | "reject"): Promise<void> {
      const pid = node.meta.proposal_id as string | undefined;
      if (!pid) return;
      mark(busy, node.id, true);
      try {
        const res = await api().decideBoardProposal(board, pid, v);
        if (v === "reject") await burn(node.id);
        toast(res.receipt);
        await load();
        if (v === "accept") flash(node.id.startsWith("prop:") ? `note:${pid}` : node.id);
      } catch (e) {
        notifyError(e, v === "accept" ? "Could not accept the proposal" : "Could not burn the proposal");
      } finally {
        mark(busy, node.id, false);
      }
    }

    async function pinCards(ids: string[]): Promise<boolean> {
      try {
        await api().pinToBoard(board, ids);
        await load();
        return true;
      } catch (e) {
        notifyError(e, "Could not pin to the board");
        return false;
      }
    }

    async function removeCards(ids: string[]): Promise<void> {
      const before = graph.value;
      for (const id of ids) graph.value = removeNode(graph.value, id);
      try {
        await api().removeFromBoard(board, ids);
        const next = new Set(selection.value);
        ids.forEach((i) => next.delete(i));
        selection.value = next;
        toast(ids.length > 1 ? `Took ${ids.length} cards off the board.` : "Took it off the board.");
      } catch (e) {
        graph.value = before;
        notifyError(e, "Could not take the card off the board");
      }
    }

    async function markSeen(): Promise<void> {
      try {
        const b = await api().boardSeen(board);
        graph.value = {
          ...graph.value,
          meta: graph.value.meta ? { ...graph.value.meta, seen_at: b.seen_at } : graph.value.meta,
          nodes: graph.value.nodes.map((n) => (n.meta?.new ? { ...n, meta: { ...n.meta, new: false } } : n)),
          edges: graph.value.edges.map((e) => (e.new ? { ...e, new: false } : e)),
        };
      } catch (e) {
        notifyError(e, "Could not mark the board as seen");
      }
    }

    // ── live updates ────────────────────────────────────────────────────
    let viewers = 0;
    let unwatch: (() => void) | null = null;
    let reloadTimer: ReturnType<typeof setTimeout> | undefined;

    function attach(focus = ""): void {
      viewers += 1;
      if (viewers === 1) {
        void load(focus);
        unwatch = useFilesStore().onKeys(touchesBoard, () => {
          clearTimeout(reloadTimer);
          reloadTimer = setTimeout(() => void load(), 1200);
        });
      }
    }

    function detach(): void {
      viewers = Math.max(0, viewers - 1);
      if (!viewers) {
        unwatch?.();
        unwatch = null;
        clearTimeout(reloadTimer);
      }
    }

    return {
      board,
      graph,
      visible,
      counts,
      queue,
      nodeById,
      loading,
      loaded,
      positions,
      viewport,
      hidden,
      showSuggestions,
      showDone,
      burning,
      flashing,
      busy,
      selection,
      meta,
      custom,
      newIds,
      setSelection,
      toggleSelect,
      decide,
      pinCards,
      removeCards,
      markSeen,
      load,
      saveLayout,
      toggleKind,
      setShow,
      tie,
      cut,
      verdict,
      lasso,
      spill,
      itemAction,
      subjectStatus,
      burn,
      attach,
      detach,
    };
  })();
}

export type BrainmapStore = ReturnType<typeof useBrainmapStore>;
