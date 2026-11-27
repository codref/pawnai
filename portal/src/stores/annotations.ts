// Annotations (highlight + note) on any target: chat messages, files, clips…
// Loaded per target in batches (one request for every message of a chat) and
// cached by id; AnnotationLayer components resolve and paint them.
import { defineStore } from "pinia";
import { ref } from "vue";
import {
  api,
  type Annotation,
  type AnnotationCreate,
  type AnnotationKind,
  type AnnotationPatch,
} from "@/api";
import { DEFAULT_COLOR, colorOf, type AnnotationColor } from "@/lib/highlights";
import { notifyError } from "@/lib/errors";
import { loadString, saveString } from "@/lib/storage";

const key = (kind: string, ref: string) => `${kind}:${ref}`;

export interface AnnotationRequest {
  id: string;
  at: number;
}

export const useAnnotationsStore = defineStore("annotations", () => {
  const byId = ref<Record<string, Annotation>>({});
  /** Targets whose annotations have been fetched at least once. */
  const loaded = ref<Record<string, true>>({});
  const lastColor = ref<AnnotationColor>(colorOf(loadString("annotation.color", DEFAULT_COLOR)));
  /** Scroll to + flash an annotation (search hit, side chat back-link). */
  const focusRequest = ref<AnnotationRequest | null>(null);
  /** Open the note editor of an annotation (right after Annotate). */
  const editRequest = ref<AnnotationRequest | null>(null);
  /** Bumped on every change, for views that list annotations. */
  const revision = ref(0);

  function put(a: Annotation): Annotation {
    byId.value[a.id] = a;
    revision.value += 1;
    return a;
  }

  function forTargets(kind: AnnotationKind, refs: string[]): Annotation[] {
    const want = new Set(refs);
    return Object.values(byId.value)
      .filter((a) => a.target_kind === kind && want.has(a.target_ref))
      .sort((a, b) => (a.created_at ?? "").localeCompare(b.created_at ?? ""));
  }

  /** Fetch annotations of *refs* (only the ones not loaded yet, unless *force*). */
  async function load(kind: AnnotationKind, refs: string[], force = false): Promise<void> {
    const todo = [...new Set(refs)].filter((r) => r && (force || !loaded.value[key(kind, r)]));
    if (!todo.length) return;
    try {
      // Long chats: keep each URL a reasonable length.
      for (let i = 0; i < todo.length; i += 50) {
        const batch = todo.slice(i, i + 50);
        const rows = await api().listAnnotations({ target_kind: kind, target_ref: batch });
        const fresh = new Set(rows.map((a) => a.id));
        for (const id of Object.keys(byId.value)) {
          const a = byId.value[id];
          if (a.target_kind === kind && batch.includes(a.target_ref) && !fresh.has(id)) delete byId.value[id];
        }
        for (const a of rows) byId.value[a.id] = a;
        for (const r of batch) loaded.value[key(kind, r)] = true;
      }
      revision.value += 1;
    } catch (e) {
      notifyError(e, "Could not load annotations");
    }
  }

  async function create(body: AnnotationCreate): Promise<Annotation | null> {
    try {
      const a = put(await api().createAnnotation({ color: lastColor.value, ...body }));
      loaded.value[key(a.target_kind, a.target_ref)] = true;
      return a;
    } catch (e) {
      notifyError(e, "Could not annotate");
      return null;
    }
  }

  /** Optimistic update; rolls back on failure. */
  async function patch(id: string, body: AnnotationPatch): Promise<Annotation | null> {
    const before = byId.value[id];
    if (!before) return null;
    put({ ...before, ...body, tags: body.tags ?? before.tags, anchor: body.anchor ?? before.anchor } as Annotation);
    if (body.color) setLastColor(body.color);
    try {
      return put(await api().patchAnnotation(id, body));
    } catch (e) {
      put(before);
      notifyError(e, "Could not update annotation");
      return null;
    }
  }

  async function remove(id: string): Promise<boolean> {
    const before = byId.value[id];
    if (!before) return false;
    delete byId.value[id];
    revision.value += 1;
    try {
      await api().deleteAnnotation(id);
      return true;
    } catch (e) {
      put(before);
      notifyError(e, "Could not delete annotation");
      return false;
    }
  }

  function setLastColor(color: string): void {
    lastColor.value = colorOf(color);
    saveString("annotation.color", lastColor.value);
  }

  function requestFocus(id: string): void {
    focusRequest.value = { id, at: Date.now() };
  }

  function requestEdit(id: string): void {
    editRequest.value = { id, at: Date.now() };
  }

  return {
    byId,
    lastColor,
    focusRequest,
    editRequest,
    revision,
    forTargets,
    load,
    create,
    patch,
    remove,
    setLastColor,
    requestFocus,
    requestEdit,
  };
});
