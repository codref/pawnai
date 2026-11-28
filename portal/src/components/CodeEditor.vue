<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, lineNumbers, highlightActiveLine, drawSelection } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { markdown } from "@codemirror/lang-markdown";
import { syntaxHighlighting, defaultHighlightStyle, bracketMatching } from "@codemirror/language";

const props = defineProps<{ modelValue: string }>();
const emit = defineEmits<{ "update:modelValue": [value: string]; save: [] }>();

const host = ref<HTMLElement | null>(null);
let view: EditorView | null = null;

onMounted(() => {
  view = new EditorView({
    parent: host.value!,
    state: EditorState.create({
      doc: props.modelValue,
      extensions: [
        lineNumbers(),
        history(),
        drawSelection(),
        highlightActiveLine(),
        bracketMatching(),
        markdown(),
        syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
        EditorView.lineWrapping,
        keymap.of([
          { key: "Mod-s", preventDefault: true, run: () => (emit("save"), true) },
          indentWithTab,
          ...defaultKeymap,
          ...historyKeymap,
        ]),
        EditorView.updateListener.of((u) => {
          if (u.docChanged) emit("update:modelValue", u.state.doc.toString());
        }),
        EditorView.theme({
          "&": { height: "100%", fontSize: "14px" },
          ".cm-scroller": { fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" },
        }),
      ],
    }),
  });
  view.focus();
});

// External replacement (reload after a conflict).
watch(
  () => props.modelValue,
  (value) => {
    if (view && value !== view.state.doc.toString()) {
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: value } });
    }
  },
);

onBeforeUnmount(() => view?.destroy());
</script>

<template>
  <div ref="host" class="code-editor" />
</template>

<style scoped>
.code-editor {
  height: 100%;
  border: 1px solid var(--pawn-border);
  border-radius: 4px;
  overflow: hidden;
}
.code-editor :deep(.cm-editor) {
  height: 100%;
}
.code-editor :deep(.cm-editor.cm-focused) {
  outline: none;
}
body.body--dark .code-editor :deep(.cm-gutters) {
  background: #1d1d1d;
  border-color: var(--pawn-border);
  color: #888;
}
body.body--dark .code-editor :deep(.cm-activeLine),
body.body--dark .code-editor :deep(.cm-activeLineGutter) {
  background: rgba(255, 255, 255, 0.04);
}
body.body--dark .code-editor :deep(.cm-content) {
  caret-color: #fff;
}
body.pawn-skin-cyberpunk .code-editor :deep(.cm-gutters) {
  background: #0d141c;
  border-color: var(--pawn-border);
  color: #7f9eae;
}
body.pawn-skin-cyberpunk .code-editor :deep(.cm-activeLine),
body.pawn-skin-cyberpunk .code-editor :deep(.cm-activeLineGutter) {
  background: rgba(92, 225, 255, 0.06);
}
body.pawn-skin-cyberpunk .code-editor :deep(.cm-content) {
  caret-color: #5ce1ff;
}
body.pawn-skin-cyberpunk.body--light .code-editor :deep(.cm-gutters) {
  background: #e7f3f6;
  color: #3d6270;
}
body.pawn-skin-cyberpunk.body--light .code-editor :deep(.cm-activeLine),
body.pawn-skin-cyberpunk.body--light .code-editor :deep(.cm-activeLineGutter) {
  background: rgba(12, 106, 128, 0.06);
}
body.pawn-skin-cyberpunk.body--light .code-editor :deep(.cm-content) {
  caret-color: #0c6a80;
}
</style>
