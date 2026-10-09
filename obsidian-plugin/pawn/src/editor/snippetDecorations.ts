import {
  Decoration,
  DecorationSet,
  EditorView,
  MatchDecorator,
  ViewPlugin,
  ViewUpdate,
  WidgetType,
} from "@codemirror/view";
import type { Extension } from "@codemirror/state";
import type { App } from "obsidian";
import { Notice } from "obsidian";
import type { PawnClient } from "../api";

/** Minimal host so this module does not import main.ts. */
export interface SnippetDecorationHost {
  app: App;
  client: PawnClient;
}

const SNIPPET_RE = /<!--\s*(\/?)pawn-snippet:([A-Za-z0-9_-]+)\s*-->/g;

function shortId(id: string): string {
  return id.length > 10 ? `${id.slice(0, 8)}…` : id;
}

function findSnippetRange(
  doc: { sliceString: (from: number, to: number) => string; length: number },
  snippetId: string,
): { from: number; to: number } | null {
  const text = doc.sliceString(0, doc.length);
  const open = new RegExp(
    `<!--\\s*pawn-snippet:${escapeRegExp(snippetId)}\\s*-->`,
  );
  const close = new RegExp(
    `<!--\\s*/pawn-snippet:${escapeRegExp(snippetId)}\\s*-->`,
  );
  const openMatch = open.exec(text);
  if (!openMatch) return null;
  const from = openMatch.index;
  close.lastIndex = from + openMatch[0].length;
  const closeMatch = close.exec(text);
  if (!closeMatch) return null;
  let to = closeMatch.index + closeMatch[0].length;
  if (text[to] === "\n") to += 1;
  return { from, to };
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

class SnippetChipWidget extends WidgetType {
  constructor(
    private readonly snippetId: string,
    private readonly closing: boolean,
    private readonly host: SnippetDecorationHost,
  ) {
    super();
  }

  eq(other: SnippetChipWidget): boolean {
    return (
      other.snippetId === this.snippetId &&
      other.closing === this.closing &&
      other.host === this.host
    );
  }

  toDOM(view: EditorView): HTMLElement {
    if (this.closing) {
      const end = document.createElement("span");
      end.className = "pawn-snippet-end";
      end.title = `End of snippet ${this.snippetId}`;
      end.textContent = "⌟";
      return end;
    }

    const wrap = document.createElement("span");
    wrap.className = "pawn-snippet-chip";
    wrap.contentEditable = "false";

    const label = document.createElement("span");
    label.className = "pawn-snippet-chip-label";
    label.textContent = `Snippet ${shortId(this.snippetId)}`;
    label.title = this.snippetId;
    wrap.appendChild(label);

    const del = document.createElement("button");
    del.type = "button";
    del.className = "pawn-snippet-chip-btn";
    del.setAttribute("aria-label", "Delete snippet");
    del.title = "Delete snippet";
    del.textContent = "×";
    del.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
    del.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      void this.deleteSnippet(view);
    });
    wrap.appendChild(del);

    return wrap;
  }

  ignoreEvent(): boolean {
    return true;
  }

  private async deleteSnippet(view: EditorView): Promise<void> {
    const range = findSnippetRange(view.state.doc, this.snippetId);
    if (!range) {
      new Notice("Could not find snippet markers in the note.");
      return;
    }
    view.dispatch({
      changes: { from: range.from, to: range.to, insert: "" },
    });

    const file = this.host.app.workspace.getActiveFile();
    if (!file) return;
    try {
      await this.host.client.deleteCaptureSnippet(this.snippetId, file.path);
    } catch {
      // Editor already updated; Sync Engine / offline is fine without API.
    }
  }
}

export function snippetDecorationsExtension(host: SnippetDecorationHost): Extension {
  const decorator = new MatchDecorator({
    regexp: SNIPPET_RE,
    decoration: (match) => {
      const closing = match[1] === "/";
      const id = match[2];
      return Decoration.replace({
        widget: new SnippetChipWidget(id, closing, host),
      });
    },
  });

  return ViewPlugin.fromClass(
    class {
      decorations: DecorationSet;

      constructor(view: EditorView) {
        this.decorations = decorator.createDeco(view);
      }

      update(update: ViewUpdate): void {
        this.decorations = decorator.updateDeco(update, this.decorations);
      }
    },
    { decorations: (v) => v.decorations },
  );
}
