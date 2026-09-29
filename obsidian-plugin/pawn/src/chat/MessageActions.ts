import { App, Notice, setIcon } from "obsidian";
import { appendToNote, insertAtCursor, replaceSelection, saveAsNewNote } from "../editing";
import type { SelectionRef } from "./conversations";

export interface ExtraAction {
  icon: string;
  label: string;
  onClick: () => void;
}

export interface MessageActionOptions {
  app: App;
  text: string;
  selection?: SelectionRef;
  notePath?: string;
  saveFolder: string;
  extra?: ExtraAction[];
}

function actionButton(
  bar: HTMLElement,
  icon: string,
  label: string,
  onClick: () => void,
): void {
  const b = bar.createEl("button", { cls: "clickable-icon", attr: { "aria-label": label } });
  setIcon(b, icon);
  b.onclick = (ev) => {
    ev.stopPropagation();
    onClick();
  };
}

/** Compact toolbar under a user bubble: copy + reuse in composer. */
export function renderUserMessageActions(
  parent: HTMLElement,
  opts: { text: string; onReuse: () => void },
): void {
  const bar = parent.createDiv({ cls: "pawn-msg-actions is-user" });
  actionButton(bar, "copy", "Copy", () => {
    void navigator.clipboard.writeText(opts.text).then(() => new Notice("Copied"));
  });
  actionButton(bar, "pencil", "Reuse in composer", opts.onReuse);
}

/** Toolbar under an assistant reply or job result. */
export function renderMessageActions(parent: HTMLElement, opts: MessageActionOptions): void {
  const bar = parent.createDiv({ cls: "pawn-msg-actions" });
  actionButton(bar, "copy", "Copy", () => {
    void navigator.clipboard.writeText(opts.text).then(() => new Notice("Copied"));
  });
  actionButton(bar, "text-cursor-input", "Insert at cursor", () =>
    insertAtCursor(opts.app, opts.text),
  );
  actionButton(bar, "replace", "Replace selection (preview)", () =>
    replaceSelection(opts.app, opts.text, opts.selection),
  );
  actionButton(bar, "list-end", "Append to note", () =>
    void appendToNote(opts.app, opts.text, opts.notePath),
  );
  actionButton(bar, "file-plus", "Save as new note", () =>
    void saveAsNewNote(opts.app, opts.text, opts.saveFolder),
  );
  for (const extra of opts.extra ?? []) {
    actionButton(bar, extra.icon, extra.label, extra.onClick);
  }
}
