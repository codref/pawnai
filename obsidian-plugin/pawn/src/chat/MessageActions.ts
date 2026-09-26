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

/** Toolbar under an assistant reply or job result. */
export function renderMessageActions(parent: HTMLElement, opts: MessageActionOptions): void {
  const bar = parent.createDiv({ cls: "pawn-msg-actions" });
  const btn = (icon: string, label: string, onClick: () => void) => {
    const b = bar.createEl("button", { cls: "clickable-icon", attr: { "aria-label": label } });
    setIcon(b, icon);
    b.onclick = (ev) => {
      ev.stopPropagation();
      onClick();
    };
  };
  btn("copy", "Copy", () => {
    void navigator.clipboard.writeText(opts.text).then(() => new Notice("Copied"));
  });
  btn("text-cursor-input", "Insert at cursor", () => insertAtCursor(opts.app, opts.text));
  btn("replace", "Replace selection (preview)", () =>
    replaceSelection(opts.app, opts.text, opts.selection),
  );
  btn("list-end", "Append to note", () => void appendToNote(opts.app, opts.text, opts.notePath));
  btn("file-plus", "Save as new note", () =>
    void saveAsNewNote(opts.app, opts.text, opts.saveFolder),
  );
  for (const extra of opts.extra ?? []) btn(extra.icon, extra.label, extra.onClick);
}
