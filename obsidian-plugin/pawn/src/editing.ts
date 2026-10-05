import { App, MarkdownView, Modal, Notice, Setting, TFile, normalizePath } from "obsidian";
import { resolveActiveMarkdownFile, resolveActiveMarkdownView } from "./active";
import type { SelectionRef } from "./chat/conversations";
import { diffLines } from "./chat/diff";

export class DiffModal extends Modal {
  private settled = false;

  constructor(
    app: App,
    private before: string,
    private after: string,
    private onDecide: (apply: boolean) => void,
  ) {
    super(app);
  }

  onOpen(): void {
    const { contentEl } = this;
    this.titleEl.setText("Replace selection?");
    contentEl.addClass("pawn-diff-modal");
    const box = contentEl.createDiv({ cls: "pawn-diff" });
    for (const op of diffLines(this.before, this.after)) {
      const line = box.createDiv({ cls: `pawn-diff-line is-${op.kind}` });
      line.createSpan({
        cls: "pawn-diff-sign",
        text: op.kind === "add" ? "+" : op.kind === "del" ? "−" : " ",
      });
      line.createSpan({ text: op.text || " " });
    }
    new Setting(contentEl)
      .addButton((b) => b.setButtonText("Cancel").onClick(() => this.finish(false)))
      .addButton((b) =>
        b
          .setButtonText("Apply")
          .setCta()
          .onClick(() => this.finish(true)),
      );
  }

  private finish(apply: boolean): void {
    if (this.settled) return;
    this.settled = true;
    this.onDecide(apply);
    this.close();
  }

  onClose(): void {
    if (!this.settled) {
      this.settled = true;
      this.onDecide(false);
    }
    this.contentEl.empty();
  }
}

function viewForPath(app: App, path: string): MarkdownView | null {
  for (const leaf of app.workspace.getLeavesOfType("markdown")) {
    const view = leaf.view;
    if (view instanceof MarkdownView && view.file?.path === path) return view;
  }
  return null;
}

export function insertAtCursor(app: App, text: string): void {
  const view = resolveActiveMarkdownView(app);
  if (!view) {
    new Notice("Open a Markdown note first.");
    return;
  }
  const editor = view.editor;
  editor.replaceRange(text, editor.getCursor("to"));
}

/** Replace the remembered selection (if still intact) or the current one, after a diff preview. */
export function replaceSelection(app: App, text: string, ref?: SelectionRef): void {
  let view: MarkdownView | null = null;
  let from = null;
  let to = null;
  if (ref) {
    view = viewForPath(app, ref.path);
    if (view && view.editor.getRange(ref.from, ref.to) === ref.text) {
      from = ref.from;
      to = ref.to;
    }
  }
  if (!from || !to) {
    view = resolveActiveMarkdownView(app);
    if (!view || !view.editor.getSelection()) {
      new Notice("Select the text to replace first.");
      return;
    }
    from = view.editor.getCursor("from");
    to = view.editor.getCursor("to");
  }
  const editor = view!.editor;
  const f = from;
  const t = to;
  const before = editor.getRange(f, t);
  new DiffModal(app, before, text, (apply) => {
    if (apply) editor.replaceRange(text, f, t);
  }).open();
}

export async function appendToNote(app: App, text: string, path?: string): Promise<void> {
  const file = path
    ? app.vault.getAbstractFileByPath(path)
    : resolveActiveMarkdownFile(app);
  if (!(file instanceof TFile)) {
    new Notice("No note to append to.");
    return;
  }
  await app.vault.process(file, (body) => `${body.replace(/\s+$/, "")}\n\n${text.trim()}\n`);
  new Notice(`Appended to ${file.basename}`);
}

export async function saveAsNewNote(app: App, text: string, folder: string): Promise<void> {
  const stamp = window.moment().format("YYYY-MM-DD HHmm");
  const heading = text.match(/^#\s+(.+)$/m)?.[1]?.trim();
  const base = (heading || `Pawn ${stamp}`).replace(/[\\/:*?"<>|#^[\]]/g, "").slice(0, 80);
  const dir = normalizePath(folder);
  if (!app.vault.getAbstractFileByPath(dir)) {
    await app.vault.createFolder(dir).catch(() => undefined);
  }
  let path = normalizePath(`${dir}/${base}.md`);
  for (let i = 2; app.vault.getAbstractFileByPath(path); i++) {
    path = normalizePath(`${dir}/${base} ${i}.md`);
  }
  const file = await app.vault.create(path, text.trim() + "\n");
  await app.workspace.getLeaf(true).openFile(file);
}
