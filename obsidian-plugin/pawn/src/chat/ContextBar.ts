import { App, FuzzySuggestModal, TFile, setIcon } from "obsidian";
import { resolveActiveMarkdownFile, resolveActiveMarkdownView } from "../active";
import type { SelectionRef } from "./conversations";

export class NotePickerModal extends FuzzySuggestModal<TFile> {
  constructor(
    app: App,
    private onPick: (file: TFile) => void,
  ) {
    super(app);
    this.setPlaceholder("Add a note as context…");
  }

  getItems(): TFile[] {
    return this.app.vault.getMarkdownFiles();
  }

  getItemText(file: TFile): string {
    return file.path;
  }

  onChooseItem(file: TFile): void {
    this.onPick(file);
  }
}

export function captureSelection(app: App): SelectionRef | null {
  const view = resolveActiveMarkdownView(app);
  if (!view?.file) return null;
  const editor = view.editor;
  const text = editor.getSelection();
  if (!text.trim()) return null;
  return {
    path: view.file.path,
    from: editor.getCursor("from"),
    to: editor.getCursor("to"),
    text,
  };
}

/** Resolve a dropped string (obsidian:// URL, [[link]], or path) to a vault file. */
export function resolveDroppedNote(app: App, raw: string): TFile | null {
  const text = raw.trim();
  if (!text) return null;
  let target = text;
  if (text.startsWith("obsidian://")) {
    try {
      const url = new URL(text);
      target = url.searchParams.get("file") ?? "";
    } catch {
      return null;
    }
  } else {
    const m = text.match(/^!?\[\[([^\]|#]+)/);
    if (m) target = m[1];
  }
  target = decodeURIComponent(target).trim();
  if (!target) return null;
  const direct = app.vault.getAbstractFileByPath(target);
  if (direct instanceof TFile) return direct;
  return app.metadataCache.getFirstLinkpathDest(target.replace(/\.md$/i, ""), "") ?? null;
}

export interface ContextSnapshot {
  activeNote: TFile | null;
  selection: SelectionRef | null;
  extra: TFile[];
}

/** Context chips above the chat composer: active note, selection, extra notes. */
export class ContextBar {
  private includeActive: boolean;
  private includeSelection = true;
  private extra: TFile[] = [];
  private el: HTMLElement | null = null;

  constructor(
    private app: App,
    defaultIncludeActive: boolean,
  ) {
    this.includeActive = defaultIncludeActive;
  }

  mount(container: HTMLElement): void {
    this.el = container.createDiv({ cls: "pawn-context-bar" });
    this.render();
  }

  addFile(file: TFile): void {
    if (file.extension !== "md") return;
    if (!this.extra.some((f) => f.path === file.path)) this.extra.push(file);
    this.render();
  }

  openPicker(): void {
    new NotePickerModal(this.app, (f) => this.addFile(f)).open();
  }

  /** Current context; resets per-message toggles afterwards when `consume` is set. */
  snapshot(consume = false): ContextSnapshot {
    const active = resolveActiveMarkdownFile(this.app);
    const snap: ContextSnapshot = {
      activeNote: this.includeActive ? active : null,
      selection: this.includeSelection ? captureSelection(this.app) : null,
      extra: [...this.extra],
    };
    if (consume) {
      this.extra = [];
      this.includeSelection = true;
      this.render();
    }
    return snap;
  }

  setIncludeActive(v: boolean): void {
    this.includeActive = v;
    this.render();
  }

  private stateKey(): string {
    const active = resolveActiveMarkdownFile(this.app);
    const sel = captureSelection(this.app);
    return [
      active?.path ?? "",
      this.includeActive,
      this.includeSelection,
      sel ? `${sel.path}:${sel.from.line}:${sel.from.ch}:${sel.to.line}:${sel.to.ch}` : "",
      this.extra.map((f) => f.path).join("|"),
    ].join("\n");
  }

  private lastKey = "";

  /** Re-render only when the visible context changed (cheap to call often). */
  refresh(): void {
    if (this.stateKey() !== this.lastKey) this.render();
  }

  render(): void {
    const el = this.el;
    if (!el || !el.isConnected) return;
    this.lastKey = this.stateKey();
    el.empty();
    const active = resolveActiveMarkdownFile(this.app);
    if (active) {
      this.chip(el, {
        icon: "file-text",
        label: active.basename,
        title: `Active note: ${active.path}`,
        muted: !this.includeActive,
        onToggle: () => {
          this.includeActive = !this.includeActive;
          this.render();
        },
      });
    }
    const sel = captureSelection(this.app);
    if (sel) {
      const words = sel.text.trim().split(/\s+/).length;
      this.chip(el, {
        icon: "text-select",
        label: `Selection (${words} word${words === 1 ? "" : "s"})`,
        title: sel.text.slice(0, 400),
        muted: !this.includeSelection,
        onToggle: () => {
          this.includeSelection = !this.includeSelection;
          this.render();
        },
      });
    }
    for (const file of this.extra) {
      this.chip(el, {
        icon: "link",
        label: file.basename,
        title: file.path,
        onRemove: () => {
          this.extra = this.extra.filter((f) => f.path !== file.path);
          this.render();
        },
      });
    }
    const add = el.createEl("button", {
      cls: "pawn-chip pawn-chip-add clickable-icon",
      attr: { "aria-label": "Add note as context (or type @)" },
    });
    setIcon(add, "plus");
    add.onclick = () => this.openPicker();
  }

  private chip(
    parent: HTMLElement,
    opts: {
      icon: string;
      label: string;
      title: string;
      muted?: boolean;
      onToggle?: () => void;
      onRemove?: () => void;
    },
  ): void {
    const chip = parent.createDiv({ cls: "pawn-chip", attr: { "aria-label": opts.title } });
    chip.toggleClass("is-muted", !!opts.muted);
    const icon = chip.createSpan({ cls: "pawn-chip-icon" });
    setIcon(icon, opts.icon);
    chip.createSpan({ text: opts.label, cls: "pawn-chip-label" });
    if (opts.onToggle) {
      chip.addClass("is-toggle");
      chip.onclick = opts.onToggle;
    }
    if (opts.onRemove) {
      const x = chip.createSpan({ cls: "pawn-chip-remove" });
      setIcon(x, "x");
      x.onclick = (ev) => {
        ev.stopPropagation();
        opts.onRemove?.();
      };
    }
  }
}
