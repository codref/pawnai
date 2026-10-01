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
  /** Setting default. Per-file pin/unpin below overrides it. */
  private defaultIncludeActive: boolean;
  /** Active-note paths the user detached with the chip close button. */
  private detached = new Set<string>();
  /** Active-note paths the user pinned back (needed when the default is off). */
  private explicitPins = new Set<string>();
  private includeSelection = true;
  private extra: TFile[] = [];
  private el: HTMLElement | null = null;
  private imageChips: { label: string; title: string; onRemove: () => void }[] = [];

  constructor(
    private app: App,
    defaultIncludeActive: boolean,
  ) {
    this.defaultIncludeActive = defaultIncludeActive;
  }

  mount(container: HTMLElement): void {
    this.el = container.createDiv({ cls: "pawn-context-bar" });
    this.render();
  }

  /** Dropped pictures, and a one-shot re-read chip after `/vision refresh`. */
  setImageChips(chips: { label: string; title: string; onRemove: () => void }[]): void {
    this.imageChips = chips;
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
      activeNote: active && this.isAttached(active) ? active : null,
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

  /** Pin or unpin the open note. Falls back to the default when no note is open. */
  setIncludeActive(v: boolean): void {
    const file = resolveActiveMarkdownFile(this.app);
    if (!file) {
      this.defaultIncludeActive = v;
      this.render();
      return;
    }
    if (v) this.pin(file.path);
    else this.detach(file.path);
  }

  private isAttached(file: TFile): boolean {
    if (this.detached.has(file.path)) return false;
    if (this.explicitPins.has(file.path)) return true;
    return this.defaultIncludeActive;
  }

  private detach(path: string): void {
    this.detached.add(path);
    this.explicitPins.delete(path);
    this.render();
  }

  private pin(path: string): void {
    this.detached.delete(path);
    this.explicitPins.add(path);
    this.render();
  }

  private stateKey(): string {
    const active = resolveActiveMarkdownFile(this.app);
    const sel = captureSelection(this.app);
    return [
      active?.path ?? "",
      active ? this.isAttached(active) : false,
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
    if (active && this.isAttached(active)) {
      this.chip(el, {
        icon: "file-text",
        label: active.basename,
        title: `${active.path} is attached to the next message`,
        removeLabel: `Unpin ${active.basename}`,
        onRemove: () => this.detach(active.path),
      });
    } else if (active) {
      this.pinChip(el, active);
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
    for (const image of this.imageChips) {
      this.chip(el, {
        icon: "image",
        label: image.label,
        title: image.title,
        removeLabel: `Remove ${image.label}`,
        onRemove: image.onRemove,
      });
    }
    for (const file of this.extra) {
      this.chip(el, {
        icon: "link",
        label: file.basename,
        title: file.path,
        removeLabel: `Remove ${file.basename}`,
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

  /** Dashed chip that reattaches the open note after it was unpinned. */
  private pinChip(parent: HTMLElement, file: TFile): void {
    const pin = parent.createEl("button", {
      cls: "pawn-chip pawn-chip-pin",
      attr: { type: "button", "aria-label": `Pin ${file.basename}` },
    });
    const icon = pin.createSpan({ cls: "pawn-chip-icon" });
    setIcon(icon, "pin");
    pin.createSpan({ text: file.basename, cls: "pawn-chip-label" });
    pin.onclick = () => this.pin(file.path);
  }

  private chip(
    parent: HTMLElement,
    opts: {
      icon: string;
      label: string;
      title: string;
      muted?: boolean;
      removeLabel?: string;
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
      const x = chip.createEl("button", {
        cls: "pawn-chip-remove clickable-icon",
        attr: { type: "button", "aria-label": opts.removeLabel ?? "Remove" },
      });
      setIcon(x, "x");
      x.onclick = (ev) => {
        ev.stopPropagation();
        opts.onRemove?.();
      };
    }
  }
}
