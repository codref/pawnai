import { MarkdownView, TFile } from "obsidian";
import type { InboxItem } from "../api";
import type PawnPlugin from "../main";
import { mountItemActions } from "./InboxView";
import { OPEN_STATUSES, isItemNotePath } from "./items";

const BAR_CLS = "pawn-item-note-bar";

/** Mount take-action / Delete buttons above an open ``pawn: item`` note. */
export class ItemNoteBar {
  private bar: HTMLElement | null = null;
  private path = "";
  private timer = 0;

  constructor(private plugin: PawnPlugin) {}

  start(): void {
    const schedule = () => {
      if (this.timer) window.clearTimeout(this.timer);
      this.timer = window.setTimeout(() => {
        this.timer = 0;
        void this.refresh();
      }, 50);
    };
    this.plugin.registerEvent(this.plugin.app.workspace.on("active-leaf-change", schedule));
    this.plugin.registerEvent(this.plugin.app.workspace.on("file-open", schedule));
    this.plugin.registerEvent(
      this.plugin.app.metadataCache.on("changed", (file) => {
        if (file.path === this.path) schedule();
      }),
    );
    void this.refresh();
  }

  private clear(): void {
    this.bar?.remove();
    this.bar = null;
    this.path = "";
  }

  private itemsDir(): string {
    return `${this.plugin.settings.agentRoot.replace(/\/+$/, "")}/Items`;
  }

  async refresh(): Promise<void> {
    const view = this.plugin.app.workspace.getActiveViewOfType(MarkdownView);
    const file = view?.file;
    if (!(file instanceof TFile) || !view) {
      this.clear();
      return;
    }
    // Cheap reject: not under Items/, or cache says not an open item.
    if (!isItemNotePath(file.path, this.itemsDir())) {
      this.clear();
      return;
    }
    const cache = this.plugin.app.metadataCache.getFileCache(file);
    const fm = cache?.frontmatter;
    let item: InboxItem | null = null;
    if (fm && String(fm.pawn || "") === "item") {
      const status = String(fm.status || "new").trim().toLowerCase();
      if (!(OPEN_STATUSES as readonly string[]).includes(status)) {
        this.clear();
        return;
      }
      item = {
        id: String(fm.id || file.path),
        short_id: String(fm.short_id || ""),
        kind: String(fm.kind || "item"),
        text: String(fm.thread || fm.kind || "item"),
        thread: fm.thread != null && String(fm.thread).trim() ? String(fm.thread) : null,
        status,
        interrupt: false,
        note_key: file.path,
      };
      // Prefer body first line when cheap.
      const text = await this.plugin.app.vault.cachedRead(file);
      item = parseOpenItemNote(file.path, text) ?? item;
    } else {
      const text = await this.plugin.app.vault.cachedRead(file);
      item = parseOpenItemNote(file.path, text);
    }
    if (!item) {
      this.clear();
      return;
    }
    if (this.bar && this.path === file.path) {
      this.bar.empty();
      mountItemActions(this.bar, this.plugin, item, () => void this.refresh());
      return;
    }
    this.clear();
    const container = view.containerEl.querySelector(".view-content") as HTMLElement | null;
    if (!container) return;
    this.bar = document.createElement("div");
    this.bar.className = BAR_CLS;
    container.insertBefore(this.bar, container.firstChild);
    this.path = file.path;
    mountItemActions(this.bar, this.plugin, item, () => void this.refresh());
  }
}

function parseOpenItemNote(path: string, text: string): InboxItem | null {
  if (!text.startsWith("---\n")) return null;
  const end = text.indexOf("\n---", 3);
  if (end < 0) return null;
  const meta = text.slice(4, end);
  const pawn = meta.match(/^pawn:\s*(.*)$/m);
  if (!pawn || pawn[1].trim() !== "item") return null;
  const val = (key: string) => {
    const m = meta.match(new RegExp(`^${key}:\\s*(.*)$`, "m"));
    return m ? m[1].trim() : "";
  };
  const status = (val("status") || "new").toLowerCase();
  if (!(OPEN_STATUSES as readonly string[]).includes(status)) return null;
  const rest = text.slice(end + 4).trim();
  const bodyLine =
    rest.split("\n").find((l) => l.trim() && !l.startsWith(">") && !l.startsWith("#")) || "";
  return {
    id: val("id") || path,
    short_id: val("short_id") || "",
    kind: val("kind") || "item",
    text: bodyLine || "(empty)",
    thread: val("thread") || null,
    status,
    interrupt: false,
    note_key: path,
  };
}
