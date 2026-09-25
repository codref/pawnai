import { App, MarkdownView, Modal, Notice, Setting, TFile } from "obsidian";

/** Active Markdown file even when a sidebar view (Pawn panel) has focus. */
export function resolveActiveMarkdownFile(app: App): TFile | null {
  const active = app.workspace.getActiveFile();
  if (active instanceof TFile && active.extension === "md") {
    return active;
  }

  const mdView = app.workspace.getActiveViewOfType(MarkdownView);
  if (mdView?.file) {
    return mdView.file;
  }

  // Panel / settings / graph often become the "active" leaf — fall back to
  // the most recently used markdown editor in the workspace.
  const leaves = app.workspace.getLeavesOfType("markdown");
  for (const leaf of leaves) {
    const view = leaf.view;
    if (view instanceof MarkdownView && view.file) {
      return view.file;
    }
  }
  return null;
}

export function resolveActiveMarkdownView(app: App): MarkdownView | null {
  const focused = app.workspace.getActiveViewOfType(MarkdownView);
  if (focused?.file) {
    return focused;
  }
  const file = resolveActiveMarkdownFile(app);
  if (!file) {
    return null;
  }
  for (const leaf of app.workspace.getLeavesOfType("markdown")) {
    const view = leaf.view;
    if (view instanceof MarkdownView && view.file?.path === file.path) {
      return view;
    }
  }
  return null;
}

/** Text prompt that works on desktop and mobile (unlike window.prompt). */
export function promptText(
  app: App,
  opts: { title: string; placeholder?: string; value?: string },
): Promise<string | null> {
  return new Promise((resolve) => {
    const modal = new (class extends Modal {
      private value = opts.value ?? "";
      private settled = false;

      onOpen(): void {
        const { contentEl } = this;
        contentEl.empty();
        contentEl.createEl("h2", { text: opts.title });
        const ta = contentEl.createEl("textarea", {
          attr: {
            rows: "6",
            placeholder: opts.placeholder ?? "",
          },
        });
        ta.value = this.value;
        ta.style.width = "100%";
        ta.style.minHeight = "8em";
        ta.focus();

        const finish = (result: string | null) => {
          if (this.settled) return;
          this.settled = true;
          resolve(result);
          this.close();
        };

        ta.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
            ev.preventDefault();
            finish(ta.value);
          }
        });

        new Setting(contentEl)
          .addButton((btn) =>
            btn.setButtonText("Cancel").onClick(() => finish(null)),
          )
          .addButton((btn) =>
            btn
              .setButtonText("Send")
              .setCta()
              .onClick(() => finish(ta.value)),
          );
      }

      onClose(): void {
        if (!this.settled) {
          this.settled = true;
          resolve(null);
        }
        this.contentEl.empty();
      }
    })(app);
    modal.open();
  });
}

export function noticeIfNoNote(): void {
  new Notice("Open a Markdown note first");
}
