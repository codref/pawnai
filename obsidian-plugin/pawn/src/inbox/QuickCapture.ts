import { App, Modal, Notice } from "obsidian";

/** One-line idea capture into Ideas/. */
export class QuickCaptureModal extends Modal {
  private text = "";

  constructor(
    app: App,
    private onSave: (title: string) => Promise<void>,
  ) {
    super(app);
  }

  onOpen(): void {
    const { contentEl } = this;
    contentEl.createEl("h3", { text: "Quick capture" });
    const input = contentEl.createEl("textarea");
    input.rows = 4;
    input.placeholder = "An idea, in one or two lines";
    input.oninput = () => {
      this.text = input.value;
    };
    const row = contentEl.createDiv({ cls: "pawn-job-actions" });
    const save = row.createEl("button", { text: "Save to Ideas", cls: "mod-cta" });
    save.onclick = () => {
      const title = this.text.trim();
      if (!title) {
        new Notice("Write something first.");
        return;
      }
      void this.onSave(title).then(() => this.close());
    };
    input.focus();
  }
}
