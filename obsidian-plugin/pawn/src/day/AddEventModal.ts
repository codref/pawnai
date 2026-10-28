import { Modal, Notice, Setting } from "obsidian";
import type PawnPlugin from "../main";

/** Manual planner event create (personal life / no Outlook). */
export class AddEventModal extends Modal {
  private title = "";
  private start = "";
  private end = "";
  private body = "";
  private saving = false;

  constructor(
    private plugin: PawnPlugin,
    private defaultDate: string,
    private onDone: () => void,
  ) {
    super(plugin.app);
    this.start = defaultDate ? `${defaultDate}T09:00` : "";
  }

  onOpen(): void {
    this.titleEl.setText("Add planner event");
    const { contentEl } = this;
    contentEl.empty();

    new Setting(contentEl).setName("Title").addText((t) => {
      t.setPlaceholder("Product weekly").onChange((v) => {
        this.title = v;
      });
      t.inputEl.focus();
    });
    new Setting(contentEl)
      .setName("Start")
      .setDesc("Local wall time, e.g. 2026-10-28T10:00")
      .addText((t) => {
        t.setValue(this.start).onChange((v) => {
          this.start = v;
        });
      });
    new Setting(contentEl).setName("End (optional)").addText((t) => {
      t.setPlaceholder("2026-10-28T10:30").onChange((v) => {
        this.end = v;
      });
    });
    new Setting(contentEl).setName("Notes").addTextArea((t) => {
      t.setPlaceholder("Optional body").onChange((v) => {
        this.body = v;
      });
    });

    new Setting(contentEl).addButton((b) => {
      b.setButtonText("Save")
        .setCta()
        .onClick(() => void this.submit(b.buttonEl));
    });
  }

  private async submit(btn?: HTMLButtonElement): Promise<void> {
    if (this.saving) return;
    const title = this.title.trim();
    if (!title) {
      new Notice("Title is required.");
      return;
    }
    this.saving = true;
    if (btn) btn.disabled = true;
    try {
      await this.plugin.client.createPlannerEvent({
        title,
        start: this.start.trim() || this.defaultDate,
        end: this.end.trim() || undefined,
        source: "manual",
        body: this.body.trim() || undefined,
      });
      new Notice("Event added.");
      this.close();
      this.onDone();
    } catch (err) {
      this.saving = false;
      if (btn) btn.disabled = false;
      new Notice(`Could not save event: ${err instanceof Error ? err.message : err}`);
    }
  }
}
