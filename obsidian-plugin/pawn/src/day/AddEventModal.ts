import { Modal, Notice, Setting } from "obsidian";
import type PawnPlugin from "../main";

export type AddActivityDefaults = {
  date: string;
  start?: string;
  end?: string;
};

/** Manual planner activity create (personal life / no Outlook). */
export class AddEventModal extends Modal {
  private title = "";
  private start = "";
  private end = "";
  private body = "";
  private saving = false;
  private defaultDate: string;

  constructor(
    private plugin: PawnPlugin,
    defaults: string | AddActivityDefaults,
    private onDone: () => void,
  ) {
    super(plugin.app);
    if (typeof defaults === "string") {
      this.defaultDate = defaults;
      this.start = defaults ? `${defaults}T09:00` : "";
      this.end = defaults ? `${defaults}T09:30` : "";
    } else {
      this.defaultDate = defaults.date;
      this.start = defaults.start || (defaults.date ? `${defaults.date}T09:00` : "");
      this.end =
        defaults.end ||
        (this.start
          ? window
              .moment(this.start, ["YYYY-MM-DDTHH:mm", "YYYY-MM-DD HH:mm"], true)
              .add(30, "minutes")
              .format("YYYY-MM-DDTHH:mm")
          : "");
    }
  }

  onOpen(): void {
    this.titleEl.setText("Add activity");
    const { contentEl } = this;
    contentEl.empty();

    new Setting(contentEl).setName("Title").addText((t) => {
      t.setPlaceholder("Coffee / deep work / call").onChange((v) => {
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
      t.setValue(this.end).setPlaceholder("2026-10-28T10:30").onChange((v) => {
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
      new Notice("Activity added.");
      this.close();
      this.onDone();
    } catch (err) {
      this.saving = false;
      if (btn) btn.disabled = false;
      new Notice(`Could not save activity: ${err instanceof Error ? err.message : err}`);
    }
  }
}
