import { Modal, Notice, Setting, TFile } from "obsidian";
import { HttpError } from "../api";
import type PawnPlugin from "../main";

/** Re-run coworker item extract and/or people refresh for a session. */
export class ExtractModal extends Modal {
  private doItems = true;
  private doPeople = false;
  private busy = false;

  constructor(
    private plugin: PawnPlugin,
    private sessionId: string,
    private transcriptPath: string | null,
    private onDone?: () => void,
  ) {
    super(plugin.app);
  }

  onOpen(): void {
    this.titleEl.setText(`Extract · ${this.sessionId}`);
    this.renderBody();
  }

  private renderBody(): void {
    const { contentEl } = this;
    contentEl.empty();

    contentEl.createEl("p", {
      text:
        "Re-runs the same analysis that runs at the end of a recording. " +
        "Edit ## Annotations on the transcript first to add people, relations, " +
        "or what to ignore — those notes are included in the extract. " +
        "Untriaged items for this session are replaced; already-triaged " +
        "fingerprints stay suppressed.",
    });

    if (this.transcriptPath) {
      new Setting(contentEl)
        .setName("Transcript")
        .setDesc(this.transcriptPath)
        .addButton((btn) => {
          btn.setButtonText("Open");
          btn.onClick(() => {
            const file = this.app.vault.getAbstractFileByPath(this.transcriptPath!);
            if (file instanceof TFile) {
              void this.app.workspace.getLeaf(false).openFile(file);
            } else {
              new Notice("Transcript not in vault yet.", 4000);
            }
          });
        });
    }

    new Setting(contentEl)
      .setName("Items")
      .setDesc("Extract decisions, commitments, questions → Pawn/Items")
      .addToggle((t) => {
        t.setValue(this.doItems);
        t.onChange((v) => {
          this.doItems = v;
        });
      });

    new Setting(contentEl)
      .setName("People refresh")
      .setDesc("Update People/ bios for gallery speakers in this session")
      .addToggle((t) => {
        t.setValue(this.doPeople);
        t.onChange((v) => {
          this.doPeople = v;
        });
      });

    const row = contentEl.createDiv({ cls: "modal-button-container" });
    const cancel = row.createEl("button", { text: "Cancel" });
    cancel.disabled = this.busy;
    cancel.onclick = () => this.close();
    const ok = row.createEl("button", {
      text: this.busy ? "Running…" : "Extract",
      cls: "mod-cta",
    });
    ok.disabled = this.busy || (!this.doItems && !this.doPeople);
    ok.onclick = () => void this.submit();
  }

  private async submit(): Promise<void> {
    if (this.busy) return;
    if (!this.doItems && !this.doPeople) {
      new Notice("Enable Items and/or People refresh.", 4000);
      return;
    }
    this.busy = true;
    this.renderBody();
    try {
      const result = await this.plugin.client.extractSession(this.sessionId, {
        items: this.doItems,
        people: this.doPeople,
      });
      const parts: string[] = [];
      if (result.items) {
        parts.push(`${result.items.items ?? 0} item(s)`);
      }
      if (result.people) {
        parts.push("people refreshed");
      }
      new Notice(
        parts.length ? `Extract done: ${parts.join(", ")}.` : "Extract done.",
        6000,
      );
      void this.plugin.items?.refresh();
      this.onDone?.();
      this.close();
    } catch (e) {
      const msg =
        e instanceof HttpError
          ? e.message
          : e instanceof Error
            ? e.message
            : String(e);
      new Notice(msg, 7000);
      this.busy = false;
      this.renderBody();
    }
  }
}
