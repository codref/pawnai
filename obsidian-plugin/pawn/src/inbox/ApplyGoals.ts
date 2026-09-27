import { App, Modal, Notice, TFile } from "obsidian";
import { diffLines } from "../chat/diff";
import { extractGoalsBlock } from "./goalsBlock";

/** Confirm a ```goals block from the active review note, then write Goals.md. */
export class ApplyGoalsModal extends Modal {
  constructor(
    app: App,
    private before: string,
    private after: string,
    private onConfirm: () => Promise<void>,
  ) {
    super(app);
  }

  onOpen(): void {
    const { contentEl } = this;
    contentEl.createEl("h3", { text: "Apply goals proposal" });
    const pre = contentEl.createEl("pre", { cls: "pawn-diff" });
    for (const op of diffLines(this.before, this.after)) {
      const line = pre.createDiv({ cls: `pawn-diff-line is-${op.kind}`, text: op.text || " " });
      line.setAttr("data-kind", op.kind);
    }
    const row = contentEl.createDiv({ cls: "pawn-job-actions" });
    const cancel = row.createEl("button", { text: "Cancel" });
    cancel.onclick = () => this.close();
    const ok = row.createEl("button", { text: "Write Goals.md", cls: "mod-cta" });
    ok.onclick = () => {
      void this.onConfirm().then(() => this.close());
    };
  }
}

export async function applyGoalsFromActiveFile(app: App, goalsPath = "Goals.md"): Promise<void> {
  const file = app.workspace.getActiveFile();
  if (!(file instanceof TFile)) {
    new Notice("Open a review note first.");
    return;
  }
  const review = await app.vault.read(file);
  const proposed = extractGoalsBlock(review);
  if (!proposed.trim()) {
    new Notice("This note has no ```goals block.");
    return;
  }
  const existing = app.vault.getAbstractFileByPath(goalsPath);
  const before = existing instanceof TFile ? await app.vault.read(existing) : "";
  new ApplyGoalsModal(app, before, proposed, async () => {
    if (existing instanceof TFile) await app.vault.modify(existing, proposed);
    else await app.vault.create(goalsPath, proposed);
    new Notice(`Wrote ${goalsPath}`);
  }).open();
}
