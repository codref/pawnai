import { Editor } from "obsidian";

export function shortTitle(instruction: string, max = 48): string {
  const line = (instruction || "").split(/\r?\n/).find((l) => l.trim())?.trim() ?? "Pawn job";
  if (line.length <= max) return line;
  return line.slice(0, max - 1) + "…";
}

/** Optional: insert `> [!pawn] [[agentRoot/Tasks/id|title]]` at the cursor. */
export function insertCallout(editor: Editor, taskPath: string, title: string): void {
  const link = taskPath.replace(/\.md$/i, "");
  const line = `> [!pawn] [[${link}|${title.replace(/\|/g, "\\|")}]]\n`;
  editor.replaceRange(line, editor.getCursor("to"));
}
