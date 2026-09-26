import { FuzzySuggestModal, Notice, TFile, normalizePath } from "obsidian";
import type PawnPlugin from "../main";

export interface PromptCommand {
  slug: string;
  name: string;
  description: string;
  prompt: string;
  contextMenu: boolean;
  background: boolean;
  source?: string;
}

const BUILTINS: PromptCommand[] = [
  {
    slug: "summarize",
    name: "Summarize",
    description: "Concise summary with key points",
    prompt:
      "Summarize {selection} in a few sentences, then list the key points as bullets. " +
      "Reply with Markdown only.",
    contextMenu: true,
    background: false,
  },
  {
    slug: "rewrite",
    name: "Rewrite for clarity",
    description: "Clearer wording, same meaning",
    prompt:
      "Rewrite {selection} to be clearer and more concise while keeping the meaning and tone. " +
      "Reply with the rewritten text only.",
    contextMenu: true,
    background: false,
  },
  {
    slug: "fix-grammar",
    name: "Fix grammar and spelling",
    description: "Minimal corrections",
    prompt:
      "Fix grammar, spelling and punctuation in {selection}. Change nothing else. " +
      "Reply with the corrected text only.",
    contextMenu: true,
    background: false,
  },
  {
    slug: "translate",
    name: "Translate to English",
    description: "Translate, keep Markdown formatting",
    prompt:
      "Translate {selection} to English, preserving Markdown formatting. " +
      "Reply with the translation only.",
    contextMenu: true,
    background: false,
  },
  {
    slug: "action-items",
    name: "Extract action items",
    description: "Checklist of tasks with owners",
    prompt:
      "Extract the action items from {selection} as a Markdown checklist (- [ ] …), " +
      "with owner and due date when mentioned.",
    contextMenu: true,
    background: false,
  },
];

export function slugify(s: string): string {
  return (
    s
      .toLowerCase()
      .replace(/\.md$/, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "command"
  );
}

/** Fill `{selection}`, `{note}` and `{date}` placeholders. */
export function renderPrompt(
  template: string,
  vars: { selection?: string | null; noteTitle?: string | null },
): string {
  const sel = vars.selection?.trim();
  const note = vars.noteTitle ? `[[${vars.noteTitle}]]` : "the active note";
  return template
    .replace(/\{selection\}/g, sel ? `the selected text` : note)
    .replace(/\{note\}/g, note)
    .replace(/\{date\}/g, window.moment().format("YYYY-MM-DD"));
}

function stripFrontmatter(text: string): string {
  return text.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, "").trim();
}

function asBool(v: unknown, dflt: boolean): boolean {
  if (typeof v === "boolean") return v;
  if (typeof v === "string") return ["true", "yes", "1", "on"].includes(v.toLowerCase());
  return dflt;
}

export class PromptCommandRegistry {
  private fromVault: PromptCommand[] = [];
  private registered = new Set<string>();

  constructor(private plugin: PawnPlugin) {}

  private get folder(): string {
    return normalizePath(this.plugin.settings.commandsFolder || "Pawn/Commands");
  }

  isCommandFile(path: string): boolean {
    return path.startsWith(`${this.folder}/`) && path.endsWith(".md");
  }

  list(): PromptCommand[] {
    const bySlug = new Map<string, PromptCommand>();
    for (const c of BUILTINS) bySlug.set(c.slug, c);
    for (const c of this.fromVault) bySlug.set(c.slug, c);
    return [...bySlug.values()].sort((a, b) => a.name.localeCompare(b.name));
  }

  find(slug: string): PromptCommand | undefined {
    return this.list().find((c) => c.slug === slug);
  }

  async reload(): Promise<void> {
    const out: PromptCommand[] = [];
    for (const file of this.plugin.app.vault.getMarkdownFiles()) {
      if (!this.isCommandFile(file.path)) continue;
      const cmd = await this.parse(file);
      if (cmd) out.push(cmd);
    }
    this.fromVault = out;
    this.registerPalette();
  }

  private async parse(file: TFile): Promise<PromptCommand | null> {
    const text = await this.plugin.app.vault.cachedRead(file);
    const fm = this.plugin.app.metadataCache.getFileCache(file)?.frontmatter ?? {};
    const prompt = stripFrontmatter(text);
    if (!prompt) return null;
    const name = String(fm.name ?? file.basename).trim();
    return {
      slug: slugify(String(fm.slash ?? file.basename)),
      name,
      description: String(fm.description ?? ""),
      prompt,
      contextMenu: asBool(fm.context_menu, true),
      background: asBool(fm.background, false),
      source: file.path,
    };
  }

  /** Palette entries; commands added later are reachable via the picker, menu and "/". */
  private registerPalette(): void {
    for (const cmd of this.list()) {
      const id = `prompt-${cmd.slug}`;
      if (this.registered.has(id)) continue;
      this.registered.add(id);
      this.plugin.addCommand({
        id,
        name: `Prompt: ${cmd.name}`,
        callback: () => {
          const current = this.find(cmd.slug);
          if (current) void this.plugin.runPromptCommand(current);
          else new Notice(`Prompt command “${cmd.name}” no longer exists.`);
        },
      });
    }
  }

  async writeDefaults(): Promise<void> {
    const vault = this.plugin.app.vault;
    if (!vault.getAbstractFileByPath(this.folder)) {
      await vault.createFolder(this.folder).catch(() => undefined);
    }
    let created = 0;
    for (const c of BUILTINS) {
      const path = normalizePath(`${this.folder}/${c.name}.md`);
      if (vault.getAbstractFileByPath(path)) continue;
      const body =
        `---\nname: ${c.name}\ndescription: ${c.description}\nslash: ${c.slug}\n` +
        `context_menu: ${c.contextMenu}\nbackground: ${c.background}\n---\n\n${c.prompt}\n`;
      await vault.create(path, body);
      created++;
    }
    await this.reload();
    new Notice(
      created ? `Created ${created} prompt command(s) in ${this.folder}` : "Defaults already exist",
    );
  }
}

export class PromptPickerModal extends FuzzySuggestModal<PromptCommand> {
  constructor(
    private plugin: PawnPlugin,
    private onPick: (cmd: PromptCommand) => void,
  ) {
    super(plugin.app);
    this.setPlaceholder("Run a Pawn prompt command…");
  }

  getItems(): PromptCommand[] {
    return this.plugin.prompts.list();
  }

  getItemText(cmd: PromptCommand): string {
    return cmd.description ? `${cmd.name} — ${cmd.description}` : cmd.name;
  }

  onChooseItem(cmd: PromptCommand): void {
    this.onPick(cmd);
  }
}
