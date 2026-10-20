import { App, FuzzySuggestModal, normalizePath } from "obsidian";
import { promptText } from "../active";

export type TodoProjectChoice =
  | { type: "cancel" }
  | { type: "none" }
  | { type: "project"; name: string };

interface ProjectOption {
  id: string;
  label: string;
  kind: "none" | "project" | "create";
  name?: string;
}

/** List TaskNotes project titles under ``{agentRoot}/TaskNotes/Projects/``. */
export function listTasknotesProjects(app: App, agentRoot: string): string[] {
  const root = agentRoot.replace(/\\/g, "/").replace(/\/+$/, "") || "Pawn";
  const folder = normalizePath(`${root}/TaskNotes/Projects`);
  const prefix = `${folder}/`;
  const names: string[] = [];
  const seen = new Set<string>();
  for (const file of app.vault.getMarkdownFiles()) {
    if (!file.path.startsWith(prefix)) continue;
    const cache = app.metadataCache.getFileCache(file);
    const title = cache?.frontmatter?.title;
    const name =
      title != null && String(title).trim() ? String(title).trim() : file.basename;
    const key = name.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    names.push(name);
  }
  names.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
  return names;
}

/**
 * Pick a project for Add TODO. Returns ``cancel`` if dismissed,
 * ``none`` for the flat-task escape hatch, or a project name.
 */
export function pickTodoProject(app: App, agentRoot: string): Promise<TodoProjectChoice> {
  return new Promise((resolve) => {
    const projects = listTasknotesProjects(app, agentRoot);
    const options: ProjectOption[] = [
      { id: "none", label: "No project", kind: "none" },
      ...projects.map((name) => ({
        id: `p:${name}`,
        label: name,
        kind: "project" as const,
        name,
      })),
      { id: "create", label: "Create new project…", kind: "create" },
    ];
    const modal = new (class extends FuzzySuggestModal<ProjectOption> {
      private settled = false;

      getItems(): ProjectOption[] {
        return options;
      }

      getItemText(item: ProjectOption): string {
        return item.label;
      }

      onChooseItem(item: ProjectOption): void {
        if (this.settled) return;
        this.settled = true;
        if (item.kind === "none") {
          resolve({ type: "none" });
          return;
        }
        if (item.kind === "project" && item.name) {
          resolve({ type: "project", name: item.name });
          return;
        }
        void promptText(app, {
          title: "New project",
          placeholder: "Project name",
        }).then((raw) => {
          const name = (raw || "").trim();
          if (!name) {
            resolve({ type: "cancel" });
            return;
          }
          resolve({ type: "project", name });
        });
      }

      onClose(): void {
        window.setTimeout(() => {
          if (!this.settled) {
            this.settled = true;
            resolve({ type: "cancel" });
          }
        }, 0);
      }
    })(app);
    modal.setPlaceholder("File TODO under project…");
    modal.open();
  });
}
