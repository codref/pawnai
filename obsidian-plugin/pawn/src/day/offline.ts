/** Lightweight frontmatter + ## Prep extract for offline Day scan. */

const FM_RE = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;
const PREP_RE = /^##\s*Prep\s*\n+([\s\S]*?)(?=^##\s|\Z)/im;

export function parseFrontmatterLocal(text: string): { body: string; prep: string } {
  const raw = text || "";
  const m = FM_RE.exec(raw);
  const rest = m ? raw.slice(m[0].length) : raw;
  const prepMatch = PREP_RE.exec(rest);
  const prep = prepMatch ? prepMatch[1].trim() : "";
  const body = prepMatch ? rest.replace(PREP_RE, "").trim() : rest.trim();
  return { body, prep };
}
