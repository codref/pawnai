/** Pull the fenced goals proposal out of a weekly review note. */
export function extractGoalsBlock(text: string): string {
  const marker = "```goals";
  const start = text.indexOf(marker);
  if (start < 0) return "";
  const body = text.slice(start + marker.length);
  const end = body.indexOf("```");
  if (end < 0) return "";
  return body.slice(0, end).trim() + "\n";
}
