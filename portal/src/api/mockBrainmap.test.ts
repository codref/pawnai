import { describe, expect, it } from "vitest";
import { MockApi } from "./mock";

describe("mock brainmap", () => {
  it("builds a web of items, hubs, people, tapes and ghosts", async () => {
    const api = new MockApi();
    const g = await api.brainmap("items");
    const kinds = new Set(g.nodes.map((n) => n.kind));
    for (const k of ["item", "thread", "question", "decision", "commitment", "prediction", "hypothesis", "person", "session"]) {
      expect(kinds.has(k as never)).toBe(true);
    }
    const ids = new Set(g.nodes.map((n) => n.id));
    for (const e of g.edges) {
      expect(ids.has(e.from)).toBe(true);
      expect(ids.has(e.to)).toBe(true);
    }
    expect(g.suggestions.length).toBeGreaterThanOrEqual(5);
  });

  it("ties, cuts, snips, lassoes and spills", async () => {
    const api = new MockApi();
    const g = await api.brainmap("items");
    const tie = await api.brainmapTie("items", "item:it4", "subject:t1", "mentions");
    expect(tie.edge).toMatchObject({ from: "item:it4", to: "subject:t1", by: "user" });
    await api.brainmapCut("items", tie.edge.id);
    expect((await api.brainmap("items")).edges.some((e) => e.id === tie.edge.id)).toBe(false);
    const sug = g.suggestions[0];
    await api.brainmapSuggestion("items", sug.id, "snip");
    expect((await api.brainmap("items")).suggestions.some((s) => s.id === sug.id)).toBe(false);
    const lasso = await api.brainmapLasso("items", { title: "Staging capacity", node_ids: ["item:it8", "item:it15"] });
    expect(lasso.subject.kind).toBe("thread");
    expect(lasso.edges.length).toBe(2);
    const spill = await api.brainmapSpill("items", "t2");
    expect(spill.nodes.every((n) => n.kind === "scrap")).toBe(true);
    expect(spill.nodes.length).toBe(3);
    const detail = await api.emergenceSubject("t2");
    expect(detail.observations.length).toBeGreaterThan(3);
  });
});
