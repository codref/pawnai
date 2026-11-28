import { describe, expect, it } from "vitest";
import { MockApi } from "./mock";

describe("mock boards", () => {
  it("creates a board from pins, talks to it, and accepts its ghost", async () => {
    const api = new MockApi();
    const items = await api.brainmap("items");
    const thread = items.nodes.find((n) => n.kind === "thread")!;
    const b = await api.createBoard({ title: "Launch", nodes: [thread.id] });
    expect((await api.listBoards()).map((x) => x.name)).toContain(b.name);
    let g = await api.brainmap(b.name);
    expect(g.meta).toMatchObject({ title: "Launch", kind: "manual" });
    expect(g.nodes.some((n) => n.id === thread.id)).toBe(true);

    const chat = await api.boardChat(b.name);
    expect((await api.boardChat(b.name)).id).toBe(chat.id);
    let proposals: string[] = [];
    await api.sendMessage(chat.id, { message: "what connects these?", board_selection: [thread.id] }, {
      onBoard: (d) => (proposals = d.proposals.map((p) => p.id)),
    });
    expect(proposals).toHaveLength(1);
    g = await api.brainmap(b.name);
    const ghost = g.nodes.find((n) => n.proposed)!;
    expect(ghost.meta.proposal_id).toBe(proposals[0]);
    await api.decideBoardProposal(b.name, proposals[0], "accept");
    g = await api.brainmap(b.name);
    expect(g.nodes.some((n) => n.id === `note:${proposals[0]}`)).toBe(true);

    await api.removeFromBoard(b.name, [thread.id]);
    expect((await api.brainmap(b.name)).nodes.some((n) => n.id === thread.id)).toBe(false);
    await api.deleteBoard(b.name);
    expect((await api.listBoards()).some((x) => x.name === b.name)).toBe(false);
  }, 20000);
});
