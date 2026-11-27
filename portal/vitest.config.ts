import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vitest/config";

// Unit tests for the portal's pure logic (anchoring). `make portal-test`.
export default defineConfig({
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  test: { include: ["src/**/*.test.ts"], environment: "jsdom" },
});
