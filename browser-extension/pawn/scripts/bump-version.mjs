#!/usr/bin/env node
/**
 * Bump the patch version in manifest.json + manifest.firefox.json.
 * Skip when CI=true (same idea as the Obsidian plugin).
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const files = ["manifest.json", "manifest.firefox.json"];

if (process.env.CI === "true") {
  console.log("CI=true — skip extension version bump");
  process.exit(0);
}

let shown = "";
for (const name of files) {
  const file = path.join(root, name);
  const raw = fs.readFileSync(file, "utf8");
  const data = JSON.parse(raw);
  const prev = String(data.version || "0.0.0");
  const parts = prev.split(".").map((n) => parseInt(n, 10) || 0);
  while (parts.length < 3) parts.push(0);
  parts[2] += 1;
  const next = parts.join(".");
  data.version = next;
  fs.writeFileSync(file, `${JSON.stringify(data, null, 2)}\n`);
  shown = next;
  console.log(`${name}: ${prev} -> ${next}`);
}
if (shown) console.log(`Pawn Capture ${shown}`);
