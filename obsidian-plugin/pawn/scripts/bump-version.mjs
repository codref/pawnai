/**
 * Bump the plugin patch version before a production build.
 *
 * Obsidian shows manifest.json in Settings → Community plugins. Each local
 * `npm run build` advances that patch so a reload is visible.
 * CI sets CI=true and must not bump: release tags have to match the committed
 * manifest. PAWN_SKIP_VERSION_BUMP=1 does the same.
 */
import { readFileSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

function read(name) {
  return readFileSync(join(root, name), "utf8");
}

function write(name, text) {
  writeFileSync(join(root, name), text);
}

const manifest = JSON.parse(read("manifest.json"));
const current = String(manifest.version || "");

if (process.env.CI === "true" || process.env.PAWN_SKIP_VERSION_BUMP === "1") {
  console.log(`plugin version ${current} (unchanged)`);
  process.exit(0);
}

const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(current);
if (!match) {
  console.error(`manifest.json version ${current} is not MAJOR.MINOR.PATCH`);
  process.exit(1);
}
const next = `${match[1]}.${match[2]}.${Number(match[3]) + 1}`;

write(
  "manifest.json",
  read("manifest.json").replace(`"version": "${current}"`, `"version": "${next}"`),
);

const pkg = read("package.json").replace(/"version": "[^"]+"/, `"version": "${next}"`);
write("package.json", pkg);

const lockOwn = JSON.parse(read("package-lock.json")).version;
let lockHits = 0;
const lock = read("package-lock.json").replace(/"version": "([^"]+)"/g, (full, ver) => {
  if (lockHits < 2 && ver === lockOwn) {
    lockHits += 1;
    return `"version": "${next}"`;
  }
  return full;
});
write("package-lock.json", lock);

const versions = JSON.parse(read("versions.json"));
versions[next] = manifest.minAppVersion;
write("versions.json", `${JSON.stringify(versions, null, 2)}\n`);

console.log(`plugin version ${current} -> ${next}`);
