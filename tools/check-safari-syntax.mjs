/**
 * Parse every module the site ships with this Mac's own JavaScriptCore, the engine of its Safari, so syntax that
 * Safari cannot parse (regex lookbehind before 16.4, class static blocks, ...) fails here instead of in a
 * visitor's browser, where one bad literal stops a whole chunk from loading. macOS only; skips elsewhere.
 *
 *     npm run check:safari
 */
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const JSC = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc";
if (!existsSync(JSC)) {
  console.log("check-safari-syntax: no system JavaScriptCore (macOS only); skipped");
  process.exit(0);
}
const files = [
  ...readdirSync("assets").filter((f) => f.endsWith(".js")).map((f) => join("assets", f)),
  ...execFileSync("git", ["ls-files", "web/*.js", "shared/*.js", "webgpu/*.js", "cpu/*.js"], { encoding: "utf8" })
    .split("\n").filter(Boolean),
];
const script = join(mkdtempSync(join(tmpdir(), "safari-syntax-")), "check.js");
writeFileSync(script, `let bad = 0;
for (const f of arguments) { try { checkModuleSyntax(read(f)); } catch (e) { bad++; print("FAIL " + f + ": " + e); } }
print("checked " + arguments.length + ", failed " + bad);`);
const r = spawnSync(JSC, [script, "--", ...files], { encoding: "utf8" });
process.stdout.write(r.stdout + r.stderr);
const safari = spawnSync("/usr/libexec/PlistBuddy", ["-c", "Print CFBundleShortVersionString", "/Applications/Safari.app/Contents/Info.plist"], { encoding: "utf8" }).stdout.trim();
if (safari) console.log(`(engine of this Mac's Safari ${safari})`);
process.exit(r.status !== 0 || /FAIL /.test(r.stdout) ? 1 : 0);
