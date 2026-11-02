/** Run with `bun tests/neo-review-git.integration.ts`; real Git, no model. */
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, symlinkSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { autoDiffOf } from "../hooks/register.ts";

const scratch = realpathSync(mkdtempSync(join(tmpdir(), "neo-review-git-")));
const repo = join(scratch, "repo");
mkdirSync(repo);
const git = (...args: string[]) => {
  const r = spawnSync("git", args, { cwd: repo, encoding: "utf8" });
  assert.equal(r.status, 0, r.stderr || String(r.error));
  return r.stdout;
};
const calls: string[][] = [];
const host = { process: { run: async (argv: string[], options: { cwd: string }) => {
  calls.push(argv);
  const r = spawnSync(argv[0]!, argv.slice(1), { cwd: options.cwd, encoding: "utf8" });
  return { exitCode: r.status ?? 128, stdout: r.stdout ?? "", stderr: r.stderr ?? "" };
} } } as any;
const review = (path: string, cwd = repo, hasHead = true) => autoDiffOf(host, cwd, repo, path, hasHead);
let checks = 0;
async function rejected(path: string) {
  assert.equal(await review(path), undefined, path);
  checks++;
}
try {
  git("init", "-q");
  writeFileSync(join(repo, "tracked.ts"), "export const value = 1;\n");
  writeFileSync(join(repo, "README.md"), "A contract.\n");
  mkdirSync(join(repo, "src"));
  writeFileSync(join(repo, "src/gone.ts"), "export const gone = 1;\n");
  git("add", ".");
  // Owned staged files in an unborn repository still count as additions.
  assert.match((await review("tracked.ts", repo, false))!.diff, /new file mode/); checks++;
  git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "base");
  const outside = join(scratch, "RESUME.md");
  writeFileSync(outside, "OUTSIDE_PROOF_PROCESS\n");
  await rejected(outside);
  mkdirSync(join(scratch, "scratchpad"));
  writeFileSync(join(scratch, "scratchpad/note.md"), "PRIVATE_NOTE\n");
  await rejected(join(scratch, "scratchpad/note.md"));
  assert.equal(calls.some((c) => c[1] === "diff" && c.includes(outside)), false);
  await rejected("tracked.ts"); // unchanged
  writeFileSync(join(repo, "tracked.ts"), "export const value = 2;\n");
  assert.match((await review("tracked.ts"))!.diff, /\+export const value = 2/); checks++;
  writeFileSync(join(repo, "README.md"), "Changed contract.\n");
  assert.equal((await review("README.md"))!.path, "README.md"); checks++;
  mkdirSync(join(repo, "run/bin"), { recursive: true });
  writeFileSync(join(repo, "run/bin/ship"), "#!/bin/sh\nprintf 'ship\\n'\n");
  assert.equal((await review("../run/bin/ship", join(repo, "src")))!.path, "run/bin/ship"); checks++;
  writeFileSync(join(repo, "src/[literal].ts"), "export const n = 1;\n");
  assert.equal((await review("[literal].ts", join(repo, "src")))!.path, "src/[literal].ts"); checks++;
  writeFileSync(join(repo, ".gitignore"), "notes/\n");
  mkdirSync(join(repo, "notes"));
  writeFileSync(join(repo, "notes/state.md"), "ignored\n");
  await rejected("notes/state.md");
  const sibling = join(scratch, "sibling");
  git("worktree", "add", "--detach", sibling, "HEAD");
  writeFileSync(join(sibling, "tracked.ts"), "sibling change\n");
  await rejected(join(sibling, "tracked.ts"));
  symlinkSync(outside, join(repo, "escape.ts"));
  symlinkSync(scratch, join(repo, "escape-dir"));
  await rejected("escape.ts");
  await rejected("escape-dir/RESUME.md");
  // The outer index can still track files inside a newly nested checkout.
  assert.equal(spawnSync("git", ["init", "-q", join(repo, "src")]).status, 0);
  writeFileSync(join(repo, "src/gone.ts"), "nested change\n");
  const beforeNested = calls.length;
  await rejected("src/gone.ts");
  assert.equal(calls.slice(beforeNested).some((c) => c[1] === "diff"), false);
  rmSync(join(repo, "src"), { recursive: true });
  assert.match((await review("src/gone.ts"))!.diff, /deleted file mode/); checks++;
  const failed = { process: { run: async () => ({ exitCode: 128, stdout: "untrusted", stderr: "failure" }) } } as any;
  assert.equal(await autoDiffOf(failed, repo, repo, "tracked.ts", true), undefined); checks++;
  console.log(`${checks} real-Git admission checks passed (including TMPDIR repository, subdirectory cwd, literal paths, deleted parents and unborn HEAD).`);
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
