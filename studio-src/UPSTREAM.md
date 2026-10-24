# Vendored AtomWorks Studio web app

Source: [RosettaCommons/atomworks-studio](https://github.com/RosettaCommons/atomworks-studio) tag `v0.0.2`
(commit `a49cf38`), directory `packages/web/src`, BSD-3-Clause (`LICENSE.md`).

`src/wasm/` is the `wasm-pack` output of upstream `packages/core` (Rust 1.97.1, `wasm-pack build --target web`),
committed so a rebuild needs only Node. Regenerate it only when updating upstream.

Local changes are ordinary commits on top of the import commit; `git log -- studio-src` lists them.

## Build

```sh
cd studio-src && npm ci && npm run build   # writes ../assets/studio-*.js|css and updates ../index.html, ../studio.html
```

Commit the rebuilt `assets/` together with the source change. Emitted files are content-hashed, so browsers never
run a stale bundle against a newer `web/app.js`.

## Updating upstream

Re-import `packages/web/src` from a newer tag in a commit of its own, then rebase or re-apply the local commits.
