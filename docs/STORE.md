# Listing TokenStation5 on the Prospero Store

The [Prospero Store](https://github.com/darkxex/ProsperoStore) installs apps
from the [`ps5-homebrew-catalog`](https://github.com/darkxex/ps5-homebrew-catalog).
Listing is a pull request to that catalog that adds one record,
`apps/PPSA99555.json`, pointing at a GitHub Release ZIP in this repository.

The [`release`](../.github/workflows/release.yml) workflow does the packaging
for you: push a version tag and it builds the title, produces
`PPSA99555.zip` (with the `PPSA99555/` folder at the archive root, which is
what the catalog requires), computes the SHA-256, and attaches both the ZIP
and a ready-to-submit `PPSA99555.json` to the release.

## Release checklist

1. **Bump the version.** Raise `contentVersion` in
   [`ps5/sce_sys/param.json`](../ps5/sce_sys/param.json) in PlayStation format
   (e.g. `01.000.001`) and commit it. Consoles use it to detect updates.
   `titleId` (`PPSA99555`), `conceptId` and `contentId` must stay the same
   across releases so updates keep the app's data.
2. **Tag and push.** Use the version as the tag, not a generic name:
   ```sh
   git tag 01.000.001 && git push origin 01.000.001
   ```
   The `release` workflow runs and publishes a GitHub Release with
   `PPSA99555.zip`, `PPSA99555.zip.sha256`, and `PPSA99555.json`.
3. **Submit to the catalog.** Fork `ps5-homebrew-catalog`, copy the generated
   `PPSA99555.json` into `apps/PPSA99555.json`, and open a pull request **from
   the account that owns this repository**, changing only that one file. The
   catalog's *Validate submission* check reports any problems; push fixes to
   the same branch to re-run it. A maintainer reviews and merges, and
   prosperostore.net updates after the merge.
4. **Updating later.** Publish a new tagged release, then open a catalog PR
   editing `version`, `artifact_url` and `sha256` (and `icon_url` if the icon
   moved). Never replace the asset on a release that's already listed — its
   SHA-256 would stop matching.

## The record

The workflow writes these fields; they already match this repo:

| Field | Value |
|---|---|
| `titleid` | `PPSA99555` |
| `name` | `TokenStation5` |
| `kind` | `app` |
| `category` | `ai` |
| `license` | `GPL-3.0-or-later` (the built title is GPL-3.0-or-later as a whole; see [LICENSING.md](../LICENSING.md)) |
| `author` | `Jared Maxwell` |
| `version` | the release tag, without a leading `v` |
| `source_repo` | `https://github.com/ThatRetiredDude/TokenStation5` |
| `artifact_url` | the release's `PPSA99555.zip` |
| `sha256` | computed from the ZIP |
| `icon_url` | `ps5/sce_sys/icon0.png`, pinned to the tag (512×512 PNG) |

## ⚠️ One thing to resolve before submitting: the license check

The catalog requires the record's `license` to **match the license GitHub
detects for the repository**. GitHub currently reports this repo's license as
**`NOASSERTION` ("Other")**, because the repository root is
[PolyForm Small Business](../LICENSE.md) and only the `ps5/` title and the
built artifact are GPL-3.0-or-later (see [LICENSING.md](../LICENSING.md)).

So a record with `license: GPL-3.0-or-later` may fail the catalog's validation
even though it correctly describes the distributed title. Options:

- Ask the catalog maintainer whether a `NOASSERTION` repo can list with the
  artifact's license (dual-licensed homebrew is common).
- Add a root `LICENSE`/SPDX file that makes GitHub detect a single license —
  this changes how the repo is presented, so it's a deliberate decision.

This is intentionally left for you to decide; the workflow and record are
otherwise ready.
