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

1. **Check the build on the console.** Install it on the PS5 and run through
   what the release changes, with the real models; a green CI build alone is
   not a pass. Then reread the README against what the build does, and bring
   the roadmap (the `roadmap` branch, PR #18) up to date: where we are, open,
   done.
2. **Bump the version.** Raise `contentVersion` in
   [`ps5/sce_sys/param.json`](../ps5/sce_sys/param.json) in PlayStation format
   (e.g. `01.000.001`) and commit it. Consoles use it to detect updates.
   `titleId` (`PPSA99555`), `conceptId` and `contentId` must stay the same
   across releases so updates keep the app's data. Leave `downloadDataSize`
   at 256: a bigger value (65536 was tried) makes the PS5 refuse to start the
   title with CE-107694-7.
3. **Tag and push.** Use the version as the tag, not a generic name:
   ```sh
   git tag 01.000.001 && git push origin 01.000.001
   ```
   The `release` workflow runs and publishes a GitHub Release with
   `PPSA99555.zip`, `PPSA99555.zip.sha256`, and `PPSA99555.json`.
4. **Submit to the catalog.** Fork `ps5-homebrew-catalog`, copy the generated
   `PPSA99555.json` into `apps/PPSA99555.json`, and open a pull request **from
   the account that owns this repository**, changing only that one file. The
   catalog's *Validate submission* check reports any problems; push fixes to
   the same branch to re-run it. A maintainer reviews and merges, and
   prosperostore.net updates after the merge.
5. **Updating later.** Publish a new tagged release, then open a catalog PR
   editing `version`, `artifact_url` and `sha256` (and `icon_url` if the icon
   moved). Never replace the asset on a release that's already listed — its
   SHA-256 would stop matching. An update keeps the user's settings, keys and
   chats (they live in `/download0` and come back) but not their downloaded
   models, so the release notes say so.

## The record

The workflow writes these fields; they already match this repo:

| Field | Value |
|---|---|
| `titleid` | `PPSA99555` |
| `name` | `TokenStation5` |
| `kind` | `app` |
| `category` | `ai` |
| `license` | `Unlicensed` (the catalog's sentinel for a repo GitHub can't map to one SPDX id; see the license note below) |
| `author` | `ThatRetiredDude` |
| `version` | the release tag, without a leading `v` |
| `source_repo` | `https://github.com/ThatRetiredDude/TokenStation5` |
| `artifact_url` | the release's `PPSA99555.zip` |
| `sha256` | computed from the ZIP |
| `icon_url` | `ps5/sce_sys/icon0.png`, pinned to the tag (512×512 PNG) |

## The license field (why it's `Unlicensed`)

The catalog's submission check compares the record's `license` with the
license GitHub detects for the repository. GitHub reports this repo as
**`NOASSERTION`** — it can't map it to a single SPDX id — because the root is
[PolyForm Small Business](../LICENSE.md) and only the `ps5/` title and the
built artifact are GPL-3.0-or-later (see [LICENSING.md](../LICENSING.md)).

For exactly this case the catalog validator
([`catalog/verify.py`](https://github.com/darkxex/ps5-homebrew-catalog/blob/main/catalog/verify.py))
accepts the sentinel value **`"Unlicensed"`** with no error and no warning when
GitHub's detection is `NOASSERTION`. So the generated record uses
`"license": "Unlicensed"`, the submission passes cleanly with no manual gate,
and the repository's actual licensing is left exactly as-is (PolyForm at the
root, GPL-3.0-or-later for the title). `"Unlicensed"` here means only "GitHub
could not detect an SPDX license," not that the software is unlicensed.

If you ever give the repo a root license GitHub can map to one SPDX id, switch
the record's `license` to that id to match — until then, `"Unlicensed"` is the
correct value.
