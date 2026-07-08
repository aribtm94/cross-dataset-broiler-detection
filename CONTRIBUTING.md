# Contributing

Thanks for your interest. This is a research / thesis codebase, so contributions
are mostly about reproducibility and clarity rather than new features.

## Getting set up

1. Create the two environments: `./scripts/setup_env.ps1` (or `.sh`).
2. Fetch the required assets: `scripts/download_assets.py --required`
   (see [`docs/DATA_SETUP.md`](docs/DATA_SETUP.md)).
3. Copy `.env.example` to `.env` if you need to re-pull external datasets.

## Ground rules

- **Never commit large files.** Datasets, weights, the MOWA checkpoint, derived
  data, and papers all belong in the Google-Drive bundles, not Git. The
  `.gitignore` is set up to block them; do not add `!` exceptions for big files.
- **Never commit secrets.** API keys go in `.env` (git-ignored). `.env.example`
  holds placeholders only.
- **Respect third-party licenses.** MOWA is non-commercial (S-Lab 1.0). Do not
  add code or docs that imply commercial use of the bundled models. See
  [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
- Keep `assets_manifest.json` the single source of truth. If you add/change a
  bundle, update it via `scripts/build_release_bundles.py` (it rewrites sizes and
  hashes) rather than editing sizes by hand.

## Adding a new large asset bundle

1. Add an entry to `assets_manifest.json` (`id`, `file`, `sources`, `extract_to`,
   `required`, and a `notes` line).
2. Run `scripts/build_release_bundles.py --only <id>` to build + hash it.
3. Upload `dist/<file>` to Google Drive (share = anyone with the link).
4. Paste the link into the manifest `gdrive_url` and the table in
   `docs/DATA_SETUP.md`.

## Style

Match the surrounding code. Scripts use `argparse`, `pathlib`, and the shared
helpers in `src/common.py`. Existing inline comments are in Indonesian; keep new
comments consistent with the file you are editing.
