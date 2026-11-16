# Documentation site

Sphinx with MyST Markdown, built from this folder, with the site's own theme in `_theme/jkp` (after just-ba's
documentation theme). `_extra/` is copied into the site as it is: the live demo (`demo/`), the benchmark page
(`bench/`), their shared code and style, the sample images and Cloudflare's `_headers`. Both pages load the
WebAssembly module from `lib/`, which `conf.py` copies in from a browser build of the library.

Rule for every page: what visitors read carries no references to the project's notes (PLAN.md, milestones,
commit hashes) and no links to the repository, which is private.

```bash
uv venv --python 3.12 docs/.venv && uv pip install --python docs/.venv/bin/python -r docs/requirements.txt
```

With a browser build in `build/wasm` (README.md, "WebGPU and the browser"; `tools/fetch_webgpu.sh browser`
fetches emdawnwebgpu):

```bash
docs/.venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
python3 -m http.server 8792 --bind 127.0.0.1 --directory docs/_build/html   # then open /demo/ in Chrome
```

`JKP_WASM_DIR` points at another build's `web/` folder. Without a module the build warns, so `-W` fails.

## Deploying

The `docs` workflow (`.github/workflows/docs.yml`) is started by hand:

```bash
gh workflow run docs                    # build the module and the site
gh workflow run docs -f deploy=true     # the same, then publish to Cloudflare Pages
```

Deploying needs the repository secrets `CLOUDFLARE_API_TOKEN` (permission "Cloudflare Pages: Edit") and
`CLOUDFLARE_ACCOUNT_ID`, and a Pages project made for direct upload, named by the repository variable
`CLOUDFLARE_PAGES_PROJECT` (default `just-keypoints`). From a machine where wrangler is logged in, the same
upload is:

```bash
npx --yes wrangler@4.162.0 pages deploy docs/_build/html --project-name just-keypoints --branch main
```
