# Sphinx configuration of the documentation site. Build: see docs/README.md.
#
# The live demo and the benchmark page (_extra/) are static pages next to the documentation. They load the
# WebAssembly module from lib/, which this file copies in after the build, from a browser build of the
# library: JKP_WASM_DIR (default build/wasm/web). Without it the build warns, so `sphinx-build -W` fails.
import os
import shutil

from sphinx.util import logging

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

project = "just-keypoints"
author = "the just-keypoints authors"
copyright = "the just-keypoints authors"
release = ""
version = ""

extensions = ["myst_parser"]
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", ".venv", "README.md", "_theme", "_extra"]

html_theme = "jkp"
html_theme_path = ["_theme"]
html_title = project
html_extra_path = ["_extra"]
html_copy_source = False
html_show_sourcelink = False

# The module's files and the weights, served from lib/.
LIB_FILES = {"jkp.js": "jkp.js", "jkp_wasm.js": "jkp_wasm.js", "jkp_wasm.wasm": "jkp_wasm.wasm",
             "aliked-n16.jkpw": "aliked-n16.jkpw", "raco.jkpw": "raco.jkpw",
             "xfeat.jkpw": "xfeat.jkpw"}


def copy_module(app, exception):
    if exception or app.builder.format != "html":
        return
    log = logging.getLogger(__name__)
    src = os.environ.get("JKP_WASM_DIR", os.path.join(ROOT, "build", "wasm", "web"))
    missing = [f for f in LIB_FILES if not os.path.isfile(os.path.join(src, f))]
    if missing:
        log.warning("the demo needs a browser build of the library; not found in %s: %s (set JKP_WASM_DIR)",
                    src, ", ".join(missing))
        return
    dest = os.path.join(app.outdir, "lib")
    os.makedirs(dest, exist_ok=True)
    for f, name in LIB_FILES.items():
        shutil.copyfile(os.path.join(src, f), os.path.join(dest, name))
    log.info("demo module copied from %s", src)


def setup(app):
    app.connect("build-finished", copy_module)
