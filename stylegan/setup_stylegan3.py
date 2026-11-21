#!/usr/bin/env python3
"""
Download NVIDIA's official StyleGAN2-ADA code and make it run on current Python/PyTorch.

"""
import py_compile
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "third_party" / "stylegan3"
REPO = "https://github.com/NVlabs/stylegan3.git"

PKG_FILES = ["torch_utils/ops/conv2d_gradfix.py", "torch_utils/ops/grid_sample_gradfix.py"]
UTIL_FILE = "dnnlib/util.py"
MISC_FILE = "torch_utils/misc.py"
OP_FILES = ["torch_utils/ops/bias_act.py", "torch_utils/ops/upfirdn2d.py", "torch_utils/ops/filtered_lrelu.py"]
FALLBACK_MARK = "# gan_vs_diffusion: fall back to the reference implementation"
FALLBACK = (
    "\n\n" + FALLBACK_MARK + "\n"
    "# NVIDIA's fast CUDA kernels are compiled on first use; on newer CUDA/compilers that build can\n"
    "# fail. Instead of crashing, use the plain-PyTorch reference implementation (same results, slower).\n"
    "_gvd_init_orig = _init\n"
    "_gvd_failed = False\n\n"
    "def _init():\n"
    "    global _gvd_failed\n"
    "    if _gvd_failed:\n"
    "        return False\n"
    "    try:\n"
    "        return _gvd_init_orig()\n"
    "    except Exception as e:  # noqa: BLE001\n"
    "        _gvd_failed = True\n"
    "        print(f'Note: custom CUDA kernel unavailable ({e.__class__.__name__}); using the reference implementation.')\n"
    "        return False\n"
)
STRTOBOOL = (
    "def strtobool(val):\n"
    "    val = str(val).lower()\n"
    "    if val in ('y', 'yes', 't', 'true', 'on', '1'):\n        return 1\n"
    "    if val in ('n', 'no', 'f', 'false', 'off', '0'):\n        return 0\n"
    "    raise ValueError(f'invalid truth value {val!r}')"
)


def compiles(paths):
    try:
        for p in paths:
            py_compile.compile(str(p), doraise=True)
        return True
    except (py_compile.PyCompileError, FileNotFoundError):
        return False


def patch():
    for rel in PKG_FILES:
        p = DEST / rel
        s = p.read_text()
        # only a top-level (unindented) import is replaced -> running twice changes nothing
        s = re.sub(r"(?m)^from pkg_resources import parse_version$",
                   "from packaging.version import parse as parse_version", s)
        p.write_text(s)
    p = DEST / UTIL_FILE
    s = p.read_text()
    s = re.sub(r"(?m)^from distutils\.util import strtobool$", STRTOBOOL, s)
    p.write_text(s)
    # PyTorch >= 2.x: Sampler.__init__ no longer takes the dataset argument
    p = DEST / MISC_FILE
    p.write_text(p.read_text().replace("super().__init__(dataset)", "super().__init__()"))
    for rel in OP_FILES:
        p = DEST / rel
        s = p.read_text()
        if FALLBACK_MARK not in s:
            p.write_text(s + FALLBACK)


def main():
    files = [DEST / r for r in PKG_FILES + [UTIL_FILE, MISC_FILE] + OP_FILES]
    if DEST.exists() and not compiles(files):
        print("Existing StyleGAN code is broken (earlier setup run) - downloading it again.")
        shutil.rmtree(DEST)
    if not DEST.exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, str(DEST)], check=True)
    patch()
    if not compiles(files):
        raise SystemExit("Patching failed - please report this error.")
    print(f"StyleGAN2-ADA code ready in {DEST}")


if __name__ == "__main__":
    main()
