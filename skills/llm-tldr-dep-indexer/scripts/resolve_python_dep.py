#!/usr/bin/env python3
import argparse
import json
import os
from importlib import util as importlib_util
from importlib import metadata as importlib_metadata
from pathlib import Path

NATIVE_EXTS = {".so", ".pyd", ".dll", ".dylib"}


def _find_dist_info(dist):
    dist_info_dir = None
    if dist.files:
        for rel in dist.files:
            rel_str = str(rel)
            if rel_str.endswith("METADATA") and ".dist-info" in rel_str:
                dist_info_dir = dist.locate_file(rel).parent
                break
    return dist_info_dir


def _read_direct_url(dist_info_dir: Path | None):
    if not dist_info_dir:
        return None
    direct_url = dist_info_dir / "direct_url.json"
    if not direct_url.exists():
        return None
    try:
        return json.loads(direct_url.read_text())
    except Exception:
        return None


def _origin_from_direct_url(direct_url: dict | None) -> str:
    if not direct_url:
        return "registry"
    if "vcs_info" in direct_url:
        return "vcs"
    dir_info = direct_url.get("dir_info")
    if isinstance(dir_info, dict) and dir_info:
        return "local"
    if direct_url.get("url"):
        return "url"
    return "unknown"


def _module_path(module_name: str | None):
    if not module_name:
        return None
    spec = importlib_util.find_spec(module_name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return str(Path(list(spec.submodule_search_locations)[0]).resolve())
    if spec.origin:
        return str(Path(spec.origin).resolve())
    return None


def _scan_native_exts(root: Path, max_files: int = 20000):
    py_count = 0
    native_count = 0
    scanned = 0
    truncated = False
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            scanned += 1
            if scanned > max_files:
                truncated = True
                return py_count, native_count, truncated
            ext = Path(name).suffix.lower()
            if ext == ".py":
                py_count += 1
            elif ext in NATIVE_EXTS:
                native_count += 1
    return py_count, native_count, truncated


def main():
    parser = argparse.ArgumentParser(description="Resolve Python dependency install info")
    parser.add_argument("dist", help="Distribution name (pip name)")
    parser.add_argument("--module", help="Import name (if different from dist)")
    args = parser.parse_args()

    try:
        dist = importlib_metadata.distribution(args.dist)
    except importlib_metadata.PackageNotFoundError:
        raise SystemExit(f"Package not found: {args.dist}")

    version = dist.version
    dist_info_dir = _find_dist_info(dist)
    direct_url = _read_direct_url(dist_info_dir)
    origin = _origin_from_direct_url(direct_url)

    module_name = args.module or args.dist
    module_path = _module_path(module_name)

    code_root = None
    if module_path:
        path = Path(module_path)
        code_root = str(path if path.is_dir() else path.parent)

    py_count = None
    native_count = None
    truncated = None
    has_native_ext = None
    if code_root and Path(code_root).exists():
        py_count, native_count, truncated = _scan_native_exts(Path(code_root))
        has_native_ext = native_count > 0

    result = {
        "dist_name": args.dist,
        "version": version,
        "module_name": module_name,
        "module_path": module_path,
        "code_root": code_root,
        "dist_info_path": str(dist_info_dir) if dist_info_dir else None,
        "direct_url": direct_url,
        "origin": origin,
        "py_file_count": py_count,
        "native_file_count": native_count,
        "has_native_ext": has_native_ext,
        "scan_truncated": truncated,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
