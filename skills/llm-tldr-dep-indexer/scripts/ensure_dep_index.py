#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd, env=None):
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _load_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON: {exc}")


def _tldr_prefix():
    tldr_cmd = os.environ.get("TLDR_CMD") or os.environ.get("TLDRF_CMD") or "tldrf"
    return shlex.split(tldr_cmd)


def _resolve_python(dist, module):
    resolver = Path(__file__).with_name("resolve_python_dep.py")
    cmd = [sys.executable, str(resolver), dist]
    if module:
        cmd.extend(["--module", module])
    out = _run(cmd)
    data = _load_json(out)
    return {
        "name": data.get("dist_name") or dist,
        "version": data.get("version"),
        "code_root": data.get("code_root"),
        "origin": data.get("origin"),
        "raw": data,
    }


def _resolve_node(pkg, from_dir):
    resolver = Path(__file__).with_name("resolve_node_dep.js")
    cmd = ["node", str(resolver), pkg]
    if from_dir:
        cmd.extend(["--from", from_dir])
    out = _run(cmd)
    data = _load_json(out)
    return {
        "name": data.get("name") or pkg,
        "version": data.get("version"),
        "code_root": data.get("package_root"),
        "origin": "registry",
        "raw": data,
    }


def _build_index_id(name, version, kind):
    base = f"dep:{name}"
    if version:
        base = f"{base}@{version}"
    if kind:
        base = f"{base}:{kind}"
    return base


def _index_list(cache_root):
    cmd = _tldr_prefix() + ["--cache-root", cache_root, "index", "list"]
    out = _run(cmd)
    return _load_json(out)


def _index_exists(index_list, index_id):
    for entry in index_list.get("indexes", []):
        if entry.get("index_id") == index_id:
            return True
    return False


def _run_index(
    *,
    cache_root,
    index_id,
    code_root,
    lang,
    model,
    device,
    rebuild,
    respect_gitignore,
):
    cmd = _tldr_prefix() + [
        "--cache-root",
        cache_root,
        "--index",
        index_id,
        "semantic",
        "index",
        code_root,
        "--lang",
        lang,
    ]
    if model:
        cmd.extend(["--model", model])
    if device:
        cmd.extend(["--device", device])
    if rebuild:
        cmd.append("--rebuild")
    if not respect_gitignore:
        cmd.append("--no-gitignore")
    out = _run(cmd)
    return out.strip()


def main():
    parser = argparse.ArgumentParser(description="Ensure a dependency index exists and is version-scoped.")
    sub = parser.add_subparsers(dest="dep_type", required=True)

    py_p = sub.add_parser("python", help="Ensure index for a Python dependency")
    py_p.add_argument("dist", help="Distribution name (pip name)")
    py_p.add_argument("--module", help="Import name (if different from dist)")

    node_p = sub.add_parser("node", help="Ensure index for a Node dependency")
    node_p.add_argument("pkg", help="Package name (npm)")
    node_p.add_argument("--from", dest="from_dir", help="Resolve from a specific directory")

    parser.add_argument("--cache-root", default="git", help="Cache root (default: git)")
    parser.add_argument("--index-id", help="Override index id")
    parser.add_argument("--kind", default="site", help="Suffix for index id (default: site)")
    parser.add_argument("--lang", help="Language (defaults to python or javascript)")
    parser.add_argument("--model", help="Embedding model")
    parser.add_argument("--device", help="Embedding device")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild")
    parser.add_argument(
        "--respect-gitignore",
        action="store_true",
        help="Respect gitignore (default: disabled for dependencies)",
    )

    args = parser.parse_args()

    if args.dep_type == "python":
        resolved = _resolve_python(args.dist, args.module)
        lang = args.lang or "python"
    else:
        resolved = _resolve_node(args.pkg, args.from_dir)
        lang = args.lang or "javascript"

    code_root = resolved.get("code_root")
    if not code_root:
        raise SystemExit("Could not resolve code root for dependency.")

    index_id = args.index_id or _build_index_id(
        resolved.get("name"), resolved.get("version"), args.kind
    )

    index_list = _index_list(args.cache_root)
    exists = _index_exists(index_list, index_id)

    action = "reuse"
    detail = None
    if args.rebuild or not exists:
        action = "rebuilt" if args.rebuild and exists else "created"
        detail = _run_index(
            cache_root=args.cache_root,
            index_id=index_id,
            code_root=code_root,
            lang=lang,
            model=args.model,
            device=args.device,
            rebuild=args.rebuild,
            respect_gitignore=args.respect_gitignore,
        )

    result = {
        "action": action,
        "index_id": index_id,
        "cache_root": args.cache_root,
        "code_root": code_root,
        "name": resolved.get("name"),
        "version": resolved.get("version"),
        "origin": resolved.get("origin"),
        "detail": detail,
        "resolver": resolved.get("raw"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
