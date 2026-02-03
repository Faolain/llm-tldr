from pathlib import Path
import importlib.util
import sys


def _load_identity_module():
    module_path = Path(__file__).resolve().parents[1] / "tldr" / "daemon" / "identity.py"
    spec = importlib.util.spec_from_file_location("tldr.daemon.identity", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tldr.daemon.identity module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


identity_module = _load_identity_module()
resolve_daemon_identity = identity_module.resolve_daemon_identity


def test_daemon_identity_isolated(tmp_path: Path):
    cache_root = tmp_path / "cache"
    cache_root_alt = tmp_path / "cache_alt"
    scan_root = tmp_path / "scan"
    scan_root_alt = tmp_path / "scan_alt"
    cache_root.mkdir()
    cache_root_alt.mkdir()
    scan_root.mkdir()
    scan_root_alt.mkdir()

    id_a = resolve_daemon_identity(scan_root, cache_root=cache_root, index_id="dep:a")
    id_b = resolve_daemon_identity(scan_root, cache_root=cache_root, index_id="dep:b")
    assert id_a.hash != id_b.hash

    id_same = resolve_daemon_identity(scan_root_alt, cache_root=cache_root, index_id="dep:a")
    assert id_same.hash == id_a.hash

    id_other_cache = resolve_daemon_identity(scan_root, cache_root=cache_root_alt, index_id="dep:a")
    assert id_other_cache.hash != id_a.hash

    legacy = resolve_daemon_identity(scan_root)
    assert legacy.hash != id_a.hash
