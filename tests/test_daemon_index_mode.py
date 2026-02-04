import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


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
get_connection_info = identity_module.get_connection_info


REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_available(name: str) -> bool:
    module = sys.modules.get(name)
    if module is not None and getattr(module, "__spec__", None) is None:
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:
        return False


def _write_stub(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _ensure_stub_modules(stub_root: Path) -> bool:
    needs_stub = False

    if not _module_available("pygments_tldr"):
        needs_stub = True
        _write_stub(
            stub_root / "pygments_tldr" / "__init__.py",
            "def highlight(code, lexer, formatter):\n    return code\n",
        )
        _write_stub(
            stub_root / "pygments_tldr" / "formatters" / "__init__.py",
            "",
        )
        _write_stub(
            stub_root / "pygments_tldr" / "formatters" / "tldr.py",
            "class TLDRFormatter:\n    def __init__(self, **kwargs):\n        self.options = kwargs\n",
        )
        _write_stub(
            stub_root / "pygments_tldr" / "lexers" / "__init__.py",
            "class _DummyLexer:\n    aliases = [\"text\"]\n\n"
            "def get_lexer_for_filename(filename):\n    return _DummyLexer()\n\n"
            "def get_lexer_by_name(name):\n    return _DummyLexer()\n",
        )
        _write_stub(
            stub_root / "pygments_tldr" / "util.py",
            "class ClassNotFound(Exception):\n    pass\n",
        )

    if not _module_available("tiktoken"):
        needs_stub = True
        _write_stub(
            stub_root / "tiktoken" / "__init__.py",
            "class Encoding:\n"
            "    def encode(self, text):\n"
            "        if not text:\n"
            "            return []\n"
            "        return list(text)\n\n"
            "def get_encoding(name):\n"
            "    return Encoding()\n",
        )

    return needs_stub


def _build_subprocess_env(tmp_path: Path) -> dict | None:
    stub_root = tmp_path / "daemon_stubs"
    if not _ensure_stub_modules(stub_root):
        return None

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{stub_root}{os.pathsep}{pythonpath}" if pythonpath else str(stub_root)
    )
    return env


def _start_daemon_process(
    project: Path,
    cache_root: Path,
    index_id: str,
    *,
    env: dict | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "tldr.daemon",
        str(project),
        "--cache-root",
        str(cache_root),
        "--index",
        index_id,
        "--foreground",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def _start_daemon_process_legacy(
    project: Path,
    *,
    env: dict | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "tldr.daemon",
        str(project),
        "--foreground",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def _wait_for_daemon(identity, proc: subprocess.Popen, timeout: float = 10.0) -> None:
    start = time.monotonic()
    last_error = None
    while time.monotonic() - start < timeout:
        if proc.poll() is not None:
            raise AssertionError(f"Daemon exited early with code {proc.returncode}")

        addr, port = get_connection_info(identity)
        try:
            if port is None:
                if not Path(addr).exists():
                    time.sleep(0.05)
                    continue
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(0.2)
                sock.connect(addr)
            else:
                sock = socket.create_connection((addr, port), timeout=0.2)
            sock.close()
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)

    raise AssertionError(f"Daemon did not become ready: {last_error}")


def _send_command(identity, payload: dict, timeout: float = 1.0) -> dict:
    addr, port = get_connection_info(identity)
    data = json.dumps(payload).encode() + b"\n"

    if port is None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(addr)
    else:
        sock = socket.create_connection((addr, port), timeout=timeout)
        sock.settimeout(timeout)

    try:
        sock.sendall(data)
        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
            if b"\n" in response:
                break
        return json.loads(response.decode().strip())
    finally:
        sock.close()


def _stop_daemon(identity, proc: subprocess.Popen) -> None:
    try:
        _send_command(identity, {"cmd": "shutdown"}, timeout=1.0)
    except Exception:
        pass

    if proc.poll() is not None:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_daemon_index_mode_isolated_sockets(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text("hello\n")

    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    index_a = "dep:a"
    index_b = "dep:b"

    env = _build_subprocess_env(tmp_path)
    proc_a = _start_daemon_process(project, cache_root, index_a, env=env)
    proc_b = _start_daemon_process(project, cache_root, index_b, env=env)

    identity_a = resolve_daemon_identity(project, cache_root=cache_root, index_id=index_a)
    identity_b = resolve_daemon_identity(project, cache_root=cache_root, index_id=index_b)

    try:
        _wait_for_daemon(identity_a, proc_a)
        _wait_for_daemon(identity_b, proc_b)

        addr_a, port_a = get_connection_info(identity_a)
        addr_b, port_b = get_connection_info(identity_b)
        assert (addr_a, port_a) != (addr_b, port_b)

        if port_a is None:
            assert Path(addr_a).exists()
            assert Path(addr_b).exists()
        else:
            assert port_a != port_b

        assert _send_command(identity_a, {"cmd": "ping"}).get("status") == "ok"
        assert _send_command(identity_b, {"cmd": "ping"}).get("status") == "ok"
    finally:
        _stop_daemon(identity_a, proc_a)
        _stop_daemon(identity_b, proc_b)


def test_daemon_impact_reads_call_graph_from_cache(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    cache_dir = project / ".tldr" / "cache"
    cache_dir.mkdir(parents=True)

    call_graph = {
        "edges": [
            {"caller": "main", "callee": "helper", "file": "main.py", "line": 12}
        ],
        "nodes": {},
    }
    (cache_dir / "call_graph.json").write_text(json.dumps(call_graph))

    env = _build_subprocess_env(tmp_path)
    proc = _start_daemon_process_legacy(project, env=env)
    identity = resolve_daemon_identity(project)

    try:
        _wait_for_daemon(identity, proc)
        result = _send_command(identity, {"cmd": "impact", "func": "helper"})
        assert result.get("status") == "ok"
        assert result.get("callers") == [
            {"caller": "main", "file": "main.py", "line": 12}
        ]
    finally:
        _stop_daemon(identity, proc)
