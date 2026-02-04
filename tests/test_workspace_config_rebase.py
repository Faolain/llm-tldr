from pathlib import Path
import json

from tldr.cross_file_calls import scan_project
from tldr.workspace import load_workspace_config


def test_workspace_config_rebase(tmp_path: Path):
    cache_root = tmp_path / "repo"
    scan_root = cache_root / "packages" / "foo"
    (scan_root / "src").mkdir(parents=True)
    (scan_root / "dist").mkdir(parents=True)

    (scan_root / "src" / "keep.py").write_text("def keep():\n    return 1\n")
    (scan_root / "dist" / "ignored.py").write_text("def ignore():\n    return 2\n")

    claude_dir = cache_root / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "workspace.json").write_text(
        json.dumps({"excludePatterns": ["**/dist/**"]})
    )

    config = load_workspace_config(cache_root)
    files = scan_project(
        scan_root,
        "python",
        workspace_config=config,
        workspace_root=cache_root,
        respect_ignore=False,
    )
    files_str = [str(Path(f)) for f in files]
    assert any("keep.py" in f for f in files_str)
    assert not any("dist/ignored.py" in f for f in files_str)
