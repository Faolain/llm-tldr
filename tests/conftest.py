from pathlib import Path
import sys
import types
from typing import Callable

import pytest


if "pygments_tldr" not in sys.modules:
    pkg = types.ModuleType("pygments_tldr")
    pkg.highlight = lambda *args, **kwargs: ""

    formatters = types.ModuleType("pygments_tldr.formatters")
    tldr_mod = types.ModuleType("pygments_tldr.formatters.tldr")
    lexers = types.ModuleType("pygments_tldr.lexers")
    util = types.ModuleType("pygments_tldr.util")

    class TLDRFormatter:
        pass

    tldr_mod.TLDRFormatter = TLDRFormatter
    formatters.tldr = tldr_mod
    pkg.formatters = formatters

    lexers.get_lexer_for_filename = lambda *args, **kwargs: None
    lexers.get_lexer_by_name = lambda *args, **kwargs: None
    pkg.lexers = lexers

    class ClassNotFound(Exception):
        pass

    util.ClassNotFound = ClassNotFound
    pkg.util = util

    sys.modules["pygments_tldr"] = pkg
    sys.modules["pygments_tldr.formatters"] = formatters
    sys.modules["pygments_tldr.formatters.tldr"] = tldr_mod
    sys.modules["pygments_tldr.lexers"] = lexers
    sys.modules["pygments_tldr.util"] = util


@pytest.fixture
def write_project() -> Callable[[Path, dict[str, str]], None]:
    """Write a small test project tree under `root`."""

    def _write(root: Path, files: dict[str, str]) -> None:
        for rel_path, content in files.items():
            file_path = root / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

    return _write
