"""TLDR ignore file handling (.tldrignore + .gitignore).

Provides gitignore-style pattern matching for excluding files from indexing.
Uses pathspec library for gitignore-compatible pattern matching.

Precedence (highest to lowest):
1. .tldrignore patterns (explicit include/exclude)
2. .gitignore patterns (via git check-ignore, if in git repo)
3. Default patterns (if no .tldrignore exists)
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from pathspec import PathSpec


class _NullSpec:
    patterns: list = []

    def match_file(self, _path: str) -> bool:
        return False

# Default .tldrignore template
DEFAULT_TEMPLATE = """\
# TLDR ignore patterns (gitignore syntax)
# Auto-generated - review and customize for your project
# Docs: https://git-scm.com/docs/gitignore

# ===================
# Dependencies
# ===================
node_modules/
.tldr/
.venv/
venv/
env/
__pycache__/
.tox/
.nox/
.pytest_cache/
.mypy_cache/
.ruff_cache/
vendor/
Pods/

# ===================
# Build outputs
# ===================
dist/
build/
out/
target/
*.egg-info/
*.whl
*.pyc
*.pyo

# ===================
# Binary/large files
# ===================
*.so
*.dylib
*.dll
*.exe
*.bin
*.o
*.a
*.lib

# ===================
# IDE/editors
# ===================
.idea/
.vscode/
*.swp
*.swo
*~

# ===================
# Security (always exclude)
# ===================
.env
.env.*
*.pem
*.key
*.p12
*.pfx
credentials.*
secrets.*

# ===================
# Version control
# ===================
.git/
.hg/
.svn/

# ===================
# OS files
# ===================
.DS_Store
Thumbs.db

# ===================
# Project-specific
# Add your custom patterns below
# ===================
# large_test_fixtures/
# data/
"""


@lru_cache(maxsize=128)
def resolve_git_root(project_dir: str | Path) -> Path | None:
    """Resolve git root for a directory (git rev-parse --show-toplevel)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(project_dir),
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        root = result.stdout.decode().strip()
        return Path(root).resolve() if root else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


@lru_cache(maxsize=128)
def is_git_repo(project_dir: str) -> bool:
    """Check if directory is inside a git repository."""
    return resolve_git_root(project_dir) is not None


def is_gitignored(
    file_path: str | Path,
    project_dir: str | Path,
    git_root: str | Path | None = None,
) -> bool:
    """Check if a file is ignored by .gitignore using git check-ignore.

    This handles all gitignore complexity including:
    - Nested .gitignore files
    - Pattern precedence
    - Negation patterns (!)
    - Directory-relative patterns

    Args:
        file_path: Path to the file to check
        project_dir: Root directory of the git repo

    Returns:
        True if file is gitignored, False otherwise
    """
    project_path = Path(git_root) if git_root is not None else Path(project_dir)
    file_path = Path(file_path)

    # Make path relative for git check-ignore
    try:
        rel_path = file_path.relative_to(project_path)
    except ValueError:
        rel_path = file_path

    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(rel_path)],
            cwd=str(project_path),
            capture_output=True,
            timeout=5,
        )
        # Return code 0 = ignored, 1 = not ignored, 128 = error
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def batch_gitignored(
    file_paths: "Sequence[str | Path]",
    project_dir: str | Path,
    git_root: str | Path | None = None,
) -> set[str]:
    """Check multiple files against .gitignore in a single subprocess call.

    This is ~35x faster than calling is_gitignored() per file.

    Args:
        file_paths: List of file paths to check
        project_dir: Root directory of the git repo

    Returns:
        Set of relative path strings that ARE gitignored
    """
    if not file_paths:
        return set()

    project_path = Path(git_root) if git_root is not None else Path(project_dir)

    # Convert to relative paths
    rel_paths = []
    for fp in file_paths:
        fp = Path(fp)
        try:
            rel_paths.append(str(fp.relative_to(project_path)))
        except ValueError:
            rel_paths.append(str(fp))

    try:
        # Use stdin with null-separated paths for efficiency
        result = subprocess.run(
            ["git", "check-ignore", "--stdin", "-z"],
            input="\0".join(rel_paths).encode(),
            capture_output=True,
            cwd=str(project_path),
            timeout=30,
        )
        # Output is null-separated list of ignored files
        if result.returncode == 0 and result.stdout:
            ignored = result.stdout.decode().rstrip("\0").split("\0")
            return set(ignored)
        return set()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return set()


def load_ignore_patterns(
    project_dir: str | Path,
    ignore_file: str | Path | None = None,
) -> "PathSpec":
    """Load ignore patterns from .tldrignore file.

    Args:
        project_dir: Root directory of the project

    Returns:
        PathSpec matcher for checking if files should be ignored
    """
    try:
        import pathspec
    except ImportError:
        return _NullSpec()

    project_path = Path(project_dir)
    if ignore_file is None:
        tldrignore_path = project_path / ".tldrignore"
    else:
        tldrignore_path = Path(ignore_file)

    patterns: list[str] = []

    if tldrignore_path.exists():
        content = tldrignore_path.read_text()
        patterns: list[str] = content.splitlines()
    else:
        # Use defaults if no .tldrignore exists
        patterns = list(DEFAULT_TEMPLATE.splitlines())

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


class IgnoreSpec:
    """Wrapper that combines .tldrignore + .gitignore checking.

    Provides a `match_file()` interface compatible with pathspec.PathSpec,
    but also checks .gitignore via batch subprocess calls for performance.
    """

    def __init__(
        self,
        project_dir: str | Path,
        use_gitignore: bool = True,
        cli_patterns: list[str] | None = None,
        ignore_file: str | Path | None = None,
        gitignore_root: str | Path | None = None,
    ):
        try:
            import pathspec
        except ImportError:
            pathspec = None

        self.project_path = Path(project_dir).resolve()
        self.use_gitignore = use_gitignore
        self.ignore_file = Path(ignore_file).resolve() if ignore_file is not None else None

        if use_gitignore:
            root_candidate = (
                Path(gitignore_root).resolve()
                if gitignore_root is not None
                else self.project_path
            )
            self._git_root = resolve_git_root(root_candidate)
            self._is_git = self._git_root is not None
        else:
            self._git_root = None
            self._is_git = False

        # Load base tldrignore patterns
        self._spec = load_ignore_patterns(self.project_path, ignore_file=self.ignore_file)

        # Add CLI --ignore patterns if provided
        if cli_patterns and pathspec is not None:
            # Combine existing patterns with CLI patterns using from_lines
            existing_lines = [str(p) for p in self._spec.patterns]
            self._spec = pathspec.PathSpec.from_lines(
                "gitwildmatch", existing_lines + cli_patterns
            )

        # Cache for batch gitignore results (populated lazily)
        self._gitignore_cache: set[str] | None = None
        self._pending_paths: list[str] = []

    def match_file(self, rel_path: str | Path) -> bool:
        """Check if a file should be ignored.

        Compatible with pathspec.PathSpec.match_file() interface.
        """
        # Ensure string for pattern matching
        rel_path_str = str(rel_path)

        # Check .tldrignore first
        has_negation = _has_negation_for_file(self._spec, rel_path_str)

        if has_negation:
            # .tldrignore has explicit opinion via negation
            return self._spec.match_file(rel_path_str)

        if self._spec.match_file(rel_path_str):
            # .tldrignore says ignore
            return True

        # .tldrignore has no opinion - check gitignore
        if self._is_git:
            return self._check_gitignore(rel_path_str)

        return False

    def _check_gitignore(self, rel_path: str) -> bool:
        """Check single file against gitignore (uses per-file call)."""
        # For single-file checks, fall back to per-file subprocess
        # Batch checking is used in filter_files() for better perf
        return is_gitignored(
            self.project_path / rel_path,
            self.project_path,
            git_root=self._git_root,
        )

    def preload_gitignore(self, paths: list[str]) -> None:
        """Batch-load gitignore status for multiple paths (performance optimization)."""
        if not self._is_git or not paths:
            return
        full_paths = [self.project_path / p for p in paths]
        self._gitignore_cache = batch_gitignored(
            full_paths,
            self.project_path,
            git_root=self._git_root,
        )

    def match_file_cached(self, rel_path: str) -> bool:
        """Check if file should be ignored, using preloaded cache if available."""
        # Check .tldrignore first
        has_negation = _has_negation_for_file(self._spec, rel_path)

        if has_negation:
            return self._spec.match_file(rel_path)

        if self._spec.match_file(rel_path):
            return True

        # Check gitignore cache
        if self._is_git and self._gitignore_cache is not None:
            return rel_path in self._gitignore_cache

        return False


def ensure_tldrignore(
    project_dir: str | Path,
    ignore_file: str | Path | None = None,
) -> tuple[bool, str]:
    """Ensure .tldrignore exists, creating with defaults if needed.

    Args:
        project_dir: Root directory of the project

    Returns:
        Tuple of (created: bool, message: str)
    """
    project_path = Path(project_dir)

    if not project_path.exists():
        return False, f"Project directory does not exist: {project_path}"

    if ignore_file is None:
        tldrignore_path = project_path / ".tldrignore"
    else:
        tldrignore_path = Path(ignore_file)

    if tldrignore_path.exists():
        return False, f".tldrignore already exists at {tldrignore_path}"

    # Create with default template
    tldrignore_path.write_text(DEFAULT_TEMPLATE)

    return (
        True,
        """Created .tldrignore with sensible defaults:
  - node_modules/, .venv/, __pycache__/
  - dist/, build/, *.egg-info/
  - Binary files (*.so, *.dll, *.whl)
  - Security files (.env, *.pem, *.key)

Review .tldrignore before indexing large codebases.
Edit to exclude vendor code, test fixtures, etc.""",
    )


def should_ignore(
    file_path: str | Path,
    project_dir: str | Path,
    spec: "PathSpec | None" = None,
    use_gitignore: bool = True,
    ignore_file: str | Path | None = None,
    gitignore_root: str | Path | None = None,
) -> bool:
    """Check if a file should be ignored.

    Precedence:
    1. .gitignore provides baseline (if in git repo)
    2. .tldrignore overrides - can add ignores OR un-ignore via ! patterns

    Args:
        file_path: Path to check (absolute or relative)
        project_dir: Root directory of the project
        spec: Optional pre-loaded PathSpec (for efficiency in loops)
        use_gitignore: Whether to also check .gitignore (default True)

    Returns:
        True if file should be ignored, False otherwise
    """
    if spec is None:
        spec = load_ignore_patterns(project_dir, ignore_file=ignore_file)

    project_path = Path(project_dir)
    file_path = Path(file_path)

    # Make path relative to project for matching
    try:
        rel_path = file_path.relative_to(project_path)
    except ValueError:
        # File is not under project_dir, use as-is
        rel_path = file_path

    rel_path_str = str(rel_path)

    # .tldrignore is the final authority - it can:
    # - Add ignores (positive patterns)
    # - Un-ignore gitignored files (! negation patterns)
    #
    # pathspec.match_file returns True if file matches a positive pattern
    # and wasn't subsequently un-matched by a negation pattern
    tldr_ignored = spec.match_file(rel_path_str)

    # Check if .tldrignore has an explicit opinion via negation
    # by checking if any negation pattern matches this file
    has_negation = _has_negation_for_file(spec, rel_path_str)

    if has_negation:
        # .tldrignore explicitly un-ignores this file - respect that
        return tldr_ignored

    if tldr_ignored:
        # .tldrignore says ignore
        return True

    # .tldrignore has no opinion - check gitignore as fallback
    if use_gitignore:
        git_root = (
            Path(gitignore_root).resolve()
            if gitignore_root is not None
            else resolve_git_root(project_path)
        )
        if git_root is not None:
            return is_gitignored(file_path, project_path, git_root=git_root)

    return False


def _has_negation_for_file(spec: "PathSpec", rel_path: str) -> bool:
    """Check if any negation pattern in the spec would match this file.

    This helps determine if .tldrignore has an explicit opinion about
    including a file (via ! pattern) vs simply not matching it.
    """
    for pattern in spec.patterns:
        # Check if this is a negation (include) pattern
        # pathspec uses 'include' attribute: True = negation (! pattern)
        if getattr(pattern, 'include', None) is True:
            # This is a negation pattern - check if it matches
            if pattern.match_file(rel_path):
                return True
    return False


def filter_files(
    files: list[Path],
    project_dir: str | Path,
    respect_ignore: bool = True,
    use_gitignore: bool = True,
    ignore_file: str | Path | None = None,
    gitignore_root: str | Path | None = None,
) -> list[Path]:
    """Filter a list of files, removing those matching ignore patterns.

    Checks both .tldrignore and .gitignore (if in a git repo).
    .tldrignore patterns take precedence over .gitignore.
    Uses batch gitignore checking for ~35x faster performance.

    Args:
        files: List of file paths to filter
        project_dir: Root directory of the project
        respect_ignore: If False, skip filtering (--no-ignore mode)
        use_gitignore: Whether to also check .gitignore (default True)

    Returns:
        Filtered list of files
    """
    if not respect_ignore:
        return files

    project_path = Path(project_dir)
    spec = load_ignore_patterns(project_dir, ignore_file=ignore_file)

    # First pass: filter by .tldrignore patterns
    # Also track files that need gitignore check (not matched by tldrignore)
    tldr_passed: list[Path] = []
    for f in files:
        try:
            rel_path = str(f.relative_to(project_path))
        except ValueError:
            rel_path = str(f)

        # Check if .tldrignore has explicit negation (!) for this file
        has_negation = _has_negation_for_file(spec, rel_path)

        if has_negation:
            # .tldrignore explicitly includes/excludes - use its decision
            if not spec.match_file(rel_path):
                tldr_passed.append(f)
        elif spec.match_file(rel_path):
            # .tldrignore says ignore
            continue
        else:
            # .tldrignore has no opinion - might need gitignore check
            tldr_passed.append(f)

    # Second pass: batch check gitignore for files that passed tldrignore
    if use_gitignore and tldr_passed:
        git_root = (
            Path(gitignore_root).resolve()
            if gitignore_root is not None
            else resolve_git_root(project_path)
        )
        if git_root is not None:
            gitignored = batch_gitignored(
                tldr_passed,
                project_path,
                git_root=git_root,
            )
        else:
            gitignored = set()
        return [f for f in tldr_passed
                if str(f.relative_to(project_path) if f.is_absolute() else f) not in gitignored]

    return tldr_passed


def compute_ignore_hash(ignore_file: str | Path | None) -> str:
    """Compute a stable hash of ignore file content (or defaults if missing)."""
    import hashlib

    content = None
    if ignore_file is not None:
        try:
            content = Path(ignore_file).read_text()
        except (OSError, IOError):
            content = None

    if content is None:
        content = DEFAULT_TEMPLATE

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return digest
