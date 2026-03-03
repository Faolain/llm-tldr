"""
Semantic search for code using 5-layer embeddings.

Embeds functions/methods using all 5 TLDR analysis layers:
- L1: Signature + docstring
- L2: Top callers + callees (from call graph)
- L3: Control flow summary
- L4: Data flow summary
- L5: Dependencies

Uses BAAI/bge-large-en-v1.5 for embeddings (1024 dimensions)
and FAISS for fast vector similarity search.
"""

import json
import hashlib
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import List, Optional, Dict, Any

logger = logging.getLogger("tldr.semantic")

ALL_LANGUAGES = ["python", "typescript", "javascript", "go", "rust", "java", "c", "cpp", "ruby", "php", "kotlin", "swift", "csharp", "scala", "lua", "luau", "elixir"]

# Avoid OpenMP runtime conflicts between torch and faiss on macOS.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# Lazy imports for heavy dependencies
_model = None
_model_name = None  # Track which model is loaded
_model_device = None  # Track device used for cached model

# Supported models with approximate download sizes
SUPPORTED_MODELS = {
    "bge-large-en-v1.5": {
        "hf_name": "BAAI/bge-large-en-v1.5",
        "size": "1.3GB",
        "dimension": 1024,
        "description": "High quality, recommended for production",
    },
    "all-MiniLM-L6-v2": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "size": "80MB",
        "dimension": 384,
        "description": "Lightweight, good for testing",
    },
}

DEFAULT_MODEL = "bge-large-en-v1.5"

LANE3_REFERENCE_BUDGET_TOKENS = 2000
LANE3_MAX_EFFECTIVE_K_MULTIPLIER = 5
LANE4_SCHEMA_VERSION = 1
LANE4_FEATURE_SET_ID = "feature.compound-semantic-impact.v1"
LANE4_IMPACT_DEFAULT_DEPTH = 3
LANE4_IMPACT_DEFAULT_LIMIT = 3
LANE5_SCHEMA_VERSION = 1
LANE5_FEATURE_SET_ID = "feature.navigate-cluster.v1"
LANE5_DEFAULT_CLUSTER_COUNT = 5
LANE5_DEFAULT_CLUSTER_MIN_SIZE = 1
LANE5_DEFAULT_CLUSTER_MAX_MEMBERS = 5
LANE5_CLUSTER_LABEL_MODES = {"auto", "file", "symbol"}
LANE5_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "app",
    "bin",
    "c",
    "cc",
    "cpp",
    "file",
    "for",
    "go",
    "h",
    "hpp",
    "in",
    "index",
    "init",
    "is",
    "java",
    "js",
    "lib",
    "main",
    "of",
    "on",
    "or",
    "pkg",
    "py",
    "rs",
    "src",
    "test",
    "to",
    "ts",
    "tsx",
}
LANE4_CALL_GRAPH_LANGUAGES = {"python", "typescript", "go", "rust", "java", "c", "php"}
LANE4_EXTENSION_TO_CALL_GRAPH_LANGUAGE = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".php": "php",
}

# Project root markers - files that indicate a project root
PROJECT_ROOT_MARKERS = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", ".tldr"]


def _find_project_root(start_path: Path) -> Path:
    """Find project root by walking up from start_path.

    Looks for common project markers (.git, pyproject.toml, etc.).
    Also respects CLAUDE_PROJECT_DIR environment variable.

    Args:
        start_path: Path to start searching from.

    Returns:
        Project root path, or start_path if no markers found.
    """
    # Check environment variable first
    env_root = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_root:
        env_path = Path(env_root).resolve()
        if env_path.exists():
            return env_path

    # Walk up looking for project markers
    current = start_path.resolve()
    while current != current.parent:
        for marker in PROJECT_ROOT_MARKERS:
            if (current / marker).exists():
                return current
        current = current.parent

    # No markers found - use start_path
    return start_path.resolve()


@dataclass
class EmbeddingUnit:
    """A code unit (function/method/class) for embedding.

    Contains information from all 5 TLDR layers:
    - L1: signature, docstring
    - L2: calls, called_by
    - L3: cfg_summary
    - L4: dfg_summary
    - L5: dependencies
    """
    name: str
    qualified_name: str
    file: str
    line: int
    language: str
    unit_type: str  # "function" | "method" | "class"
    signature: str
    docstring: str
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    cfg_summary: str = ""
    dfg_summary: str = ""
    dependencies: str = ""
    code_preview: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file": self.file,
            "line": self.line,
            "language": self.language,
            "unit_type": self.unit_type,
            "signature": self.signature,
            "docstring": self.docstring,
            "calls": self.calls,
            "called_by": self.called_by,
            "cfg_summary": self.cfg_summary,
            "dfg_summary": self.dfg_summary,
            "dependencies": self.dependencies,
            "code_preview": self.code_preview,
        }


MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Legacy, use SUPPORTED_MODELS


def _canonical_model_id(model_name: Optional[str]) -> Optional[str]:
    if model_name is None:
        return None
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]["hf_name"]
    for info in SUPPORTED_MODELS.values():
        if model_name == info["hf_name"]:
            return info["hf_name"]
    return model_name


def _model_dimension(model_name: Optional[str]) -> Optional[int]:
    canonical = _canonical_model_id(model_name)
    if canonical is None:
        return None
    for info in SUPPORTED_MODELS.values():
        if info["hf_name"] == canonical:
            return info["dimension"]
    return None


def _model_exists_locally(hf_name: str) -> bool:
    """Check if a model is already downloaded locally."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # Check if model config exists in cache
        result = try_to_load_from_cache(hf_name, "config.json")
        return result is not None
    except Exception:
        return False


def _confirm_download(model_key: str) -> bool:
    """Prompt user to confirm model download. Returns True if confirmed."""
    model_info = SUPPORTED_MODELS.get(model_key, {})
    size = model_info.get("size", "unknown size")
    hf_name = model_info.get("hf_name", model_key)

    # Skip prompt if TLDR_AUTO_DOWNLOAD is set or not a TTY
    if os.environ.get("TLDR_AUTO_DOWNLOAD") == "1":
        return True
    if not sys.stdin.isatty():
        # Non-interactive: warn but proceed
        print(f"⚠️  Downloading {hf_name} ({size})...", file=sys.stderr)
        return True

    print(f"\n⚠️  Semantic search requires embedding model: {hf_name}", file=sys.stderr)
    print(f"   Download size: {size}", file=sys.stderr)
    print("   (Set TLDR_AUTO_DOWNLOAD=1 to skip this prompt)\n", file=sys.stderr)

    try:
        response = input("Continue with download? [Y/n] ").strip().lower()
        return response in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _get_device() -> str:
    """Determine inference device, defaulting to CPU on macOS for stability."""
    env_device = os.environ.get("TLDR_DEVICE")
    if env_device:
        return env_device

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    # If MPS fallback enabled, prefer CPU for stability on macOS
    if sys.platform == "darwin" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        return "cpu"

    return "cpu"


def get_model(model_name: Optional[str] = None, device: Optional[str] = None):
    """Lazy-load the embedding model (cached).

    Args:
        model_name: Model key from SUPPORTED_MODELS, or None for default.
                   Can also be a full HuggingFace model name.

    Returns:
        SentenceTransformer model instance.

    Raises:
        ValueError: If model not found or user declines download.
    """
    global _model, _model_name, _model_device

    # Resolve model name
    if model_name is None:
        model_name = DEFAULT_MODEL

    # Get HuggingFace name
    if model_name in SUPPORTED_MODELS:
        hf_name = SUPPORTED_MODELS[model_name]["hf_name"]
    else:
        # Allow arbitrary HuggingFace model names
        hf_name = model_name

    if device is None:
        device = _get_device()

    # Return cached model if same
    if _model is not None and _model_name == hf_name and _model_device == device:
        return _model

    # Check if model needs downloading
    if not _model_exists_locally(hf_name):
        model_key = model_name if model_name in SUPPORTED_MODELS else None
        if model_key and not _confirm_download(model_key):
            raise ValueError("Model download declined. Use --model to choose a smaller model.")

    logger.info("Loading model %s on device: %s", hf_name, device)
    from sentence_transformers import SentenceTransformer
    if device:
        _model = SentenceTransformer(hf_name, device=device)
    else:
        _model = SentenceTransformer(hf_name)
    _model_name = hf_name
    _model_device = device
    return _model


def build_embedding_text(unit: EmbeddingUnit) -> str:
    """Build rich text for embedding from all 5 layers.

    Creates a single text string containing information from all
    analysis layers, suitable for embedding with a language model.

    Args:
        unit: The EmbeddingUnit containing code analysis.

    Returns:
        A text string combining all layer information.
    """
    parts = []

    # L1: Signature + docstring
    if unit.signature:
        parts.append(f"Signature: {unit.signature}")
    if unit.docstring:
        parts.append(f"Description: {unit.docstring}")

    # L2: Call graph (forward - callees)
    if unit.calls:
        calls_str = ", ".join(unit.calls[:5])  # Top 5
        parts.append(f"Calls: {calls_str}")

    # L2: Call graph (backward - callers)
    if unit.called_by:
        callers_str = ", ".join(unit.called_by[:5])  # Top 5
        parts.append(f"Called by: {callers_str}")

    # L3: Control flow summary
    if unit.cfg_summary:
        parts.append(f"Control flow: {unit.cfg_summary}")

    # L4: Data flow summary
    if unit.dfg_summary:
        parts.append(f"Data flow: {unit.dfg_summary}")

    # L5: Dependencies
    if unit.dependencies:
        parts.append(f"Dependencies: {unit.dependencies}")

    # Code preview (first 10 lines of function body)
    if unit.code_preview:
        parts.append(f"Code:\n{unit.code_preview}")

    # Add name and type for context
    type_str = unit.unit_type if unit.unit_type else "function"
    parts.insert(0, f"{type_str.capitalize()}: {unit.name}")

    return "\n".join(parts)


def compute_embedding(text: str, model_name: Optional[str] = None, device: Optional[str] = None):
    """Compute embedding vector for text.

    Args:
        text: The text to embed.
        model_name: Model to use (from SUPPORTED_MODELS or HF name).

    Returns:
        numpy array with L2-normalized embedding.
    """
    import numpy as np

    model = get_model(model_name, device=device)

    # BGE models work best with instruction prefix for queries
    # For document embedding, we use text directly
    embedding = model.encode(text, normalize_embeddings=True)

    return np.array(embedding, dtype=np.float32)


def extract_units_from_project(
    project_path: str,
    lang: str = "python",
    respect_ignore: bool = True,
    ignore_spec=None,
    progress_callback=None,
    workspace_root: Path | None = None,
) -> List[EmbeddingUnit]:
    """Extract all functions/methods/classes from a project.

    Uses existing TLDR APIs:
    - tldr.api.get_code_structure() for L1 (signatures)
    - tldr.cross_file_calls for L2 (call graph)
    - CFG/DFG extractors for L3/L4 summaries
    - tldr.api.get_imports for L5 (dependencies)

    Args:
        project_path: Path to project root.
        lang: Programming language ("python", "typescript", "go", "rust").
        respect_ignore: If True, respect .tldrignore patterns (default True).

    Returns:
        List of EmbeddingUnit objects with enriched metadata.
    """
    from tldr.api import get_code_structure, build_project_call_graph
    from tldr.tldrignore import IgnoreSpec

    project = Path(project_path).resolve()
    units = []

    if respect_ignore and ignore_spec is None:
        ignore_spec = IgnoreSpec(project)

    # Get code structure (L1) - use high limit for semantic index
    structure = get_code_structure(
        str(project),
        language=lang,
        max_results=100000,
        ignore_spec=ignore_spec,
    )

    # Build call graph (L2)
    try:
        if workspace_root is not None:
            call_graph = build_project_call_graph(
                str(project),
                language=lang,
                ignore_spec=ignore_spec,
                workspace_root=workspace_root,
            )
        else:
            call_graph = build_project_call_graph(
                str(project),
                language=lang,
                ignore_spec=ignore_spec,
            )

        # Build call/called_by maps
        calls_map = {}  # func -> [called functions]
        called_by_map = {}  # func -> [calling functions]

        for edge in call_graph.edges:
            src_file, src_func, dst_file, dst_func = edge

            # Forward: src calls dst
            if src_func not in calls_map:
                calls_map[src_func] = []
            calls_map[src_func].append(dst_func)

            # Backward: dst is called by src
            if dst_func not in called_by_map:
                called_by_map[dst_func] = []
            called_by_map[dst_func].append(src_func)
    except Exception:
        # Call graph may not be available for all projects
        calls_map = {}
        called_by_map = {}

    # Process files in parallel for better performance
    files = structure.get("files", [])
    max_workers = int(os.environ.get("TLDR_MAX_WORKERS", os.cpu_count() or 4))

    # Use parallel processing if we have multiple files
    if len(files) > 1 and max_workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _process_file_for_extraction,
                        file_info,
                        str(project),
                        lang,
                        calls_map,
                        called_by_map,
                    ): file_info
                    for file_info in files
                }

                for future in as_completed(futures):
                    file_info = futures[future]
                    try:
                        file_units = future.result(timeout=60)
                        units.extend(file_units)
                        if progress_callback:
                            progress_callback(file_info.get('path', 'unknown'), len(units), len(files))
                    except Exception as e:
                        logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {e}")

        except Exception as e:
            logger.warning(f"Parallel extraction failed: {e}, falling back to sequential")
            for file_info in files:
                try:
                    file_units = _process_file_for_extraction(
                        file_info, str(project), lang, calls_map, called_by_map
                    )
                    units.extend(file_units)
                    if progress_callback:
                        progress_callback(file_info.get('path', 'unknown'), len(units), len(files))
                except Exception as fe:
                    logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {fe}")
    else:
        for file_info in files:
            try:
                file_units = _process_file_for_extraction(
                    file_info, str(project), lang, calls_map, called_by_map
                )
                units.extend(file_units)
                if progress_callback:
                    progress_callback(file_info.get('path', 'unknown'), len(units), len(files))
            except Exception as e:
                logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {e}")

    return units


def _parse_file_ast(file_path: Path, lang: str) -> dict:
    """Parse file AST to extract line numbers and code previews.

    Returns:
        Dict with structure:
        {
            "functions": {func_name: {"line": int, "code_preview": str}},
            "classes": {class_name: {"line": int}},
            "methods": {"ClassName.method": {"line": int, "code_preview": str}}
        }
    """
    result = {"functions": {}, "classes": {}, "methods": {}}

    if not file_path.exists():
        return result

    try:
        content = file_path.read_text()
        lines = content.split('\n')

        if lang == "python":
            import ast
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Check if this is a method (inside a class)
                    parent_class = None
                    for potential_parent in ast.walk(tree):
                        if isinstance(potential_parent, ast.ClassDef):
                            if node in ast.walk(potential_parent) and node.name != potential_parent.name:
                                # Check if node is a direct child method
                                for item in potential_parent.body:
                                    if item is node:
                                        parent_class = potential_parent.name
                                        break

                    # Extract code preview (first 10 lines of body)
                    start_line = node.lineno
                    end_line = getattr(node, 'end_lineno', start_line + 10)
                    body_lines = lines[start_line - 1:min(end_line, start_line + 10) - 1]
                    code_preview = '\n'.join(body_lines[:10])

                    if parent_class:
                        result["methods"][f"{parent_class}.{node.name}"] = {
                            "line": node.lineno,
                            "code_preview": code_preview
                        }
                    else:
                        result["functions"][node.name] = {
                            "line": node.lineno,
                            "code_preview": code_preview
                        }

                elif isinstance(node, ast.ClassDef):
                    result["classes"][node.name] = {"line": node.lineno}

    except Exception:
        # Return empty result on any parsing error
        pass

    return result


def _get_file_dependencies(file_path: Path, lang: str) -> str:
    """Get file-level import dependencies as a string."""
    if not file_path.exists():
        return ""

    try:
        from tldr.api import get_imports
        imports = get_imports(str(file_path), language=lang)

        # Extract module names (limit to first 5 for brevity)
        modules = []
        for imp in imports[:5]:
            module = imp.get("module", "")
            if module:
                modules.append(module)

        return ", ".join(modules) if modules else ""
    except Exception:
        return ""


def _get_cfg_summary(file_path: Path, func_name: str, lang: str) -> str:
    """Get CFG summary (complexity, block count) for a function."""
    if not file_path.exists():
        return ""

    try:
        content = file_path.read_text()

        # Import the appropriate CFG extractor based on language
        from tldr import cfg_extractor

        extractor_map = {
            "python": cfg_extractor.extract_python_cfg,
            "typescript": cfg_extractor.extract_typescript_cfg,
            "javascript": cfg_extractor.extract_typescript_cfg,  # JS uses TS extractor
            "go": cfg_extractor.extract_go_cfg,
            "rust": cfg_extractor.extract_rust_cfg,
            "java": cfg_extractor.extract_java_cfg,
            "c": cfg_extractor.extract_c_cfg,
            "cpp": cfg_extractor.extract_cpp_cfg,
            "php": cfg_extractor.extract_php_cfg,
            "ruby": cfg_extractor.extract_ruby_cfg,
            "swift": cfg_extractor.extract_swift_cfg,
            "csharp": cfg_extractor.extract_csharp_cfg,
            "kotlin": cfg_extractor.extract_kotlin_cfg,
            "scala": cfg_extractor.extract_scala_cfg,
            "lua": cfg_extractor.extract_lua_cfg,
            "luau": cfg_extractor.extract_luau_cfg,
            "elixir": cfg_extractor.extract_elixir_cfg,
        }

        extractor = extractor_map.get(lang)
        if extractor:
            cfg = extractor(content, func_name)
            return f"complexity:{cfg.cyclomatic_complexity}, blocks:{len(cfg.blocks)}"
    except Exception:
        pass

    return ""


def _get_dfg_summary(file_path: Path, func_name: str, lang: str) -> str:
    """Get DFG summary (variable count, def-use chains) for a function."""
    if not file_path.exists():
        return ""

    try:
        content = file_path.read_text()

        # Import the appropriate DFG extractor based on language
        from tldr import dfg_extractor

        extractor_map = {
            "python": dfg_extractor.extract_python_dfg,
            "typescript": dfg_extractor.extract_typescript_dfg,
            "javascript": dfg_extractor.extract_typescript_dfg,  # JS uses TS extractor
            "go": dfg_extractor.extract_go_dfg,
            "rust": dfg_extractor.extract_rust_dfg,
            "java": dfg_extractor.extract_java_dfg,
            "c": dfg_extractor.extract_c_dfg,
            "cpp": dfg_extractor.extract_cpp_dfg,
            "php": dfg_extractor.extract_php_dfg,
            "ruby": dfg_extractor.extract_ruby_dfg,
            "swift": dfg_extractor.extract_swift_dfg,
            "csharp": dfg_extractor.extract_csharp_dfg,
            "kotlin": dfg_extractor.extract_kotlin_dfg,
            "scala": dfg_extractor.extract_scala_dfg,
            "lua": dfg_extractor.extract_lua_dfg,
            "luau": dfg_extractor.extract_luau_dfg,
            "elixir": dfg_extractor.extract_elixir_dfg,
        }

        extractor = extractor_map.get(lang)
        if extractor:
            dfg = extractor(content, func_name)

            # Count unique variables and def-use chains
            var_names = set()
            for ref in dfg.var_refs:
                var_names.add(ref.name)

            return f"vars:{len(var_names)}, def-use chains:{len(dfg.dataflow_edges)}"
    except Exception:
        pass

    return ""


def _get_function_signature(file_path: Path, func_name: str, lang: str) -> Optional[str]:
    """Extract function signature from file."""
    if not file_path.exists():
        return None

    try:
        content = file_path.read_text()

        if lang == "python":
            import ast
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    # Build signature from args
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)

                    returns = ""
                    if node.returns:
                        returns = f" -> {ast.unparse(node.returns)}"

                    return f"def {func_name}({', '.join(args)}){returns}"


        # For other languages, return simple signature
        return f"function {func_name}(...)"

    except Exception:
        return None


def _get_function_docstring(file_path: Path, func_name: str, lang: str) -> Optional[str]:
    """Extract function docstring from file."""
    if not file_path.exists():
        return None

    try:
        content = file_path.read_text()

        if lang == "python":
            import ast
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    return ast.get_docstring(node)

        return None

    except Exception:
        return None


def _process_file_for_extraction(
    file_info: Dict[str, Any],
    project_path: str,
    lang: str,
    calls_map: Dict[str, List[str]],
    called_by_map: Dict[str, List[str]],
) -> List[EmbeddingUnit]:
    """Process a single file and extract all units. Top-level for pickling.

    This function reads the file ONCE and extracts all information in a single pass,
    avoiding the O(n*m) file read issue where n=files and m=functions.

    Args:
        file_info: Dict with 'path', 'functions', 'classes' from get_code_structure.
        project_path: Absolute path to project root.
        lang: Programming language.
        calls_map: Map of function name -> list of called functions.
        called_by_map: Map of function name -> list of calling functions.

    Returns:
        List of EmbeddingUnit objects for this file.
    """
    units = []
    project = Path(project_path)
    file_path = file_info.get("path", "")
    full_path = project / file_path

    if not full_path.exists():
        return units

    try:
        # Read file content ONCE
        content = full_path.read_text()
        lines = content.split('\n')
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return units

    # Parse AST once for all function info
    ast_info = {"functions": {}, "classes": {}, "methods": {}}
    all_signatures = {}
    all_docstrings = {}

    if lang == "python":
        try:
            import ast
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check if this is a method (inside a class)
                    parent_class = None
                    for potential_parent in ast.walk(tree):
                        if isinstance(potential_parent, ast.ClassDef):
                            for item in potential_parent.body:
                                if item is node:
                                    parent_class = potential_parent.name
                                    break

                    # Extract code preview (first 10 lines of body)
                    start_line = node.lineno
                    end_line = getattr(node, 'end_lineno', start_line + 10)
                    body_lines = lines[start_line - 1:min(end_line, start_line + 10) - 1]
                    code_preview = '\n'.join(body_lines[:10])

                    # Build signature
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)
                    returns = ""
                    if node.returns:
                        returns = f" -> {ast.unparse(node.returns)}"
                    signature = f"def {node.name}({', '.join(args)}){returns}"

                    # Get docstring
                    docstring = ast.get_docstring(node) or ""

                    if parent_class:
                        key = f"{parent_class}.{node.name}"
                        ast_info["methods"][key] = {
                            "line": node.lineno,
                            "code_preview": code_preview
                        }
                        all_signatures[key] = signature
                        all_docstrings[key] = docstring
                    else:
                        ast_info["functions"][node.name] = {
                            "line": node.lineno,
                            "code_preview": code_preview
                        }
                        all_signatures[node.name] = signature
                        all_docstrings[node.name] = docstring

                elif isinstance(node, ast.ClassDef):
                    ast_info["classes"][node.name] = {"line": node.lineno}

        except Exception as e:
            logger.debug(f"AST parse failed for {file_path}: {e}")

    # Get dependencies (imports) - single call
    dependencies = ""
    try:
        from tldr.api import get_imports
        imports = get_imports(str(full_path), language=lang)
        modules = [imp.get("module", "") for imp in imports[:5] if imp.get("module")]
        dependencies = ", ".join(modules)
    except Exception:
        pass

    # Pre-compute CFG/DFG for all functions at once
    cfg_cache = {}
    dfg_cache = {}

    # Language-to-extractor mapping for CFG/DFG analysis
    def _get_extractors(language: str):
        """Return (cfg_extractor, dfg_extractor) for the given language."""
        if language == "python":
            from tldr.cfg_extractor import extract_python_cfg
            from tldr.dfg_extractor import extract_python_dfg
            return extract_python_cfg, extract_python_dfg
        elif language in ("typescript", "javascript"):
            from tldr.cfg_extractor import extract_typescript_cfg
            from tldr.dfg_extractor import extract_typescript_dfg
            return extract_typescript_cfg, extract_typescript_dfg
        return None, None

    cfg_extractor, dfg_extractor = _get_extractors(lang)

    if cfg_extractor and dfg_extractor:
        # Get all function names we need to process
        all_func_names = list(file_info.get("functions", []))
        for class_info in file_info.get("classes", []):
            if isinstance(class_info, dict):
                all_func_names.extend(class_info.get("methods", []))

        for func_name in all_func_names:
            try:
                cfg = cfg_extractor(content, func_name)
                cfg_cache[func_name] = f"complexity:{cfg.cyclomatic_complexity}, blocks:{len(cfg.blocks)}"
            except Exception:
                cfg_cache[func_name] = ""

            try:
                dfg = dfg_extractor(content, func_name)
                var_names = {ref.name for ref in dfg.var_refs}
                dfg_cache[func_name] = f"vars:{len(var_names)}, def-use chains:{len(dfg.dataflow_edges)}"
            except Exception:
                dfg_cache[func_name] = ""

    # Process functions
    for func_name in file_info.get("functions", []):
        func_info = ast_info.get("functions", {}).get(func_name, {})
        unit = EmbeddingUnit(
            name=func_name,
            qualified_name=f"{file_path.replace('/', '.')}.{func_name}",
            file=file_path,
            line=func_info.get("line", 1),
            language=lang,
            unit_type="function",
            signature=all_signatures.get(func_name, f"def {func_name}(...)"),
            docstring=all_docstrings.get(func_name, ""),
            calls=calls_map.get(func_name, [])[:5],
            called_by=called_by_map.get(func_name, [])[:5],
            cfg_summary=cfg_cache.get(func_name, ""),
            dfg_summary=dfg_cache.get(func_name, ""),
            dependencies=dependencies,
            code_preview=func_info.get("code_preview", ""),
        )
        units.append(unit)

    # Process classes
    for class_info in file_info.get("classes", []):
        if isinstance(class_info, dict):
            class_name = class_info.get("name", "")
            methods = class_info.get("methods", [])
        else:
            class_name = class_info
            methods = []

        class_line = ast_info.get("classes", {}).get(class_name, {}).get("line", 1)

        # Add class itself
        unit = EmbeddingUnit(
            name=class_name,
            qualified_name=f"{file_path.replace('/', '.')}.{class_name}",
            file=file_path,
            line=class_line,
            language=lang,
            unit_type="class",
            signature=f"class {class_name}",
            docstring="",
            calls=[],
            called_by=[],
            cfg_summary="",
            dfg_summary="",
            dependencies=dependencies,
            code_preview="",
        )
        units.append(unit)

        # Add methods
        for method in methods:
            method_key = f"{class_name}.{method}"
            method_info = ast_info.get("methods", {}).get(method_key, {})

            unit = EmbeddingUnit(
                name=method,
                qualified_name=f"{file_path.replace('/', '.')}.{method_key}",
                file=file_path,
                line=method_info.get("line", 1),
                language=lang,
                unit_type="method",
                signature=all_signatures.get(method_key, f"def {method}(self, ...)"),
                docstring=all_docstrings.get(method_key, ""),
                calls=calls_map.get(method, [])[:5],
                called_by=called_by_map.get(method, [])[:5],
                cfg_summary=cfg_cache.get(method, ""),
                dfg_summary=dfg_cache.get(method, ""),
                dependencies=dependencies,
                code_preview=method_info.get("code_preview", ""),
            )
            units.append(unit)

    return units


def _get_progress_console():
    """Get rich Console if available and TTY, else None."""
    if not sys.stdout.isatty():
        return None
    if os.environ.get("NO_PROGRESS") or os.environ.get("CI"):
        return None
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def _detect_project_languages(
    project_path: Path,
    respect_ignore: bool = True,
    ignore_spec=None,
) -> List[str]:
    """Scan project files to detect present languages."""
    from tldr.tldrignore import IgnoreSpec

    # Extension map (copied from cli.py to avoid circular import)
    EXTENSION_TO_LANGUAGE = {
        '.java': 'java',
        '.py': 'python',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.go': 'go',
        '.rs': 'rust',
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.hpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.hh': 'cpp',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.cs': 'csharp',
        '.kt': 'kotlin',
        '.kts': 'kotlin',
        '.scala': 'scala',
        '.sc': 'scala',
        '.lua': 'lua',
        '.luau': 'luau',
        '.ex': 'elixir',
        '.exs': 'elixir',
    }

    found_languages = set()
    spec = ignore_spec
    if respect_ignore and spec is None:
        spec = IgnoreSpec(project_path)

    for root, dirs, files in os.walk(project_path):
        # Prune common heavy dirs immediately for speed
        dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '.tldr', 'venv', '.venv', '__pycache__', '.idea', '.vscode', 'env', '.env', 'vendor', 'deps', '_build', 'cover'}]

        for file in files:
             file_path = Path(root) / file

             # Check ignore patterns
             if respect_ignore and spec:
                 try:
                     rel_path = file_path.relative_to(project_path)
                 except ValueError:
                     rel_path = file_path
                 if spec.match_file(str(rel_path)):
                     continue

             ext = file_path.suffix.lower()
             if ext in EXTENSION_TO_LANGUAGE:
                 found_languages.add(EXTENSION_TO_LANGUAGE[ext])

    # Return sorted list intersect with ALL_LANGUAGES to ensure validity
    return sorted(list(found_languages & set(ALL_LANGUAGES)))


def build_semantic_index(
    project_path: str,
    lang: str = "python",
    model: Optional[str] = None,
    device: Optional[str] = None,
    show_progress: bool = True,
    respect_ignore: bool = True,
    ignore_spec=None,
    index_paths=None,
    index_config=None,
    rebuild: bool = False,
) -> int:
    """Build and save FAISS index + metadata for a project.

    Creates:
    - .tldr/cache/semantic/index.faiss - Vector index
    - .tldr/cache/semantic/metadata.json - Unit metadata

    Args:
        project_path: Path to project root.
        lang: Programming language.
        model: Model name from SUPPORTED_MODELS or HuggingFace name.
        show_progress: Show progress spinner (default: True).
        respect_ignore: If True, respect .tldrignore patterns (default True).

    Returns:
        Number of indexed units.
    """
    import numpy as np
    from tldr.tldrignore import ensure_tldrignore, IgnoreSpec

    console = _get_progress_console() if show_progress else None

    # Resolve paths: scan_path is where to look for code, project_root is where to store cache
    scan_path = Path(project_path).resolve()
    if index_paths is None:
        project_root = _find_project_root(scan_path)
    else:
        if index_config is None:
            raise ValueError("index_config is required when index_paths is provided")
        project_root = scan_path

    if index_paths is None:
        # Ensure .tldrignore exists at project root (create with defaults if not)
        created, message = ensure_tldrignore(project_root)
        if created and console:
            console.print(f"[yellow]{message}[/yellow]")
    else:
        # Ensure index-scoped ignore file exists (only within cache_root)
        ignore_path = index_paths.ignore_file
        if not ignore_path.exists():
            try:
                ignore_path.resolve().relative_to(index_config.cache_root.resolve())
                created, message = ensure_tldrignore(
                    index_paths.index_dir, ignore_file=ignore_path
                )
                if created and console:
                    console.print(f"[yellow]{message}[/yellow]")
            except ValueError:
                raise ValueError(
                    f"Ignore file does not exist: {ignore_path}. Create it manually or choose a path under cache-root."
                )

    # Resolve model name early to get HF name for metadata
    model_key = model if model else DEFAULT_MODEL
    hf_name = _canonical_model_id(model_key) or model_key

    # Always store cache at project root, not scan path (legacy)
    cache_dir = project_root / ".tldr" / "cache" / "semantic"
    if index_paths is not None:
        cache_dir = index_paths.semantic_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # If an index already exists, enforce model compatibility unless rebuild is requested
    metadata_file = cache_dir / "metadata.json"
    if metadata_file.exists() and not rebuild:
        try:
            existing = json.loads(metadata_file.read_text())
            existing_model = _canonical_model_id(existing.get("model"))
            if existing_model and existing_model != _canonical_model_id(hf_name):
                raise ValueError(
                    "Semantic index model mismatch. Use --rebuild to overwrite."
                )
        except (json.JSONDecodeError, OSError):
            pass

    # Extract all units (respecting .tldrignore) - scan from scan_path, not project_root
    workspace_root = None
    if index_config is not None:
        try:
            scan_path.resolve().relative_to(index_config.cache_root.resolve())
            workspace_root = index_config.cache_root
        except ValueError:
            workspace_root = None

    if respect_ignore and ignore_spec is None:
        if index_config is not None:
            ignore_spec = IgnoreSpec(
                scan_path,
                use_gitignore=index_config.use_gitignore,
                cli_patterns=list(index_config.cli_patterns or ()),
                ignore_file=index_config.ignore_file,
                gitignore_root=index_config.gitignore_root,
            )
        else:
            ignore_spec = IgnoreSpec(
                scan_path,
                use_gitignore=True,
                cli_patterns=None,
            )

    if console:
        with console.status("[bold green]Extracting code units...") as status:
            def update_progress(file_path, units_count, total_files):
                short_path = file_path if len(file_path) < 50 else "..." + file_path[-47:]
                status.update(f"[bold green]Processing {short_path}... ({units_count} units)")

            if lang == "all":
                status.update("[bold green]Scanning project languages...")
                target_languages = _detect_project_languages(
                    scan_path,
                    respect_ignore=respect_ignore,
                    ignore_spec=ignore_spec,
                )
                if not target_languages:
                    console.print("[yellow]No supported languages detected in project[/yellow]")
                    return 0
                if console:
                    console.print(f"[dim]Detected languages: {', '.join(target_languages)}[/dim]")

                units = []
                for lang_name in target_languages:
                    status.update(f"[bold green]Extracting {lang_name} code units...")
                    units.extend(
                        extract_units_from_project(
                            str(scan_path),
                            lang=lang_name,
                            respect_ignore=respect_ignore,
                            ignore_spec=ignore_spec,
                            progress_callback=update_progress,
                            workspace_root=workspace_root,
                        )
                    )
            else:
                units = extract_units_from_project(
                    str(scan_path),
                    lang=lang,
                    respect_ignore=respect_ignore,
                    ignore_spec=ignore_spec,
                    progress_callback=update_progress,
                    workspace_root=workspace_root,
                )
            status.update(f"[bold green]Extracted {len(units)} code units")
    else:
        if lang == "all":
            target_languages = _detect_project_languages(
                scan_path,
                respect_ignore=respect_ignore,
                ignore_spec=ignore_spec,
            )
            if not target_languages:
                return 0
            units = []
            for lang_name in target_languages:
                units.extend(
                    extract_units_from_project(
                        str(scan_path),
                        lang=lang_name,
                        respect_ignore=respect_ignore,
                        ignore_spec=ignore_spec,
                        workspace_root=workspace_root,
                    )
                )
        else:
            units = extract_units_from_project(
                str(scan_path),
                lang=lang,
                respect_ignore=respect_ignore,
                ignore_spec=ignore_spec,
                workspace_root=workspace_root,
            )

    if not units:
        return 0

    BATCH_SIZE = 64
    num_units = len(units)
    texts = [build_embedding_text(unit) for unit in units]

    if console:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Computing embeddings...", total=num_units)

            model_obj = get_model(model, device=device)
            all_embeddings = []

            for i in range(0, num_units, BATCH_SIZE):
                chunk_end = min(i + BATCH_SIZE, num_units)
                chunk_texts = texts[i:chunk_end]

                current_unit = units[i]
                short_path = current_unit.file if len(current_unit.file) < 40 else "..." + current_unit.file[-37:]
                progress.update(task, description=f"[bold green]Embedding {short_path}::{current_unit.name}")

                result = model_obj.encode(
                    chunk_texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                all_embeddings.extend(np.array(result, dtype=np.float32))

                progress.update(task, completed=chunk_end)

            embeddings_matrix = np.vstack(all_embeddings)
    else:
        model_obj = get_model(model, device=device)
        result = model_obj.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True
        )
        embeddings_matrix = np.array(result, dtype=np.float32)

    import faiss
    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)

    # Save index
    index_file = cache_dir / "index.faiss"
    faiss.write_index(index, str(index_file))

    # Save metadata with actual model used
    metadata = {
        "units": [u.to_dict() for u in units],
        "model": _canonical_model_id(hf_name),
        "dimension": dimension,
        "count": len(units),
    }
    metadata_file = cache_dir / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))

    if index_paths is not None and index_config is not None:
        from tldr.indexing import update_meta_semantic
        update_meta_semantic(
            index_paths,
            index_config,
            model=_canonical_model_id(hf_name) or hf_name,
            dim=dimension,
            lang=None if lang == "all" else lang,
        )

    if console:
        console.print(f"[bold green]✓[/] Indexed {len(units)} code units")

    return len(units)


def _semantic_unit_search(
    project_path: str,
    query: str,
    k: int = 5,
    expand_graph: bool = False,
    model: Optional[str] = None,
    device: Optional[str] = None,
    index_paths=None,
    index_config=None,
) -> List[dict]:
    """Search for code units semantically.

    Args:
        project_path: Path to project root.
        query: Natural language query.
        k: Number of results to return.
        expand_graph: If True, include callers/callees in results.
        model: Model to use for query embedding. If None, uses
               the model from the index metadata.

    Returns:
        List of result dictionaries with name, file, line, score, etc.
    """
    # Handle empty query
    if not query or not query.strip():
        return []

    # Find project root for cache location (matches build_semantic_index behavior)
    scan_path = Path(project_path).resolve()
    if index_paths is None:
        project_root = _find_project_root(scan_path)
    else:
        project_root = scan_path
    cache_dir = project_root / ".tldr" / "cache" / "semantic"
    if index_paths is not None:
        cache_dir = index_paths.semantic_dir

    index_file = cache_dir / "index.faiss"
    metadata_file = cache_dir / "metadata.json"

    # Check index exists
    if not index_file.exists():
        raise FileNotFoundError(f"Semantic index not found at {index_file}. Run build_semantic_index first.")

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_file}. Run build_semantic_index first.")

    # Load metadata
    metadata = json.loads(metadata_file.read_text())
    units = metadata["units"]

    # Use model from metadata if not specified (ensures matching embeddings)
    index_model = _canonical_model_id(metadata.get("model"))
    if model is None and index_model:
        model = index_model
    elif model is not None:
        requested = _canonical_model_id(model)
        if index_model and requested and requested != index_model:
            raise ValueError(
                "Semantic search model mismatch with index. Use the index model or rebuild."
            )

    # Embed query (with instruction prefix for BGE)
    query_text = f"Represent this code search query: {query}"
    query_embedding = compute_embedding(query_text, model_name=model, device=device)
    query_embedding = query_embedding.reshape(1, -1)

    # Load index after torch initialization to avoid OpenMP conflicts on macOS
    import faiss
    index = faiss.read_index(str(index_file))

    # Validate dimension compatibility
    meta_dim = metadata.get("dimension")
    if meta_dim and index.d != meta_dim:
        raise ValueError("Semantic index dimension mismatch; rebuild required.")
    expected_dim = _model_dimension(model)
    if expected_dim and meta_dim and expected_dim != meta_dim:
        raise ValueError("Semantic model dimension mismatch; rebuild required.")

    # Search
    k = min(k, len(units))
    scores, indices = index.search(query_embedding, k)

    # Build results
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0 or idx >= len(units):
            continue

        unit = units[idx]
        result = {
            "name": unit["name"],
            "qualified_name": unit["qualified_name"],
            "file": unit["file"],
            "line": unit["line"],
            "unit_type": unit["unit_type"],
            "signature": unit["signature"],
            "score": float(score),
        }

        # Include graph expansion if requested
        if expand_graph:
            result["calls"] = unit.get("calls", [])
            result["called_by"] = unit.get("called_by", [])
            result["related"] = list(set(unit.get("calls", []) + unit.get("called_by", [])))

        results.append(result)

    return results


def _dedupe_preserve(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _rg_rank_files(
    repo_root: Path,
    *,
    pattern: str,
    glob: Optional[str],
    fixed_string: bool,
) -> List[str]:
    """Rank files by lexical match density + earliest hit line (deterministic)."""
    normalized_pattern = str(pattern or "").strip()
    if not normalized_pattern:
        return []

    cmd = ["rg", "-n", "--no-messages"]
    if fixed_string:
        cmd.append("--fixed-strings")
    if glob:
        cmd.extend(["--glob", glob])
    cmd.append(normalized_pattern)
    cmd.append(".")

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):  # 1 = no matches
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"rg failed (rc={proc.returncode}): {stderr}")

    hits_by_file: dict[str, dict[str, int]] = {}
    for line in (proc.stdout or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        file_path, line_s, _ = parts
        if file_path.startswith("./"):
            file_path = file_path[2:]
        file_path = file_path.replace("\\", "/")
        try:
            line_no = int(line_s)
        except ValueError:
            continue
        info = hits_by_file.setdefault(file_path, {"hits": 0, "min_line": line_no})
        info["hits"] += 1
        if line_no < info["min_line"]:
            info["min_line"] = line_no

    ranked = sorted(
        hits_by_file.items(),
        key=lambda kv: (-kv[1]["hits"], kv[1]["min_line"], kv[0]),
    )
    return [file_path for file_path, _ in ranked]


def _semantic_files_from_results(results: List[dict]) -> List[str]:
    ranked: List[str] = []
    for item in results or []:
        if not isinstance(item, dict):
            continue
        file_path = item.get("file") or item.get("path")
        if not isinstance(file_path, str):
            continue
        if file_path.startswith("./"):
            file_path = file_path[2:]
        ranked.append(file_path.replace("\\", "/"))
    return _dedupe_preserve(ranked)


def _rrf_file_scores(rankings: List[List[str]], *, rrf_k: int = 60) -> dict[str, float]:
    k = int(rrf_k)
    if k <= 0:
        k = 60
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, file_path in enumerate(ranking, start=1):
            scores[file_path] = scores.get(file_path, 0.0) + (1.0 / float(k + rank))
    return scores


def _rrf_fuse_file_rankings(rankings: List[List[str]], *, rrf_k: int = 60) -> List[str]:
    """Deterministic RRF fuse: score desc, then filepath asc for ties."""
    scores = _rrf_file_scores(rankings, rrf_k=rrf_k)
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [file_path for file_path, _ in fused]


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _effective_k_from_budget_tokens(
    k: Any,
    *,
    budget_tokens: Any,
    reference_budget_tokens: int = LANE3_REFERENCE_BUDGET_TOKENS,
) -> int:
    """Deterministically map budget_tokens to an effective retrieval k.

    - Reference budget (2000) preserves the requested k.
    - Invalid/non-positive budgets safely fall back to the requested k.
    - Result is clamped to [1, requested_k * LANE3_MAX_EFFECTIVE_K_MULTIPLIER].
    """
    requested_k = _safe_int(k)
    if requested_k is None or requested_k <= 0:
        requested_k = 1

    budget = _safe_int(budget_tokens)
    if budget is None or budget <= 0:
        return int(requested_k)

    reference = _safe_int(reference_budget_tokens)
    if reference is None or reference <= 0:
        reference = LANE3_REFERENCE_BUDGET_TOKENS

    # Integer half-up rounding avoids platform-dependent float edge cases.
    scaled = int(((requested_k * budget) + (reference // 2)) // reference)
    max_k = max(int(requested_k), int(requested_k) * LANE3_MAX_EFFECTIVE_K_MULTIPLIER)
    return max(1, min(max_k, scaled))


def _semantic_file_scores(results: List[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in results or []:
        if not isinstance(item, dict):
            continue
        file_path = item.get("file") or item.get("path")
        score = _safe_float(item.get("score"))
        if not isinstance(file_path, str) or score is None:
            continue
        if file_path.startswith("./"):
            file_path = file_path[2:]
        normalized = file_path.replace("\\", "/")
        prev = out.get(normalized)
        if prev is None or score > prev:
            out[normalized] = float(score)
    return out


def _lane2_ratio(value: Any) -> Optional[float]:
    ratio = _safe_float(value)
    if ratio is None or ratio <= 0:
        return None
    return float(ratio)


def _approx_token_count(text: str) -> int:
    normalized = (
        str(text or "")
        .replace("{", " ")
        .replace("}", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace(":", " ")
        .replace(",", " ")
        .replace('"', " ")
    )
    return len(normalized.split())


def _row_payload_tokens(row: dict) -> int:
    try:
        payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = str(row)
    return _approx_token_count(payload)


def _lane2_confidence(row: dict, *, rank: int, top_score: Optional[float]) -> float:
    semantic_score = _safe_float(row.get("semantic_score"))
    score = _safe_float(row.get("score"))

    if semantic_score is not None and -1.0 <= semantic_score <= 1.0:
        # Prefer semantic similarity for confidence so abstention can reflect
        # genuine relevance instead of always-normalized rank-based scores.
        base = (semantic_score + 1.0) / 2.0
    elif score is not None and -1.0 <= score <= 1.0:
        base = (score + 1.0) / 2.0
    elif score is not None and top_score is not None and top_score > 0:
        base = score / top_score
    else:
        base = 1.0 / float(max(1, rank))

    source_ranks = row.get("source_ranks")
    if isinstance(source_ranks, dict):
        if source_ranks.get("lexical") is not None and source_ranks.get("semantic") is not None:
            base += 0.05

    return max(0.0, min(1.0, float(base)))


def _lane2_row_sort_key(row: dict) -> tuple[str, str]:
    file_path = row.get("file") or row.get("path") or ""
    symbol = row.get("qualified_name") or row.get("name") or ""
    return (str(file_path), str(symbol))


def _lane2_postprocess(
    rows: List[dict],
    *,
    query: str,
    rerank: bool = False,
    rerank_top_n: int = 5,
    max_latency_ms_p50_ratio: Optional[float] = None,
    max_payload_tokens_median_ratio: Optional[float] = None,
) -> List[dict]:
    if not rows:
        return []

    out: List[dict] = [dict(row) for row in rows if isinstance(row, dict)]
    if not out:
        return []

    scores = [_safe_float(row.get("score")) for row in out]
    positive_scores = [score for score in scores if score is not None and score > 0]
    top_score = max(positive_scores) if positive_scores else None

    for rank, row in enumerate(out, start=1):
        row["confidence"] = _lane2_confidence(row, rank=rank, top_score=top_score)

    rerank_applied = bool(rerank)
    if rerank_applied:
        try:
            top_n = int(rerank_top_n)
        except (TypeError, ValueError):
            top_n = len(out)
        if top_n <= 0:
            top_n = len(out)
        top_n = min(len(out), max(1, top_n))
        head = sorted(
            out[:top_n],
            key=lambda row: (-float(row.get("confidence", 0.0)), *_lane2_row_sort_key(row)),
        )
        out = head + out[top_n:]

    for rank, row in enumerate(out, start=1):
        row["rank"] = int(rank)

    query_tokens = max(1, _approx_token_count(query))
    latency_ms_p50 = float(query_tokens * max(1, len(out)))
    payload_tokens_median = float(median([_row_payload_tokens(row) for row in out]))
    latency_bound = _lane2_ratio(max_latency_ms_p50_ratio)
    payload_bound = _lane2_ratio(max_payload_tokens_median_ratio)

    for row in out:
        row["rerank_applied"] = rerank_applied
        row["latency_ms_p50"] = latency_ms_p50
        row["payload_tokens_median"] = payload_tokens_median
        if latency_bound is not None:
            row["max_latency_ms_p50_ratio"] = latency_bound
        if payload_bound is not None:
            row["max_payload_tokens_median_ratio"] = payload_bound

    return out


def _lane2_apply_abstention(
    rows: List[dict],
    *,
    abstain_threshold: Optional[float],
    abstain_empty: bool,
) -> List[dict]:
    threshold = _safe_float(abstain_threshold)
    if threshold is None or threshold <= 0.0 or not rows:
        return rows
    threshold = min(float(threshold), 1.0)

    top_confidence = _safe_float(rows[0].get("confidence"))
    should_abstain = top_confidence is not None and top_confidence < threshold

    for row in rows:
        row["abstain_threshold"] = threshold
        row["abstained"] = bool(should_abstain)

    if should_abstain and bool(abstain_empty):
        return []
    return rows


def _lane4_normalize_call_graph_language(language: Any) -> Optional[str]:
    if not isinstance(language, str):
        return None
    normalized = language.strip().lower()
    if not normalized:
        return None
    if normalized == "javascript":
        normalized = "typescript"
    if normalized in LANE4_CALL_GRAPH_LANGUAGES:
        return normalized
    return None


def _lane4_languages_from_semantic_rows(
    semantic_rows: List[dict],
    *,
    impact_language: Optional[str],
) -> List[str]:
    normalized_impact = _lane4_normalize_call_graph_language(impact_language)
    impact_mode = str(impact_language or "auto").strip().lower()
    if normalized_impact is not None:
        return [normalized_impact]
    if impact_mode == "all":
        return sorted(LANE4_CALL_GRAPH_LANGUAGES)

    languages: set[str] = set()
    for row in semantic_rows:
        if not isinstance(row, dict):
            continue
        from_row = _lane4_normalize_call_graph_language(row.get("language"))
        if from_row is not None:
            languages.add(from_row)
            continue
        file_path = row.get("file")
        if not isinstance(file_path, str):
            continue
        suffix = Path(file_path).suffix.lower()
        from_suffix = LANE4_EXTENSION_TO_CALL_GRAPH_LANGUAGE.get(suffix)
        normalized_suffix = _lane4_normalize_call_graph_language(from_suffix)
        if normalized_suffix is not None:
            languages.add(normalized_suffix)

    return sorted(languages)


def _lane4_target_aliases(row: dict) -> List[str]:
    aliases: List[str] = []
    for key in ("qualified_name", "name"):
        value = row.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized:
            aliases.append(normalized)

    expanded: List[str] = []
    for alias in aliases:
        expanded.append(alias)
        for separator in ("::", "."):
            if separator in alias:
                expanded.append(alias.rsplit(separator, 1)[-1])

    return _dedupe_preserve(expanded)


def _lane4_impact_targets(
    semantic_rows: List[dict],
    *,
    impact_limit: Optional[int],
) -> tuple[List[dict], List[dict]]:
    limit = _safe_int(impact_limit)
    if limit is None:
        limit = LANE4_IMPACT_DEFAULT_LIMIT
    limit = max(0, int(limit))

    targets: List[dict] = []
    partial_failures: List[dict] = []
    seen_keys: set[tuple[Optional[str], str]] = set()

    for row_index, row in enumerate(semantic_rows, start=1):
        if len(targets) >= limit:
            break
        if not isinstance(row, dict):
            partial_failures.append(
                {
                    "stage": "target_selection",
                    "row_index": row_index,
                    "reason": "invalid_semantic_row",
                    "message": "semantic row is not an object",
                }
            )
            continue

        file_path = row.get("file")
        normalized_file = file_path if isinstance(file_path, str) and file_path else None
        aliases = _lane4_target_aliases(row)
        if not aliases:
            partial_failures.append(
                {
                    "stage": "target_selection",
                    "row_index": row_index,
                    "file": normalized_file,
                    "reason": "missing_symbol",
                    "message": "semantic row does not include a function or method name",
                }
            )
            continue

        dedupe_key = (normalized_file, aliases[0])
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        targets.append(
            {
                "row_index": row_index,
                "file": normalized_file,
                "aliases": aliases,
            }
        )

    return targets, partial_failures


def _lane4_symbol_from_row(row: dict) -> dict:
    name = row.get("name")
    qualified_name = row.get("qualified_name")
    if not isinstance(name, str):
        name = None
    if not isinstance(qualified_name, str):
        qualified_name = None
    line = row.get("line")
    if not isinstance(line, int):
        line = None
    unit_type = row.get("unit_type")
    if not isinstance(unit_type, str):
        unit_type = None
    return {
        "name": name,
        "qualified_name": qualified_name,
        "line": line,
        "unit_type": unit_type,
    }


def _lane4_build_result_row(row: dict, *, rank: int) -> dict:
    file_path = row.get("file")
    if not isinstance(file_path, str):
        file_path = None

    retrieval: dict[str, Any] = {
        "score": _safe_float(row.get("score")),
        "semantic_score": _safe_float(row.get("semantic_score")),
    }
    source_ranks = row.get("source_ranks")
    if isinstance(source_ranks, dict):
        retrieval["source_ranks"] = source_ranks
    confidence = _safe_float(row.get("confidence"))
    if confidence is not None:
        retrieval["confidence"] = confidence

    return {
        "rank": int(rank),
        "file": file_path,
        "symbol": _lane4_symbol_from_row(row),
        "retrieval": retrieval,
        "impact": {
            "status": "skipped",
            "latency_ms": None,
            "caller_count": 0,
            "truncated": None,
            "callers": [],
            "error_code": "not_selected",
            "message": "not selected for impact analysis",
        },
    }


def _lane4_collect_callers(node: dict, *, depth: int, out: List[dict]) -> bool:
    truncated = bool(node.get("truncated"))
    callers = node.get("callers")
    if not isinstance(callers, list):
        return truncated
    for caller in callers:
        if not isinstance(caller, dict):
            continue
        file_path = caller.get("file")
        function = caller.get("function")
        if isinstance(file_path, str) and isinstance(function, str):
            out.append(
                {
                    "file": file_path,
                    "function": function,
                    "line": caller.get("line") if isinstance(caller.get("line"), int) else None,
                    "depth": int(depth),
                }
            )
        if _lane4_collect_callers(caller, depth=depth + 1, out=out):
            truncated = True
    return truncated


def _lane4_extract_callers(payload: dict) -> tuple[List[dict], bool]:
    targets = payload.get("targets") if isinstance(payload, dict) else None
    if not isinstance(targets, dict):
        return ([], False)

    callers: List[dict] = []
    truncated = False
    for _, node in sorted(targets.items(), key=lambda item: str(item[0])):
        if not isinstance(node, dict):
            continue
        if _lane4_collect_callers(node, depth=1, out=callers):
            truncated = True

    deduped: List[dict] = []
    seen: set[tuple[int, str, str, Optional[int]]] = set()
    for caller in sorted(
        callers,
        key=lambda c: (int(c.get("depth", 0)), str(c.get("file")), str(c.get("function")), str(c.get("line"))),
    ):
        key = (
            int(caller.get("depth", 0)),
            str(caller.get("file")),
            str(caller.get("function")),
            caller.get("line") if isinstance(caller.get("line"), int) else None,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(caller)
    return (deduped, truncated)


def _lane4_partial_failure(
    *,
    stage: str,
    code: str,
    message: str,
    rank: Optional[int] = None,
    file_path: Optional[str] = None,
    symbol: Optional[str] = None,
    recoverable: bool = True,
    latency_ms: Optional[float] = None,
) -> dict:
    out: dict[str, Any] = {
        "stage": str(stage),
        "code": str(code),
        "message": str(message),
        "recoverable": bool(recoverable),
    }
    if isinstance(rank, int):
        out["rank"] = int(rank)
    if isinstance(file_path, str):
        out["file"] = file_path
    if isinstance(symbol, str):
        out["symbol"] = symbol
    if latency_ms is not None:
        out["latency_ms"] = float(latency_ms)
    return out


def _lane5_normalize_label_mode(label_mode: Any) -> str:
    mode = str(label_mode or "auto").strip().lower()
    if mode not in LANE5_CLUSTER_LABEL_MODES:
        return "auto"
    return mode


def _lane5_tokens(value: Any) -> List[str]:
    if not isinstance(value, str):
        return []
    pieces = re.split(r"[^A-Za-z0-9]+", value.strip().lower())
    tokens: List[str] = []
    for piece in pieces:
        if len(piece) < 2:
            continue
        if piece in LANE5_TOKEN_STOPWORDS:
            continue
        tokens.append(piece)
    return _dedupe_preserve(tokens)


def _lane5_file_tokens(file_path: Any) -> List[str]:
    if not isinstance(file_path, str):
        return []
    normalized = file_path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    tokens: List[str] = []
    for segment in normalized.split("/"):
        tokens.extend(_lane5_tokens(segment))
    return _dedupe_preserve(tokens)


def _lane5_symbol_tokens(row: dict) -> List[str]:
    values: List[str] = []
    for key in ("qualified_name", "name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value)
    tokens: List[str] = []
    for value in values:
        tokens.extend(_lane5_tokens(value))
    return _dedupe_preserve(tokens)


def _lane5_member_sort_key(member: dict) -> tuple[int, str, str, str]:
    return (
        int(member.get("rank", 0)),
        str(member.get("file") or ""),
        str(member.get("qualified_name") or member.get("name") or ""),
        str(member.get("line") if isinstance(member.get("line"), int) else ""),
    )


def _lane5_cluster_anchor_key(cluster: dict) -> tuple[int, str, str]:
    members = cluster.get("members")
    if not isinstance(members, list) or not members:
        return (0, "", "")
    first = min(
        (member for member in members if isinstance(member, dict)),
        key=_lane5_member_sort_key,
        default={},
    )
    return (
        int(first.get("rank", 0)),
        str(first.get("file") or ""),
        str(first.get("qualified_name") or first.get("name") or ""),
    )


def _lane5_overlap_signature(
    *,
    lhs_file: set[str],
    lhs_symbol: set[str],
    lhs_all: set[str],
    rhs_file: set[str],
    rhs_symbol: set[str],
    rhs_all: set[str],
) -> tuple[int, int, int, int]:
    file_overlap = len(lhs_file.intersection(rhs_file))
    symbol_overlap = len(lhs_symbol.intersection(rhs_symbol))
    any_overlap = len(lhs_all.intersection(rhs_all))
    total = file_overlap + symbol_overlap + any_overlap
    return (int(total), int(file_overlap), int(symbol_overlap), int(any_overlap))


def _lane5_cluster_pair_overlap(lhs: dict, rhs: dict) -> tuple[int, int, int, int]:
    return _lane5_overlap_signature(
        lhs_file=set(lhs.get("tokens_file", set())),
        lhs_symbol=set(lhs.get("tokens_symbol", set())),
        lhs_all=set(lhs.get("tokens_all", set())),
        rhs_file=set(rhs.get("tokens_file", set())),
        rhs_symbol=set(rhs.get("tokens_symbol", set())),
        rhs_all=set(rhs.get("tokens_all", set())),
    )


def _lane5_merge_clusters(dst: dict, src: dict) -> None:
    dst_members = dst.get("members")
    src_members = src.get("members")
    if not isinstance(dst_members, list) or not isinstance(src_members, list):
        return
    dst_members.extend(member for member in src_members if isinstance(member, dict))
    dst["tokens_file"] = set(dst.get("tokens_file", set())).union(set(src.get("tokens_file", set())))
    dst["tokens_symbol"] = set(dst.get("tokens_symbol", set())).union(
        set(src.get("tokens_symbol", set()))
    )
    dst["tokens_all"] = set(dst.get("tokens_all", set())).union(set(src.get("tokens_all", set())))


def _lane5_row_identity(member: dict) -> str:
    symbol = member.get("qualified_name") or member.get("name") or ""
    line = member.get("line") if isinstance(member.get("line"), int) else ""
    return (
        f"{member.get('row_id') or ''}|{member.get('file') or ''}|"
        f"{symbol}|{line}|{member.get('rank') or ''}"
    )


def _lane5_cluster_id(cluster: dict) -> str:
    members = cluster.get("members")
    if not isinstance(members, list):
        return "cluster-000000000000"
    identities = [
        _lane5_row_identity(member)
        for member in members
        if isinstance(member, dict)
    ]
    payload = "|".join(sorted(identities))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"cluster-{digest}"


def _lane5_assignment_digest(clusters: List[dict]) -> str:
    rows: List[str] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "")
        members = cluster.get("members")
        if not isinstance(members, list):
            rows.append(cluster_id)
            continue
        member_rows: List[str] = []
        for member in members:
            if not isinstance(member, dict):
                continue
            symbol = member.get("symbol")
            if isinstance(symbol, dict):
                symbol_name = symbol.get("qualified_name") or symbol.get("name") or ""
                line = symbol.get("line")
            else:
                symbol_name = member.get("qualified_name") or member.get("name") or ""
                line = member.get("line")
            member_rows.append(
                f"{member.get('file') or ''}|{symbol_name}|{line or ''}|{member.get('rank') or ''}"
            )
        rows.append(f"{cluster_id}:{'|'.join(sorted(member_rows))}")
    payload = "||".join(sorted(rows))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _lane5_top_token(token_sets: List[set[str]]) -> Optional[str]:
    counts: dict[str, int] = {}
    for token_set in token_sets:
        for token in sorted(token_set):
            counts[token] = counts.get(token, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]


def _lane5_cluster_label(cluster: dict, *, label_mode: str) -> str:
    members = cluster.get("members")
    if not isinstance(members, list) or not members:
        return "cluster"

    first = min(
        (member for member in members if isinstance(member, dict)),
        key=_lane5_member_sort_key,
        default={},
    )
    file_token = _lane5_top_token(
        [
            set(member.get("tokens_file", set()))
            for member in members
            if isinstance(member, dict)
        ]
    )
    symbol_token = _lane5_top_token(
        [
            set(member.get("tokens_symbol", set()))
            for member in members
            if isinstance(member, dict)
        ]
    )
    first_file = str(first.get("file") or "")
    first_symbol = str(first.get("qualified_name") or first.get("name") or "")

    mode = _lane5_normalize_label_mode(label_mode)
    if mode == "file":
        if isinstance(file_token, str):
            return file_token
        if first_file:
            return Path(first_file).name or first_file
        return "cluster"
    if mode == "symbol":
        if isinstance(symbol_token, str):
            return symbol_token
        if first_symbol:
            return first_symbol
        if first_file:
            return Path(first_file).stem or first_file
        return "cluster"

    # auto
    if isinstance(file_token, str) and isinstance(symbol_token, str):
        return f"{file_token}:{symbol_token}"
    if isinstance(symbol_token, str):
        return symbol_token
    if isinstance(file_token, str):
        return file_token
    if first_symbol:
        return first_symbol
    if first_file:
        return Path(first_file).name or first_file
    return "cluster"


def _lane5_common_tokens(members: List[dict], field: str) -> List[str]:
    if not members:
        return []
    common: Optional[set[str]] = None
    for member in members:
        token_set = set(member.get(field, set()))
        if common is None:
            common = token_set
            continue
        common = common.intersection(token_set)
    if common is None:
        return []
    return sorted(common)


def _lane5_result_row(member: dict) -> dict:
    retrieval: dict[str, Any] = {
        "score": _safe_float(member.get("score")),
        "semantic_score": _safe_float(member.get("semantic_score")),
    }
    confidence = _safe_float(member.get("confidence"))
    if confidence is not None:
        retrieval["confidence"] = confidence
    source_ranks = member.get("source_ranks")
    if isinstance(source_ranks, dict):
        retrieval["source_ranks"] = source_ranks

    return {
        "rank": int(member.get("rank", 0)),
        "file": member.get("file") if isinstance(member.get("file"), str) else None,
        "symbol": {
            "name": member.get("name") if isinstance(member.get("name"), str) else None,
            "qualified_name": (
                member.get("qualified_name")
                if isinstance(member.get("qualified_name"), str)
                else None
            ),
            "line": member.get("line") if isinstance(member.get("line"), int) else None,
            "unit_type": (
                member.get("unit_type") if isinstance(member.get("unit_type"), str) else None
            ),
        },
        "retrieval": retrieval,
    }


def _lane5_normalize_semantic_rows(rows: List[dict]) -> List[dict]:
    prepared: List[tuple[int, int, str, str, str, dict]] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue

        input_rank = _safe_int(row.get("rank"))
        if input_rank is None or input_rank <= 0:
            input_rank = idx

        file_path = row.get("file")
        if isinstance(file_path, str):
            file_path = file_path.replace("\\", "/")
            if file_path.startswith("./"):
                file_path = file_path[2:]
        else:
            file_path = None

        name = row.get("name") if isinstance(row.get("name"), str) else None
        qualified_name = row.get("qualified_name") if isinstance(row.get("qualified_name"), str) else None
        line = row.get("line") if isinstance(row.get("line"), int) else None
        file_tokens = set(_lane5_file_tokens(file_path))
        symbol_tokens = set(_lane5_symbol_tokens(row))
        all_tokens = set(file_tokens).union(symbol_tokens)

        member = {
            "row_id": f"row-{idx:04d}",
            "rank": int(input_rank),
            "file": file_path,
            "name": name,
            "qualified_name": qualified_name,
            "line": line,
            "unit_type": row.get("unit_type") if isinstance(row.get("unit_type"), str) else None,
            "score": _safe_float(row.get("score")),
            "semantic_score": _safe_float(row.get("semantic_score")),
            "confidence": _safe_float(row.get("confidence")),
            "source_ranks": row.get("source_ranks") if isinstance(row.get("source_ranks"), dict) else None,
            "tokens_file": file_tokens,
            "tokens_symbol": symbol_tokens,
            "tokens_all": all_tokens,
        }
        prepared.append(
            (
                int(input_rank),
                idx,
                str(file_path or ""),
                str(qualified_name or name or ""),
                str(line if line is not None else ""),
                member,
            )
        )

    prepared = sorted(prepared, key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    out: List[dict] = []
    for rank, (_, _, _, _, _, member) in enumerate(prepared, start=1):
        member["rank"] = int(rank)
        member["row_id"] = f"row-{rank:04d}"
        out.append(member)
    return out


def _lane5_cluster_rows(
    members: List[dict],
    *,
    cluster_count: Optional[int],
    cluster_min_size: int,
) -> List[dict]:
    clusters: List[dict] = []
    target_count = _safe_int(cluster_count)
    if target_count is not None and target_count <= 0:
        target_count = None
    min_size = _safe_int(cluster_min_size)
    if min_size is None or min_size <= 0:
        min_size = LANE5_DEFAULT_CLUSTER_MIN_SIZE

    for member in members:
        best_idx: Optional[int] = None
        best_overlap = (0, 0, 0, 0)
        for idx, cluster in enumerate(clusters):
            overlap = _lane5_overlap_signature(
                lhs_file=set(member.get("tokens_file", set())),
                lhs_symbol=set(member.get("tokens_symbol", set())),
                lhs_all=set(member.get("tokens_all", set())),
                rhs_file=set(cluster.get("tokens_file", set())),
                rhs_symbol=set(cluster.get("tokens_symbol", set())),
                rhs_all=set(cluster.get("tokens_all", set())),
            )
            if overlap[0] <= 0:
                continue
            if best_idx is None or overlap > best_overlap:
                best_idx = idx
                best_overlap = overlap
                continue
            if overlap == best_overlap:
                if _lane5_cluster_anchor_key(cluster) < _lane5_cluster_anchor_key(clusters[best_idx]):
                    best_idx = idx
                    best_overlap = overlap

        if best_idx is None:
            if target_count is None or len(clusters) < target_count:
                clusters.append(
                    {
                        "members": [member],
                        "tokens_file": set(member.get("tokens_file", set())),
                        "tokens_symbol": set(member.get("tokens_symbol", set())),
                        "tokens_all": set(member.get("tokens_all", set())),
                    }
                )
                continue

            # Target count reached: deterministically assign by smallest cluster first.
            fallback_idx = min(
                range(len(clusters)),
                key=lambda i: (len(clusters[i].get("members", [])), _lane5_cluster_anchor_key(clusters[i])),
            )
            _lane5_merge_clusters(
                clusters[fallback_idx],
                {
                    "members": [member],
                    "tokens_file": set(member.get("tokens_file", set())),
                    "tokens_symbol": set(member.get("tokens_symbol", set())),
                    "tokens_all": set(member.get("tokens_all", set())),
                },
            )
            continue

        _lane5_merge_clusters(
            clusters[best_idx],
            {
                "members": [member],
                "tokens_file": set(member.get("tokens_file", set())),
                "tokens_symbol": set(member.get("tokens_symbol", set())),
                "tokens_all": set(member.get("tokens_all", set())),
            },
        )

    if min_size > 1:
        while len(clusters) > 1:
            small_indexes = [
                idx for idx, cluster in enumerate(clusters) if len(cluster.get("members", [])) < min_size
            ]
            if not small_indexes:
                break

            source_idx = min(small_indexes, key=lambda idx: _lane5_cluster_anchor_key(clusters[idx]))
            source = clusters[source_idx]

            best_target_idx: Optional[int] = None
            best_overlap = (-1, -1, -1, -1)
            for idx, candidate in enumerate(clusters):
                if idx == source_idx:
                    continue
                overlap = _lane5_cluster_pair_overlap(source, candidate)
                if best_target_idx is None or overlap > best_overlap:
                    best_target_idx = idx
                    best_overlap = overlap
                    continue
                if overlap == best_overlap:
                    if _lane5_cluster_anchor_key(candidate) < _lane5_cluster_anchor_key(
                        clusters[best_target_idx]
                    ):
                        best_target_idx = idx
                        best_overlap = overlap

            if best_target_idx is None:
                break

            _lane5_merge_clusters(clusters[best_target_idx], source)
            del clusters[source_idx]

    for cluster in clusters:
        members_sorted = sorted(
            [
                member
                for member in cluster.get("members", [])
                if isinstance(member, dict)
            ],
            key=_lane5_member_sort_key,
        )
        cluster["members"] = members_sorted

    return sorted(
        clusters,
        key=lambda cluster: (
            -max(
                [
                    float(member.get("score"))
                    for member in cluster.get("members", [])
                    if _safe_float(member.get("score")) is not None
                ]
                or [0.0]
            ),
            _lane5_cluster_anchor_key(cluster),
        ),
    )


def semantic_navigation_cluster_search(
    project_path: str,
    query: str,
    k: int = 5,
    *,
    expand_graph: bool = False,
    model: Optional[str] = None,
    device: Optional[str] = None,
    index_paths=None,
    index_config=None,
    retrieval_mode: str = "semantic",
    no_result_guard: str = "none",
    rg_pattern: Optional[str] = None,
    rg_glob: Optional[str] = None,
    rrf_k: int = 60,
    abstain_threshold: Optional[float] = None,
    abstain_empty: bool = False,
    rerank: bool = False,
    rerank_top_n: int = 5,
    max_latency_ms_p50_ratio: Optional[float] = None,
    max_payload_tokens_median_ratio: Optional[float] = None,
    budget_tokens: Optional[int] = None,
    cluster_count: Optional[int] = None,
    cluster_min_size: int = LANE5_DEFAULT_CLUSTER_MIN_SIZE,
    cluster_max_members: Optional[int] = LANE5_DEFAULT_CLUSTER_MAX_MEMBERS,
    cluster_label_mode: str = "auto",
) -> dict:
    """Lane5 deterministic semantic navigation/clustering contract."""
    total_started = time.perf_counter()

    k_requested = _safe_int(k)
    if k_requested is None or k_requested <= 0:
        k_requested = 5
    k_effective = int(k_requested)
    if budget_tokens is not None:
        k_effective = _effective_k_from_budget_tokens(k_requested, budget_tokens=budget_tokens)

    requested_cluster_count = _safe_int(cluster_count)
    if requested_cluster_count is not None and requested_cluster_count <= 0:
        requested_cluster_count = None
    requested_min_size = _safe_int(cluster_min_size)
    if requested_min_size is None or requested_min_size <= 0:
        requested_min_size = LANE5_DEFAULT_CLUSTER_MIN_SIZE
    requested_max_members = _safe_int(cluster_max_members)
    if requested_max_members is None or requested_max_members <= 0:
        requested_max_members = LANE5_DEFAULT_CLUSTER_MAX_MEMBERS
    requested_label_mode = _lane5_normalize_label_mode(cluster_label_mode)

    default_cluster_count = min(LANE5_DEFAULT_CLUSTER_COUNT, max(1, int(k_effective)))
    cluster_count_target = (
        int(requested_cluster_count)
        if requested_cluster_count is not None
        else int(default_cluster_count)
    )

    semantic_started = time.perf_counter()
    try:
        semantic_rows = semantic_search(
            project_path,
            query,
            k=k_requested,
            expand_graph=expand_graph,
            model=model,
            device=device,
            index_paths=index_paths,
            index_config=index_config,
            retrieval_mode=retrieval_mode,
            no_result_guard=no_result_guard,
            rg_pattern=rg_pattern,
            rg_glob=rg_glob,
            rrf_k=rrf_k,
            abstain_threshold=abstain_threshold,
            abstain_empty=abstain_empty,
            rerank=rerank,
            rerank_top_n=rerank_top_n,
            max_latency_ms_p50_ratio=max_latency_ms_p50_ratio,
            max_payload_tokens_median_ratio=max_payload_tokens_median_ratio,
            budget_tokens=budget_tokens,
        )
    except Exception as exc:
        total_ms = max(0.0, (time.perf_counter() - total_started) * 1000.0)
        latency_bound = _lane2_ratio(max_latency_ms_p50_ratio)
        payload_bound = _lane2_ratio(max_payload_tokens_median_ratio)
        return {
            "schema_version": LANE5_SCHEMA_VERSION,
            "feature_set_id": LANE5_FEATURE_SET_ID,
            "status": "error",
            "query": query,
            "budget_tokens": budget_tokens,
            "retrieval_mode": str(retrieval_mode or "semantic"),
            "k_requested": int(k_requested),
            "k_effective": int(k_effective),
            "timing_ms": {
                "total": float(total_ms),
                "semantic": float(total_ms),
                "clustering": 0.0,
                "labeling": 0.0,
            },
            "clustering": {
                "cluster_count_requested": requested_cluster_count,
                "cluster_count_target": int(cluster_count_target),
                "cluster_min_size": int(requested_min_size),
                "cluster_max_members": int(requested_max_members),
                "cluster_label_mode": requested_label_mode,
            },
            "counts": {
                "retrieval_results": 0,
                "clusters": 0,
                "cluster_count": 0,
                "clustered_results": 0,
                "unclustered_results": 0,
                "truncated_clusters": 0,
            },
            "results": [],
            "clusters": [],
            "partial_failures": [
                {
                    "stage": "semantic",
                    "code": "semantic_runtime_error",
                    "message": str(exc),
                    "recoverable": False,
                }
            ],
            "regression_metadata": {
                "budget_tokens": budget_tokens,
                "max_latency_ms_p50_ratio": latency_bound,
                "max_payload_tokens_median_ratio": payload_bound,
                "latency_ms_p50": float(total_ms),
                "payload_tokens_median": 0.0,
                "assignment_digest": "",
            },
        }
    semantic_ms = max(0.0, (time.perf_counter() - semantic_started) * 1000.0)

    members = _lane5_normalize_semantic_rows(
        [row for row in semantic_rows if isinstance(row, dict)]
    )
    if members:
        cluster_count_target = max(1, min(int(cluster_count_target), len(members)))
    else:
        cluster_count_target = 0

    clustering_started = time.perf_counter()
    internal_clusters = _lane5_cluster_rows(
        members,
        cluster_count=cluster_count_target if cluster_count_target > 0 else None,
        cluster_min_size=int(requested_min_size),
    )
    clustering_ms = max(0.0, (time.perf_counter() - clustering_started) * 1000.0)

    labeling_started = time.perf_counter()
    internal_rows: List[dict] = []
    truncated_clusters = 0

    for cluster in internal_clusters:
        cluster_id = _lane5_cluster_id(cluster)
        label = _lane5_cluster_label(cluster, label_mode=requested_label_mode)
        cluster_members = [
            member for member in cluster.get("members", []) if isinstance(member, dict)
        ]
        overflow = max(0, len(cluster_members) - int(requested_max_members))
        if overflow > 0:
            truncated_clusters += 1
        visible_members = (
            cluster_members[: int(requested_max_members)]
            if requested_max_members > 0
            else cluster_members
        )

        member_rows: List[dict] = []
        for member in visible_members:
            member_row = _lane5_result_row(member)
            member_row["cluster_id"] = cluster_id
            member_rows.append(member_row)

        internal_rows.append(
            {
                "cluster_id": cluster_id,
                "label": label,
                "size": len(cluster_members),
                "truncated": bool(overflow > 0),
                "member_overflow": int(overflow),
                "tokens": {
                    "file": _lane5_common_tokens(cluster_members, "tokens_file"),
                    "symbol": _lane5_common_tokens(cluster_members, "tokens_symbol"),
                },
                "members": member_rows,
                "_all_members": cluster_members,
            }
        )

    internal_rows = sorted(
        internal_rows,
        key=lambda cluster: (
            str(cluster.get("cluster_id") or ""),
            str(cluster.get("label") or ""),
        ),
    )
    row_to_cluster: dict[str, tuple[str, int]] = {}
    clusters: List[dict] = []
    for cluster_rank, cluster in enumerate(internal_rows, start=1):
        all_members = cluster.get("_all_members")
        if isinstance(all_members, list):
            for member in all_members:
                if not isinstance(member, dict):
                    continue
                row_to_cluster[str(member.get("row_id"))] = (
                    str(cluster.get("cluster_id") or ""),
                    int(cluster_rank),
                )
        cluster["rank"] = int(cluster_rank)
        for row in cluster.get("members", []):
            if isinstance(row, dict):
                row["cluster_rank"] = int(cluster_rank)
        cluster.pop("_all_members", None)
        clusters.append(cluster)
    labeling_ms = max(0.0, (time.perf_counter() - labeling_started) * 1000.0)

    results: List[dict] = []
    for member in members:
        row = _lane5_result_row(member)
        cluster_info = row_to_cluster.get(str(member.get("row_id")))
        if cluster_info is not None:
            row["cluster_id"] = cluster_info[0]
            row["cluster_rank"] = int(cluster_info[1])
        results.append(row)

    latency_bound = _lane2_ratio(max_latency_ms_p50_ratio)
    payload_bound = _lane2_ratio(max_payload_tokens_median_ratio)
    payload_tokens_median = (
        float(median([_row_payload_tokens(row) for row in results])) if results else 0.0
    )
    unclustered_results = max(0, len(results) - len(row_to_cluster))
    total_ms = max(0.0, (time.perf_counter() - total_started) * 1000.0)
    return {
        "schema_version": LANE5_SCHEMA_VERSION,
        "feature_set_id": LANE5_FEATURE_SET_ID,
        "status": "ok",
        "query": query,
        "budget_tokens": budget_tokens,
        "retrieval_mode": str(retrieval_mode or "semantic"),
        "k_requested": int(k_requested),
        "k_effective": int(k_effective),
        "timing_ms": {
            "total": float(total_ms),
            "semantic": float(semantic_ms),
            "clustering": float(clustering_ms),
            "labeling": float(labeling_ms),
        },
        "clustering": {
            "cluster_count_requested": requested_cluster_count,
            "cluster_count_target": int(cluster_count_target),
            "cluster_min_size": int(requested_min_size),
            "cluster_max_members": int(requested_max_members),
            "cluster_label_mode": requested_label_mode,
        },
        "counts": {
            "retrieval_results": len(members),
            "clusters": len(clusters),
            "cluster_count": len(clusters),
            "clustered_results": len(results),
            "unclustered_results": int(unclustered_results),
            "truncated_clusters": int(truncated_clusters),
        },
        "results": results,
        "clusters": clusters,
        "partial_failures": [],
        "regression_metadata": {
            "budget_tokens": budget_tokens,
            "max_latency_ms_p50_ratio": latency_bound,
            "max_payload_tokens_median_ratio": payload_bound,
            "latency_ms_p50": float(total_ms),
            "payload_tokens_median": float(payload_tokens_median),
            "assignment_digest": _lane5_assignment_digest(clusters),
        },
    }


semantic_navigate_cluster_search = semantic_navigation_cluster_search
navigate_cluster_search = semantic_navigation_cluster_search
semantic_navigate_search = semantic_navigation_cluster_search


def hybrid_file_search(
    project_path: str,
    query: str,
    k: int = 5,
    *,
    model: Optional[str] = None,
    device: Optional[str] = None,
    index_paths=None,
    index_config=None,
    no_result_guard: str = "none",
    rg_pattern: Optional[str] = None,
    rg_glob: Optional[str] = None,
    rrf_k: int = 60,
    abstain_threshold: Optional[float] = None,
    abstain_empty: bool = False,
    rerank: bool = False,
    rerank_top_n: int = 5,
    max_latency_ms_p50_ratio: Optional[float] = None,
    max_payload_tokens_median_ratio: Optional[float] = None,
    budget_tokens: Optional[int] = None,
) -> List[dict]:
    """Hybrid file retrieval using lexical + semantic rank fusion."""
    if not query or not query.strip():
        return []

    effective_k = k
    if budget_tokens is not None:
        effective_k = _effective_k_from_budget_tokens(k, budget_tokens=budget_tokens)

    if effective_k <= 0:
        return []

    if no_result_guard not in {"none", "rg_empty"}:
        raise ValueError("no_result_guard must be 'none' or 'rg_empty'")

    scan_path = Path(project_path).resolve()
    has_explicit_pattern = isinstance(rg_pattern, str) and bool(rg_pattern.strip())
    lexical_pattern = rg_pattern.strip() if has_explicit_pattern else query.strip()

    lexical_rank = _rg_rank_files(
        scan_path,
        pattern=lexical_pattern,
        glob=rg_glob,
        fixed_string=not has_explicit_pattern,
    )
    if no_result_guard == "rg_empty" and not lexical_rank:
        return []

    semantic_k = max(int(effective_k), int(effective_k) * 5)
    semantic_results = _semantic_unit_search(
        project_path,
        query,
        k=semantic_k,
        expand_graph=False,
        model=model,
        device=device,
        index_paths=index_paths,
        index_config=index_config,
    )
    semantic_rank = _semantic_files_from_results(semantic_results)
    semantic_scores = _semantic_file_scores(semantic_results)

    fused_rank = _rrf_fuse_file_rankings([lexical_rank, semantic_rank], rrf_k=rrf_k)
    fused_scores = _rrf_file_scores([lexical_rank, semantic_rank], rrf_k=rrf_k)

    lexical_pos = {fp: i for i, fp in enumerate(lexical_rank, start=1)}
    semantic_pos = {fp: i for i, fp in enumerate(semantic_rank, start=1)}

    out: List[dict] = []
    for rank, file_path in enumerate(fused_rank[: int(effective_k)], start=1):
        out.append(
            {
                "file": file_path,
                "score": float(fused_scores.get(file_path, 0.0)),
                "semantic_score": _safe_float(semantic_scores.get(file_path)),
                "rank": int(rank),
                "source_ranks": {
                    "lexical": lexical_pos.get(file_path),
                    "semantic": semantic_pos.get(file_path),
                },
            }
        )
    out = _lane2_postprocess(
        out,
        query=query,
        rerank=rerank,
        rerank_top_n=rerank_top_n,
        max_latency_ms_p50_ratio=max_latency_ms_p50_ratio,
        max_payload_tokens_median_ratio=max_payload_tokens_median_ratio,
    )
    out = _lane2_apply_abstention(
        out,
        abstain_threshold=abstain_threshold,
        abstain_empty=abstain_empty,
    )
    return out


def semantic_search(
    project_path: str,
    query: str,
    k: int = 5,
    expand_graph: bool = False,
    model: Optional[str] = None,
    device: Optional[str] = None,
    index_paths=None,
    index_config=None,
    retrieval_mode: str = "semantic",
    no_result_guard: str = "none",
    rg_pattern: Optional[str] = None,
    rg_glob: Optional[str] = None,
    rrf_k: int = 60,
    abstain_threshold: Optional[float] = None,
    abstain_empty: bool = False,
    rerank: bool = False,
    rerank_top_n: int = 5,
    max_latency_ms_p50_ratio: Optional[float] = None,
    max_payload_tokens_median_ratio: Optional[float] = None,
    budget_tokens: Optional[int] = None,
) -> List[dict]:
    """Search for code semantically or with hybrid lexical+semantic retrieval.

    Default behavior remains semantic unit search.
    """
    mode = str(retrieval_mode or "semantic")
    if mode not in {"semantic", "hybrid"}:
        raise ValueError("retrieval_mode must be 'semantic' or 'hybrid'")

    if no_result_guard not in {"none", "rg_empty"}:
        raise ValueError("no_result_guard must be 'none' or 'rg_empty'")

    if mode == "hybrid":
        return hybrid_file_search(
            project_path,
            query,
            k=k,
            model=model,
            device=device,
            index_paths=index_paths,
            index_config=index_config,
            no_result_guard=no_result_guard,
            rg_pattern=rg_pattern,
            rg_glob=rg_glob,
            rrf_k=rrf_k,
            abstain_threshold=abstain_threshold,
            abstain_empty=abstain_empty,
            rerank=rerank,
            rerank_top_n=rerank_top_n,
            max_latency_ms_p50_ratio=max_latency_ms_p50_ratio,
            max_payload_tokens_median_ratio=max_payload_tokens_median_ratio,
            budget_tokens=budget_tokens,
        )

    # Guard mode is opt-in for semantic mode too (helps reduce negative-query FPR).
    if no_result_guard == "rg_empty" and query and query.strip():
        scan_path = Path(project_path).resolve()
        has_explicit_pattern = isinstance(rg_pattern, str) and bool(rg_pattern.strip())
        lexical_pattern = rg_pattern.strip() if has_explicit_pattern else query.strip()
        lexical_rank = _rg_rank_files(
            scan_path,
            pattern=lexical_pattern,
            glob=rg_glob,
            fixed_string=not has_explicit_pattern,
        )
        if not lexical_rank:
            return []

    effective_k = k
    if budget_tokens is not None:
        effective_k = _effective_k_from_budget_tokens(k, budget_tokens=budget_tokens)

    results = _semantic_unit_search(
        project_path,
        query,
        k=effective_k,
        expand_graph=expand_graph,
        model=model,
        device=device,
        index_paths=index_paths,
        index_config=index_config,
    )
    lane2_enabled = (
        abstain_threshold is not None
        or bool(abstain_empty)
        or bool(rerank)
        or max_latency_ms_p50_ratio is not None
        or max_payload_tokens_median_ratio is not None
    )
    if not lane2_enabled:
        return results

    results = _lane2_postprocess(
        results,
        query=query,
        rerank=rerank,
        rerank_top_n=rerank_top_n,
        max_latency_ms_p50_ratio=max_latency_ms_p50_ratio,
        max_payload_tokens_median_ratio=max_payload_tokens_median_ratio,
    )
    return _lane2_apply_abstention(
        results,
        abstain_threshold=abstain_threshold,
        abstain_empty=abstain_empty,
    )


def compound_semantic_impact_search(
    project_path: str,
    query: str,
    k: int = 5,
    *,
    expand_graph: bool = False,
    model: Optional[str] = None,
    device: Optional[str] = None,
    index_paths=None,
    index_config=None,
    retrieval_mode: str = "semantic",
    no_result_guard: str = "none",
    rg_pattern: Optional[str] = None,
    rg_glob: Optional[str] = None,
    rrf_k: int = 60,
    abstain_threshold: Optional[float] = None,
    abstain_empty: bool = False,
    rerank: bool = False,
    rerank_top_n: int = 5,
    max_latency_ms_p50_ratio: Optional[float] = None,
    max_payload_tokens_median_ratio: Optional[float] = None,
    budget_tokens: Optional[int] = None,
    impact_depth: int = LANE4_IMPACT_DEFAULT_DEPTH,
    impact_limit: Optional[int] = LANE4_IMPACT_DEFAULT_LIMIT,
    impact_language: Optional[str] = "auto",
    ignore_spec=None,
    workspace_root: Optional[Path] = None,
) -> dict:
    """Run lane4 compound retrieval: semantic retrieval with impact enrichment."""

    k_requested = _safe_int(k)
    if k_requested is None or k_requested <= 0:
        k_requested = 5
    k_effective = int(k_requested)
    if budget_tokens is not None:
        k_effective = _effective_k_from_budget_tokens(k_requested, budget_tokens=budget_tokens)

    total_start = time.perf_counter()
    semantic_start = time.perf_counter()
    try:
        semantic_rows = semantic_search(
            project_path,
            query,
            k=k_requested,
            expand_graph=expand_graph,
            model=model,
            device=device,
            index_paths=index_paths,
            index_config=index_config,
            retrieval_mode=retrieval_mode,
            no_result_guard=no_result_guard,
            rg_pattern=rg_pattern,
            rg_glob=rg_glob,
            rrf_k=rrf_k,
            abstain_threshold=abstain_threshold,
            abstain_empty=abstain_empty,
            rerank=rerank,
            rerank_top_n=rerank_top_n,
            max_latency_ms_p50_ratio=max_latency_ms_p50_ratio,
            max_payload_tokens_median_ratio=max_payload_tokens_median_ratio,
            budget_tokens=budget_tokens,
        )
    except Exception as exc:
        semantic_ms = (time.perf_counter() - semantic_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        return {
            "schema_version": LANE4_SCHEMA_VERSION,
            "feature_set_id": LANE4_FEATURE_SET_ID,
            "status": "error",
            "query": query,
            "budget_tokens": _safe_int(budget_tokens),
            "retrieval_mode": str(retrieval_mode or "semantic"),
            "k_requested": int(k_requested),
            "k_effective": int(k_effective),
            "timing_ms": {
                "total": float(total_ms),
                "semantic": float(semantic_ms),
                "impact_total": 0.0,
                "impact_p50": 0.0,
            },
            "counts": {
                "retrieval_results": 0,
                "impact_attempted": 0,
                "impact_ok": 0,
                "impact_partial": 0,
                "impact_error": 0,
            },
            "results": [],
            "partial_failures": [
                _lane4_partial_failure(
                    stage="semantic",
                    code="semantic_runtime_error",
                    message=str(exc),
                    recoverable=False,
                    latency_ms=semantic_ms,
                )
            ],
            "regression_metadata": {
                "budget_tokens": _safe_int(budget_tokens),
                "latency_ms_p50": float(total_ms),
                "payload_tokens_median": 0.0,
                "max_latency_ms_p50_ratio": _lane2_ratio(max_latency_ms_p50_ratio),
                "max_payload_tokens_median_ratio": _lane2_ratio(max_payload_tokens_median_ratio),
            },
        }

    semantic_ms = (time.perf_counter() - semantic_start) * 1000.0
    normalized_rows = [dict(row) for row in semantic_rows if isinstance(row, dict)]
    results: List[dict] = [
        _lane4_build_result_row(row, rank=rank)
        for rank, row in enumerate(normalized_rows, start=1)
    ]
    by_rank = {int(row["rank"]): row for row in results if isinstance(row, dict)}

    targets, selection_failures = _lane4_impact_targets(
        normalized_rows,
        impact_limit=impact_limit,
    )

    partial_failures: List[dict] = []
    for failure in selection_failures:
        rank = failure.get("row_index")
        rank_int = rank if isinstance(rank, int) else None
        row_out = by_rank.get(rank_int) if rank_int is not None else None
        file_path = row_out.get("file") if isinstance(row_out, dict) else None
        symbol_name = None
        if isinstance(row_out, dict):
            symbol_obj = row_out.get("symbol")
            if isinstance(symbol_obj, dict):
                symbol_name = (
                    symbol_obj.get("qualified_name")
                    if isinstance(symbol_obj.get("qualified_name"), str)
                    else symbol_obj.get("name")
                )
            impact_obj = row_out.get("impact")
            if isinstance(impact_obj, dict):
                impact_obj["error_code"] = "skipped_no_symbol"
                impact_obj["message"] = str(failure.get("message") or "semantic row has no symbol")
        partial_failures.append(
            _lane4_partial_failure(
                stage="target_selection",
                code="skipped_no_symbol",
                message=str(failure.get("message") or "semantic row has no symbol"),
                rank=rank_int,
                file_path=file_path if isinstance(file_path, str) else None,
                symbol=symbol_name if isinstance(symbol_name, str) else None,
                recoverable=True,
            )
        )

    try:
        depth = int(impact_depth)
    except (TypeError, ValueError):
        depth = LANE4_IMPACT_DEFAULT_DEPTH
    if depth <= 0:
        depth = LANE4_IMPACT_DEFAULT_DEPTH

    impact_latencies: List[float] = []
    impact_attempted = 0
    impact_ok = 0

    if targets:
        languages = _lane4_languages_from_semantic_rows(
            normalized_rows,
            impact_language=impact_language,
        )

        from tldr.analysis import impact_analysis
        from tldr.cross_file_calls import ProjectCallGraph, build_project_call_graph

        combined_graph = ProjectCallGraph()
        if not languages:
            for target in targets:
                rank = int(target["row_index"])
                row_out = by_rank.get(rank)
                if not isinstance(row_out, dict):
                    continue
                symbol_obj = row_out.get("symbol")
                symbol_name = None
                if isinstance(symbol_obj, dict):
                    symbol_name = symbol_obj.get("qualified_name") or symbol_obj.get("name")
                impact_obj = row_out.get("impact")
                if isinstance(impact_obj, dict):
                    impact_obj.update(
                        {
                            "status": "error",
                            "error_code": "impact_runtime_error",
                            "message": "No supported call-graph language could be inferred",
                        }
                    )
                partial_failures.append(
                    _lane4_partial_failure(
                        stage="impact",
                        code="impact_runtime_error",
                        message="No supported call-graph language could be inferred",
                        rank=rank,
                        file_path=row_out.get("file") if isinstance(row_out.get("file"), str) else None,
                        symbol=symbol_name if isinstance(symbol_name, str) else None,
                        recoverable=True,
                    )
                )
        else:
            for language in languages:
                try:
                    language_graph = build_project_call_graph(
                        project_path,
                        language=language,
                        ignore_spec=ignore_spec,
                        workspace_root=workspace_root,
                    )
                except Exception as exc:
                    partial_failures.append(
                        _lane4_partial_failure(
                            stage="call_graph",
                            code="impact_runtime_error",
                            message=f"{language}: {exc}",
                            recoverable=True,
                        )
                    )
                    continue
                for edge in language_graph.sorted_edges():
                    combined_graph.add_edge(*edge)

            for target in targets:
                rank = int(target["row_index"])
                row_out = by_rank.get(rank)
                if not isinstance(row_out, dict):
                    continue
                symbol_obj = row_out.get("symbol")
                symbol_name = None
                if isinstance(symbol_obj, dict):
                    symbol_name = symbol_obj.get("qualified_name") or symbol_obj.get("name")
                target_file = target.get("file")
                aliases = [alias for alias in target.get("aliases", []) if isinstance(alias, str)]
                if not aliases:
                    continue

                impact_attempted += 1
                target_start = time.perf_counter()
                chosen_alias: Optional[str] = None
                payload: Optional[dict] = None
                last_error = "Function not found in call graph"
                for alias in aliases:
                    if not combined_graph.edges:
                        last_error = "Unable to build call graph for impact analysis"
                        break
                    candidate = impact_analysis(
                        combined_graph,
                        alias,
                        max_depth=depth,
                        target_file=target_file,
                    )
                    if isinstance(candidate, dict) and candidate.get("error"):
                        last_error = str(candidate.get("error"))
                        continue
                    chosen_alias = alias
                    payload = candidate if isinstance(candidate, dict) else {}
                    break

                target_latency_ms = (time.perf_counter() - target_start) * 1000.0
                impact_latencies.append(float(target_latency_ms))
                impact_obj = row_out.get("impact")
                if not isinstance(impact_obj, dict):
                    impact_obj = {}
                    row_out["impact"] = impact_obj

                if payload is None:
                    impact_obj.update(
                        {
                            "status": "error",
                            "latency_ms": float(target_latency_ms),
                            "caller_count": 0,
                            "truncated": None,
                            "callers": [],
                            "error_code": "impact_not_found",
                            "message": last_error,
                        }
                    )
                    partial_failures.append(
                        _lane4_partial_failure(
                            stage="impact",
                            code="impact_not_found",
                            message=last_error,
                            rank=rank,
                            file_path=row_out.get("file") if isinstance(row_out.get("file"), str) else None,
                            symbol=symbol_name if isinstance(symbol_name, str) else None,
                            recoverable=True,
                            latency_ms=target_latency_ms,
                        )
                    )
                else:
                    callers, truncated = _lane4_extract_callers(payload)
                    impact_obj.update(
                        {
                            "status": "ok",
                            "latency_ms": float(target_latency_ms),
                            "caller_count": int(len(callers)),
                            "truncated": bool(truncated),
                            "callers": callers,
                            "error_code": None,
                            "message": None,
                            "function": chosen_alias,
                        }
                    )
                    impact_ok += 1

    impact_total_ms = float(sum(impact_latencies))
    impact_p50_ms = float(median(impact_latencies)) if impact_latencies else 0.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    partial_failures = sorted(
        partial_failures,
        key=lambda item: (
            str(item.get("stage")),
            int(item.get("rank")) if isinstance(item.get("rank"), int) else 0,
            str(item.get("file") or ""),
            str(item.get("symbol") or ""),
            str(item.get("code") or ""),
        ),
    )
    impact_partial = sum(1 for item in partial_failures if bool(item.get("recoverable")))
    impact_error = sum(1 for item in partial_failures if not bool(item.get("recoverable")))
    status = "ok"
    if impact_error > 0:
        status = "error"
    elif impact_partial > 0:
        status = "partial"

    payload_tokens_median = (
        float(median([_row_payload_tokens(row) for row in results])) if results else 0.0
    )

    return {
        "schema_version": LANE4_SCHEMA_VERSION,
        "feature_set_id": LANE4_FEATURE_SET_ID,
        "status": status,
        "query": query,
        "budget_tokens": _safe_int(budget_tokens),
        "retrieval_mode": str(retrieval_mode or "semantic"),
        "k_requested": int(k_requested),
        "k_effective": int(k_effective),
        "timing_ms": {
            "total": float(total_ms),
            "semantic": float(semantic_ms),
            "impact_total": float(impact_total_ms),
            "impact_p50": float(impact_p50_ms),
        },
        "counts": {
            "retrieval_results": int(len(results)),
            "impact_attempted": int(impact_attempted),
            "impact_ok": int(impact_ok),
            "impact_partial": int(impact_partial),
            "impact_error": int(impact_error),
        },
        "results": results,
        "partial_failures": partial_failures,
        "regression_metadata": {
            "budget_tokens": _safe_int(budget_tokens),
            "latency_ms_p50": float(total_ms),
            "payload_tokens_median": float(payload_tokens_median),
            "max_latency_ms_p50_ratio": _lane2_ratio(max_latency_ms_p50_ratio),
            "max_payload_tokens_median_ratio": _lane2_ratio(max_payload_tokens_median_ratio),
        },
    }
