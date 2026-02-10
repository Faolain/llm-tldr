import runpy
import sys
from pathlib import Path


def _load_mod():
    # Load script as a module dict without executing main().
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_rg_impact_baseline.py")


def test_derive_rg_pattern_function():
    mod = _load_mod()
    derive = mod["_derive_rg_pattern"]
    assert derive("loadEnvConfig") == r"\bloadEnvConfig\s*\("


def test_derive_rg_pattern_method():
    mod = _load_mod()
    derive = mod["_derive_rg_pattern"]
    assert derive("Server.handleRequestImpl") == r"\.handleRequestImpl\s*\("


def test_derive_rg_pattern_constructor():
    mod = _load_mod()
    derive = mod["_derive_rg_pattern"]
    assert (
        derive("LRUCache.constructor")
        == r"\bnew\s+LRUCache(?:\s*<[^>\n]+>)?\s*\("
    )


def test_filter_definition_hits_for_function_symbol():
    mod = _load_mod()
    RgMatch = mod["RgMatch"]
    filt = mod["_filter_definition_hits"]

    matches = [
        RgMatch(file="a.ts", line=1, text="export function loadEnvConfig(dir: string) {"),
        RgMatch(file="b.ts", line=10, text="loadEnvConfig(dir, false, Log)"),
    ]
    out = filt(matches, callee_symbol="loadEnvConfig")
    assert len(out) == 1
    assert out[0].file == "b.ts"


def test_guess_enclosing_symbol_class_method():
    mod = _load_mod()
    guess = mod["_guess_enclosing_symbol"]

    lines = [
        "export default abstract class Server {",
        "  private async handleRequestImpl(req: any) {",
        "    const url = parseUrlUtil(req.url)",
        "    return url",
        "  }",
        "}",
    ]
    sym = guess(lines, line_no_1=3)
    assert sym == "Server.handleRequestImpl"


def test_guess_enclosing_symbol_function_decl():
    mod = _load_mod()
    guess = mod["_guess_enclosing_symbol"]

    lines = [
        "export async function doThing() {",
        "  const x = 1",
        "  foo(x)",
        "}",
    ]
    sym = guess(lines, line_no_1=3)
    assert sym == "doThing"


def test_guess_enclosing_symbol_arrow_binding():
    mod = _load_mod()
    guess = mod["_guess_enclosing_symbol"]

    lines = [
        "export const formatDynamicImportPath = (dir: string) => {",
        "  return join(dir, 'x')",
        "}",
    ]
    sym = guess(lines, line_no_1=2)
    assert sym == "formatDynamicImportPath"
