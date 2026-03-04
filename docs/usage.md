# Execution Modes: When the Daemon Is Used

llm-tldr has three execution paths for queries. Understanding which one is active
determines whether you pay model-load overhead on every query or amortize it once.

## The Three Paths

### 1. CLI Direct (`tldrf <command>`)

Each invocation of `tldrf` is a **fresh Python process**. It imports the analysis
functions directly (e.g. `semantic_search()`, `get_cfg_context()`), loads the
embedding model into memory, runs the query, prints output, and exits.

```bash
tldrf semantic search "authentication" --path .
tldrf impact "login" . --file src/auth.py
tldrf cfg src/auth.py login
```

**Cost per invocation:** ~500-1000ms startup + model load overhead (MPS GPU
auto-detected). The model is loaded fresh each time because the process exits
after each command.

**When to use:** Occasional one-off queries. Fine for exploring, but expensive
if scripting repeated queries.

### 2. Daemon (`tldrf daemon start` + socket queries)

The daemon is a long-running background process that keeps indexes and the
embedding model warm in memory. Queries arrive over a Unix domain socket and
return in milliseconds.

```bash
# Start the daemon (auto-forks to background)
tldrf daemon start --project .

# Query via raw JSON (fast, model already loaded)
tldrf daemon query '{"cmd":"semantic","action":"search","query":"authentication","k":10}' --project .
tldrf daemon query '{"cmd":"impact","func":"login","file":"src/auth.py"}' --project .
tldrf daemon query '{"cmd":"cfg","file":"src/auth.py","function":"login"}' --project .

# Check status
tldrf daemon status --project .

# Stop when done
tldrf daemon stop --project .
```

**Cost per query:** ~18-300ms (warm model, no startup overhead).

**When to use:** Repeated queries from scripts, editor integrations, or any
workflow where you run multiple commands against the same project.

### 3. MCP Server (`tldrf mcp`)

The MCP server is the intended integration path for AI tools (Claude Code,
Cursor, etc.). It **auto-starts the daemon** and routes all queries through it,
so the model stays warm transparently.

```bash
tldrf mcp  # starts MCP server, auto-manages daemon lifecycle
```

**Cost per query:** Same as daemon path (~18-300ms). The MCP server handles
daemon lifecycle automatically -- starts it on first query, keeps it alive for
the session.

**When to use:** Editor/AI tool integrations. This is the recommended path for
Claude Code hooks and MCP clients.

## Benchmark Harness (`bench_h2h_predict.py`)

The benchmark script has its own execution modes because it invokes tool profiles
as shell commands:

### Without `--use-daemon` (default)

Each benchmark query spawns a **subprocess**: `subprocess.run(["uv", "run", "tldrf", ...])`.
This pays full process startup + model load on every single query.

```bash
uv run python scripts/bench_h2h_predict.py \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.hybrid_lane1.v1.json \
  --category retrieval --trial 1 --budget-tokens 2000
```

### With `--use-daemon` (recommended)

Queries are routed through the daemon's Unix socket. The script manages the
daemon lifecycle automatically (start before loop, ping, stop after).

```bash
uv run python scripts/bench_h2h_predict.py \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.hybrid_lane1.v1.json \
  --use-daemon \
  --category retrieval --trial 1 --budget-tokens 2000
```

Add `--daemon-keep-alive` to leave the daemon running between invocations
(useful when running multiple profiles back-to-back).

**Constraints:**
- `--use-daemon` only works with `tool_id='llm-tldr'` profiles.
- Non-daemon templates (contextplus, rg-native) automatically fall back to subprocess.
- Run metadata records `"execution_mode": "daemon"` for auditability.

## Measured Latency Comparison

All lanes benchmarked on Django corpus (budget 2000, trial 1).
MPS GPU confirmed via `torch.backends.mps.is_available()=True`, inference device `mps`.
Result correctness: all predictions byte-identical between subprocess and daemon modes.

### Daemon vs Subprocess Latency — All Retrieval Lanes

| Lane | Feature | Subprocess p50 | Daemon p50 | Speedup | Daemon mean | Daemon p90 | Parity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| lane1 | hybrid | 5426.2 ms | 296.9 ms | **18.3x** | 373.8 ms | 319.7 ms | 60/60 |
| lane2 | abstain/rerank | 4989.0 ms | 292.5 ms | **17.1x** | 412.9 ms | 314.7 ms | 60/60 |
| lane3 | budget-aware | 4912.8 ms | 290.1 ms | **16.9x** | 286.7 ms | 300.1 ms | 60/60 |
| lane4 | compound | 5161.9 ms | 295.7 ms | **17.5x** | 295.9 ms | 321.5 ms | 12/12 |
| lane5 | navigate-cluster | 5170.4 ms | 293.1 ms | **17.6x** | 288.7 ms | 301.0 ms | 60/60 |

### Daemon vs Subprocess Latency — Structural Categories (Gate B)

| Category | Subprocess p50 | Daemon p50 | Speedup | Parity |
| --- | ---: | ---: | ---: | --- |
| impact | 212.2 ms | 13.9 ms | **15.2x** | 15/15 semantic match |
| slice | 168.9 ms | 5.2 ms | **32.3x** | 10/10 exact |
| complexity | 156.9 ms | 1.7 ms | **93.1x** | 10/10 exact |
| data_flow | 158.2 ms | 2.9 ms | **54.5x** | 10/10 exact |

### Daemon vs rg-native Latency Comparison

| Metric | rg-native | Daemon (lane1) | Daemon / rg-native ratio |
| --- | ---: | ---: | ---: |
| p50 latency | 216.2 ms | 296.9 ms | 1.37x |

- Daemon-mode lanes are within `1.34-1.37x` of rg-native p50 (was `23-25x` in subprocess mode).
- The remaining gap is embedding inference cost (semantic+hybrid) vs pure regex.

### Verified

- **Daemon used**: `execution_mode: "daemon"` in run metadata for all lanes
- **GPU (MPS) used**: `_get_device()` returns `"mps"` -- the daemon auto-detects and uses Apple Silicon GPU
- **Result parity**: All retrieval predictions across lanes 1-5 are byte-identical between subprocess and daemon modes
- **Structural parity**: impact 15/15 semantic match (caller ordering differs), slice/complexity/data_flow exact match

## Summary: Which Path Am I On?

| Context | Execution Path | Model Load | Latency |
| --- | --- | --- | ---: |
| `tldrf semantic search ...` | CLI direct | Every invocation | ~5000ms |
| `tldrf daemon query '{...}'` | Daemon socket | Once at start | ~300ms |
| `tldrf mcp` (Claude Code, etc.) | MCP -> daemon | Once at start | ~300ms |
| `bench_h2h_predict.py` (default) | Subprocess per query | Every query | ~5000ms |
| `bench_h2h_predict.py --use-daemon` | Daemon socket | Once at start | ~300ms |

**Rule of thumb:** If you're running more than one query, use the daemon.
The MCP server does this automatically. For benchmarks, add `--use-daemon`.
For shell scripts, start the daemon first with `tldrf daemon start`.

## Gap: No `--use-daemon` on CLI Commands (Yet)

There is currently no `tldrf semantic search --use-daemon` flag that would
transparently route a regular CLI command through an already-running daemon.
Each `tldrf` invocation calls the Python function directly in-process.

If you need daemon-speed queries from the shell, the options today are:
1. Use `tldrf daemon query` with JSON commands
2. Use the MCP server via an editor integration
3. Start the daemon and query the socket directly from a script

A future enhancement could add `--use-daemon` to all CLI commands, making
`tldrf semantic search "query" --use-daemon` transparently route through the
daemon when one is running.
