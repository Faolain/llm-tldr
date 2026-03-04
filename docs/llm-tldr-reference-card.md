# llm-tldr Reference Card

> Code analysis CLI with semantic retrieval, structural analysis, and daemon mode.
> Invoke via `tldrf` (CLI), daemon socket (JSON), or MCP server (editor integrations).

## Capabilities

| Capability | What it does | When to reach for it | CLI | Daemon JSON | Daemon p50 | Tokens |
|---|---|---|---|---|---:|---:|
| **Semantic search** | Hybrid lexical+semantic code retrieval for NL queries | Don't know exact names; exploring | `tldrf semantic search "{q}" --path .` | `{"cmd":"semantic","action":"search","query":"{q}","k":10}` | 290 ms | 78 |
| **Impact analysis** | All callers of a function (who calls this?) | Blast radius before refactoring | `tldrf impact "{fn}" . --file {f}` | `{"cmd":"impact","func":"{fn}","file":"{f}"}` | 14 ms | 26 |
| **Context / call graph** | Reachable functions from entry point at depth N | What does a change propagate to? | `tldrf context "{entry}" --project . --depth {n}` | `{"cmd":"context","entry":"{entry}","depth":{n}}` | 0.2 ms* | 128 |
| **Slice** | Backward trace: lines influencing a variable at a point | Where does a bad value originate? | `tldrf slice "{file}" "{fn}" {line}` | `{"cmd":"slice","file":"{f}","function":"{fn}","line":{n}}` | 5 ms | 11 |
| **Data flow (DFG)** | Forward trace: how data moves through a function | Data transformations after slicing | `tldrf dfg "{file}" "{fn}"` | `{"cmd":"dfg","file":"{f}","function":"{fn}"}` | 3 ms | 13 |
| **CFG / complexity** | Cyclomatic complexity of a function | Complex hotspots; test prioritization | `tldrf cfg "{file}" "{fn}"` | `{"cmd":"cfg","file":"{f}","function":"{fn}"}` | 1.7 ms | 8 |

*Context: 0.2 ms warm (cached call graph); ~15.5 s cold start on first call per entry point.
Dispatch rule: `entry` with `/` and no `.` uses module-path mode; all other entries use symbol mode.

## Proven Workflows

| Workflow | Steps | Use when | Expected signal |
|---|---|---|---|
| **Refactor path** | `impact` -> `context` -> `rg` | Changing a function; need blast radius + affected code + lexical confirmation | impact f1=0.848, context f1=0.880, then rg for final grep validation |
| **Debug path** | `slice` -> `dfg` | Tracing a bad value back to its source, then forward through transformations | slice f1=0.919, dfg origin/flow accuracy=1.0 |
| **Discovery path** | `semantic search` (lane 2 or 3) | Onboarding to codebase; finding implementations by concept | MRR=0.874, recall@5=0.877, zero false positives |

## Retrieval Lanes

Lanes stack incrementally. **Lane 1 (hybrid) is the recommended default** — highest recall. Use lane 3 when token budget is tight.

| Lane | Strategy | Adds over previous | MRR | R@5 | Best for |
|---|---|---|---:|---:|---|
| 1 | Hybrid | Lexical + semantic fusion | 0.856 | **0.930** | **Default** — finds the most relevant results |
| 2 | Abstain/rerank | Confidence filtering + reranking | 0.874 | 0.877 | Higher ranking precision; filtering low-confidence noise |
| 3 | Budget-aware | Dynamic k based on token budget | 0.874 | 0.877 | Token-constrained contexts (best MRR + budget respect) |
| 4 | Compound | Semantic + impact jointly | 0.745 | 0.818 | Finding code that is both relevant AND called often |
| 5 | Navigate-cluster | Semantic clustering of results | 0.874 | 0.877 | Exploring multiple facets of a broad query |

Lane 5 determinism: 1.0 coverage, 1.0 digest match, 0.982 cluster recall@3 (n=180).

## Performance (Daemon vs Subprocess)

Always use daemon for multi-query sessions. MCP server auto-starts it.

| Category | Subprocess p50 | Daemon p50 | Speedup |
|---|---:|---:|---:|
| Retrieval (lanes 1-5) | ~5000 ms | ~293 ms | 17x |
| Impact | 212 ms | 14 ms | 15x |
| Context (warm) | 15,516 ms | 0.2 ms | 82,000x |
| Slice | 169 ms | 5 ms | 32x |
| Data flow | 158 ms | 3 ms | 55x |
| Complexity | 157 ms | 1.7 ms | 93x |

Daemon retrieval is within **1.37x of rg-native** p50 (216 ms). All results byte-identical to subprocess.

## Comparison (Retrieval @ 2000 token budget)

| Tool | MRR | Recall@5 | FPR@5 | p50 | Payload tokens | Structural workflows |
|---|---:|---:|---:|---:|---:|---|
| **llm-tldr** (lane 2/3) | **0.874** | **0.877** | **0.0** | 293 ms (daemon) | 78 | impact, context, slice, dfg, cfg |
| rg-native | 0.813 | 0.877 | 0.0 | 216 ms | 12 | none |
| contextplus | 0.216 | 0.298 | 1.0 | 7,717 ms | 329 | none |

**Bottom line:** llm-tldr matches rg recall with +7% MRR at 1.37x latency, plus six structural capabilities alternatives lack. contextplus is 4x slower with 100% false positive rate.

## Execution Modes

| Context | Path | Model load | Latency |
|---|---|---|---:|
| `tldrf <cmd>` | CLI direct | Every call | ~5000 ms |
| `tldrf daemon query --json '{...}'` | Daemon socket | Once | ~300 ms |
| `tldrf mcp` | MCP -> daemon | Once | ~300 ms |

**Rule of thumb:** More than one query? Use the daemon. MCP does this automatically.

Start: `tldrf daemon start --project .` | Stop: `tldrf daemon stop --project .`

> Deep dive: `docs/usage.md` (execution modes), `implementations/008-benchmark-summary.md` (full runbook)
