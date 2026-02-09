It was noticed after trying the benchmark in benchmarks/benchmark/bench-tldrf-vs-rg.sh that rg is typically far better along than just using tldrf. This came as quite a shock because the idea was the 5 layer system described in the README would make performance far better. However the results were the following:

<Results>
## State Machine Match (Docs vs Peerbit)

  Docs (docs/architecture-viz-phase2.html) connection state machine

  - connecting (created in createConnection at docs/architecture-viz-phase2.html:554)
  - connecting -> negotiating when transport becomes ready (docs/architecture-viz-phase2.html:767 to docs/architecture-viz-phase2.html:773)
  - negotiating -> connected once both pubsub + blocks layers are ready (docs/architecture-viz-phase2.html:775 to docs/architecture-viz-phase2.html:815)
  - * -> disconnecting on hangup (docs/architecture-viz-phase2.html:817 to docs/architecture-viz-phase2.html:834)

  Peerbit equivalents

  - Dial / “connecting”: Peerbit.dial calls libp2p.dial(...). (packages/clients/peerbit/src/peer.ts:295)
  - “pubsub-ready” + “blocks-ready” gating: Peerbit.dial explicitly waits for both services to become available:
      - this.libp2p.services.pubsub.waitFor(..., { target: "neighbor" }) (packages/clients/peerbit/src/peer.ts:310)
      - this.libp2p.services.blocks.waitFor(..., { target: "neighbor" }) (packages/clients/peerbit/src/peer.ts:318)
  - Negotiation phase: stream setup negotiates protocols on outbound stream creation with connection.newStream(..., { negotiateFully: true }). (packages/transport/stream/src/
    index.ts:1288, packages/transport/stream/src/index.ts:1318)
  - “ready/connected”: the underlying readiness check used by waitFor is “neighbor” = readable + writable streams. (packages/transport/stream/src/index.ts:2621)
  - Hangup / “disconnecting”: Peerbit.hangUp calls libp2p.hangUp(...). (packages/clients/peerbit/src/peer.ts:327)

  Conclusion: The Phase2 HTML is a simplified but directionally correct model of Peerbit’s real flow: dial, negotiate stream/protocols, then consider the peer usable only
  once both pubsub and blocks services report readiness. The main mismatch is that Peerbit does not have a single explicit conn.state field with connecting/negotiating/
  connected/disconnecting; those states are implicit across libp2p connection status + stream readiness + waitFor logic.

  ## Benchmark: rg vs tldrf (Daemon)

  ### What changed (why it’s faster now)

  Daemon semantic search was failing with “Broken pipe” in non-interactive runs because the unix-forked daemon inherited dead stdout/stderr pipes. I fixed daemon stdio
  detachment in ../llm-tldr/tldr/daemon/startup.py (redirects fds 0/1/2 to /dev/null in the forked child). After that:

  - daemon starts cleanly and stays running
  - semantic search works and is fast when warm

  ### Speed (measured)

  - Daemon start: ~0.23s (warm cache reuse via TLDR_DAEMON_DIR=/tmp/tldr and consistent --cache-root/--index/--scan-root)
  - tldrf daemon semantic query:
      - first query after daemon start (model load): ~6.7s
      - subsequent warm queries: ~0.11s to 0.15s each

  For comparable “find exact string” lookups, rg was effectively instantaneous (sub-10ms; time rounded to 0.00s) for:

  - services.pubsub.waitFor|services.blocks.waitFor in packages/clients/peerbit/src/peer.ts
  - negotiateFully in packages/transport/stream/src/index.ts
  - "peer:reachable" in packages/transport/stream/src/index.ts

  ### Usefulness (coverage/noise)

  What tldrf (semantic) did better

  - Finding the right area of the codebase from a conceptual query, e.g. this semantic query reliably surfaced Peerbit.dial in ~0.12s warm:
      - query: class Peerbit implements ProgramClient async dial libp2p.dial
      - top hit: packages/clients/peerbit/src/peer.ts (dial)

  What rg did better

  - Exact matching for specific mechanisms once you know what to look for (e.g. negotiateFully, "peer:reachable", exact event names, exact method calls).
  - It also did the full doc→code mapping across the big event list faster in practice because tldrf context/graph features are weak/slow on TS here.

  Where tldrf under-delivered

  - tldrf context Peerbit.dial was slow (~43s) and the output was minimal (basically a header), so it didn’t add value for TS comprehension compared to opening the file
    directly.
  - Semantic search could be noisy unless the query includes distinctive identifiers; “peer:reachable dispatch” initially returned irrelevant hits until narrowed.

  ## Comparison vs the rg-only subagent

  The rg-only agent extracted the doc state machine and mapped many doc events to concrete Peerbit implementations (including SharedLog replicator events and a code-only
  extra event replication:change in packages/programs/data/shared-log/src/index.ts:442). It also correctly identified doc-only conceptual events like replication:no-leader /
  replication:no-route that do not exist as emitted event names.

  Net:

  - Faster per-query: rg
  - Faster end-to-end for broad, exact mappings: rg
  - Most useful unique value from tldrf (with daemon): fast warm semantic “where is this concept implemented?” when strings don’t match docs
</Results>

<Discussion>
## When tldrf Is Worth Using (In Peerbit)

  ### 1. “I know what it does, not what it’s called”

  This is where rg is weakest and where tldrf semantic search (with the daemon warm) is genuinely good.

  Example from what we just did:

  - Query like “wait for pubsub and blocks after dial” got you to the right subsystem quickly even if you didn’t know the exact strings to rg.

  Rule of thumb:

  - Use tldrf semantic to get 3-10 candidate files/functions fast.
  - Then switch to rg (or open the file) to verify the exact mechanics.

  ### 2. Dependency debugging (node_modules / installed artifacts)

  rg can find strings, but it can’t answer “what calls this” or “what does this do” without you manually walking.
  tldrf + isolated dependency indexing is a good fit for “where is X implemented in this installed version?”.

  ### 3. Quick structural inventory

  extract, imports, importers, structure can be useful for “what’s in this file/module” without you hand-scanning. In practice, extract is the most reliable one on TS here.

  ### 4. Value/path debugging (sometimes)

  In languages where dfg/slice/cfg are strong, these can beat rg because they answer:

  - “where did this value come from?”
  - “what affects line X?”
  - “what branches lead to this state?”

  Caveat: in this codebase (mostly TypeScript), these analyses may be incomplete depending on how well the TS front-end is working for your project shape.

  ## When rg Is Still the Better Default

  ### 1. Anything docs/config/string-literal driven

  Your architecture viz HTML is basically a bundle of strings and JS objects.

  - rg is the right tool to extract event names and verify they exist in code.

  ### 2. Exact event names / exact API usage

  If you know the string (e.g. "replicator:join", negotiateFully, services.pubsub.waitFor), rg is faster, more precise, and lower-noise.

  ## Are impact/calls Useful Here?

  Right now: not reliably enough to beat rg for Peerbit TS.

  Reasons we observed:

  - tldrf context on TS was slow and low-information for Peerbit.dial.
  - Call-graph dependent commands (calls/impact) tend to underperform when the TS call graph isn’t robust for the project (monorepo + build tooling + TS patterns). When they
    work they’re great, but you can’t count on them yet.

  If you want impact/calls to become worth it, the next step is to explicitly benchmark call-graph quality on a few known edges (“who calls X?”) and only recommend it in
  Agents.md if it consistently returns real edges.

  ## Practical Workflow Recommendation (What I’d Put in Agents.md)

  - Use rg for: docs alignment, configs, event-name auditing, exhaustive scans.
  - Use tldrf (daemon) for: semantic “where is the implementation of this concept?” discovery, then verify with rg.
  - Use tldrf extract/imports/importers for: quick module mapping.
  - Treat impact/calls/dfg/slice as “try it when stuck”, not default, until you confirm they work well on this repo’s TypeScript.

  If you want, I can run a small, concrete benchmark specifically for impact/calls accuracy on 5 known functions (expected callers known ahead of time) and then either
  tighten or remove those recommendations from Agents.md.
</Discussion>

<Addendum>
 Call-graph quality can be off because building a correct call graph for TypeScript requires real module + symbol resolution, and tldrf’s TS call graph is mostly syntax +
  heuristics, so it will miss or mis-link a lot of edges in a monorepo like Peerbit.

  Concrete reasons (from ../llm-tldr/tldr/cross_file_calls.py):

  1. Module resolution is naive

  - _build_typescript_call_graph treats non-relative imports (e.g. @peerbit/...) as just the string module name and never resolves it to the workspace source path. (../llm-
    tldr/tldr/cross_file_calls.py:3495)
  - That means cross-package calls across the monorepo often can’t be connected, so calls/impact will look sparse.

  2. Cross-file linking uses lossy keys

  - It reduces a module to Path(module_path).stem and joins on (stem, exported_name). (../llm-tldr/tldr/cross_file_calls.py:3559)
  - This fails badly with index.ts barrels, re-exports, and common file names, and it can also create collisions (wrong edges).

  3. Call extraction is purely syntactic

  - _extract_ts_file_calls only understands:
      - foo() as a “direct” call
      - obj.method() as an “attr” call
      - this.method() as an “intra” call (even if it’s inherited/dynamic)
  - It doesn’t do TypeScript symbol/type resolution, so it can’t reliably connect “instance.method()” to a specific class method across files/packages. (../llm-tldr/tldr/
    cross_file_calls.py:2586)

  4. Modern TS patterns are hard without the TS compiler

  - Re-exports, export * from, conditional exports, path aliases, interface merging, DI, event emitters, higher-order functions, and callbacks all obscure “who calls what”
    unless you use the TypeScript compiler API / language service.

  ### What this implies for Peerbit

  - impact/calls depend on that call graph, so if the graph can’t resolve @peerbit/* workspace imports and type-based dispatch, results will have lots of false negatives (and
    sometimes wrong positives).
  - That’s why rg often beats it for “find the real implementation” in this repo.

  ### So are impact/calls ever useful here?

  Yes, but only in narrower cases:

  - Within a single package where imports are mostly relative and functions are called directly by name.
  - For quick “intra-file” understanding (extract + local calls), not for whole-repo “who calls X” across packages.

  If you want, I can run a focused accuracy check: pick 5 functions where we already know at least one caller via rg, then compare tldrf impact/calls output to ground truth.
  That will tell us whether it’s worth recommending those commands in Agents.md for Peerbit specifically.
</Addendum>

<Conclusion>
tldrf isn’t useless, but most of what you had in Agents.md was overselling it for this repo.

  In a TypeScript monorepo like Peerbit, tldrf’s call-graph-driven commands (calls, impact, change-impact, and anything that depends on “who calls what”) can be incomplete
  because it doesn’t do real TS compiler-style symbol + workspace import resolution. That’s why rg wins for correctness and for doc/event-name audits.

  The parts of tldrf that are still useful here:

  - Daemon-backed semantic search to jump to the right files when you don’t know the exact strings/symbol names yet (then confirm with rg).
  - Dependency deep dives via isolated indexing of the installed dependency version (this is something rg can’t really replace).

  I updated Agents.md to reflect this reality:

  - rg is the default for exhaustive scans and exact matching.
  - tldrf is optional and scoped to semantic discovery + extract/import plumbing.
  - Explicit warning that calls/impact/cfg/dfg/slice/change-impact can be incomplete in this repo.
  - Kept the daemon cache-reuse instructions, with a note that tldrf semantic search CLI doesn’t reuse the daemon for embeddings.

   What’s needed is TypeScript compiler-grade resolution (module resolution + symbol/type resolution). An LSP (tsserver) is one way to access that capability,
  but the capability itself comes from the TypeScript Language Service / compiler API.

  So you have three realistic options to fix tldrf call-graph quality in a TS monorepo:

  1. Integrate with tsserver (LSP-ish approach)

  - Ask tsserver for definitions/references/signature help per call site, then build edges from that.
  - Pros: reuses battle-tested project + module resolution (paths, project references, pnpm workspace layout).
  - Cons: orchestration overhead, performance, caching, and you still need logic to turn refs into a call graph.

  2. Use TypeScript compiler API directly (no LSP)

  - Create Programs from the repo’s tsconfig.json / project references, get a TypeChecker, and resolve each CallExpression to a Signature/Symbol.
  - Pros: no daemon protocol, direct control.
  - Cons: you must correctly load all tsconfigs and handle project references, incremental builds, and performance.

  3. Stay syntax-only (what tldrf does now)

  - Fast, but will remain incomplete/wrong for cross-package calls, barrels (index.ts), re-exports, path aliases, and obj.method() dispatch.

  Even with LSP/compiler integration, call graphs won’t be perfect in JS/TS because of dynamic patterns (callbacks, event emitters, DI, polymorphism), but it would eliminate
  the big structural gaps you saw (workspace import resolution and correct symbol binding).
</Conclusion>

<FutureWork>
 Here’s a validation plan that makes it very hard to “think it works” while it’s actually missing most TS edges.

  ## 1) Define Success Criteria (Before Code)

  - Correctness: tldrf calls/impact must resolve cross-file edges across:
      - workspace package imports (@peerbit/*)
      - path aliases (compilerOptions.paths)
      - barrel re-exports (index.ts, export * from)
      - named/default exports, renamed imports
      - class methods (obj.method() where the type is known)
  - Soundness constraints: no edges for:
      - dynamic property access (obj[method]()), untyped any, eval-ish patterns
      - cases where tsserver can’t confidently resolve a symbol
  - Stability: repeated runs on the same checkout produce identical edges (order-independent, same file/func ids).
  - Performance: daemon warm usage should not regress badly:
      - incremental re-run after a single-file edit should be noticeably faster than full rebuild

  ## 2) Create a “TS Monorepo Fixture” With Known Ground Truth

  Add a small test workspace (in llm-tldr) that intentionally contains the patterns Peerbit uses:

  - packages/a exports foo via src/index.ts re-export from src/foo.ts
  - packages/b imports foo via:
      - import { foo } from "@scope/a" (workspace package import)
      - import { foo as f } from "@scope/a" (alias)
      - import FooDefault from "@scope/a" (default export)
  - paths alias case (@lib/* -> packages/a/src/*)
  - class/interface dispatch case:
      - interface I { m(): void }, class C implements I { m(){} }, const x: I = new C(); x.m() should resolve to C.m
  - callback/ref patterns (should be explicitly defined as “supported” or “not supported”):
      - const handlers = { onFoo } and handlers.onFoo() (resolvable)
      - const handlers: Record<string, Function> = ...; handlers[name]() (not resolvable)

  For this fixture, write a golden expected edge list (source file+symbol -> dest file+symbol).

  ## 3) Automated Tests (Must-Have)

  ### A. Call-graph build tests

  - Run the new tsserver-backed builder and assert:
      - it produces at least the golden edges
      - it produces no extra edges for the explicitly “not resolvable” cases

  ### B. Impact query tests

  For a few targets (e.g. foo, C.m), assert tldrf impact returns the known callers from the fixture.

  ### C. Import resolution tests

  Assert that @scope/a and @lib/* resolve to the correct physical files in the fixture.

  ## 4) Real-Repo Validation on Peerbit (Curated Checks)

  Pick ~20 “known edges” in Peerbit where the caller/callee is obvious via rg + manual inspection (one-time curation), across packages. Examples:

  - Peerbit.dial is called by bootstrap and test utils.
  - DirectStream.waitFor is called by multiple services.
  - shared-log replication event dispatchers are consumed by tests/proxies.

  Acceptance:

  - tldrf impact <symbol> must return the curated caller set with high recall (target something like 80-90% to start), and any misses must be explainable (dynamic patterns).

  ## 5) Performance/Regression Benchmarks

  Track these in CI (or at least as a script you can run consistently):

  - Full call-graph build time on the fixture
  - Incremental update time after touching 1 file in the fixture
  - For Peerbit: time to answer 5 fixed impact queries (warm daemon)

  Fail the benchmark if:

  - full build regresses > X%
  - incremental path regresses > X%
  - daemon memory/CPU spikes uncontrollably (basic sanity)

  ## 6) Negative Testing (Failure Modes)

  - tsserver missing/unavailable: ensure tldrf falls back cleanly (and marks results as incomplete).
  - multiple tsconfigs/project references: ensure it selects the right project(s) (or reports ambiguity).
  - path casing / symlinks: ensure stable canonical paths (avoid duplicate nodes).

  ## 7) Debuggability Requirements

  Add logging/trace artifacts for a failing edge:

  - “callsite location” + “tsserver resolved symbol” + “resolved definition location”
    So when an expected edge is missing, you can see whether it was:
  - parse failure
  - module resolution failure
  - symbol resolution failure
  - filtering/normalization bug

  If you want, I can turn this into concrete work items (fixture layout, exact golden edges, and the first 20 curated Peerbit edges) so validation is executable, not just
  conceptual.
  </FutureWork>