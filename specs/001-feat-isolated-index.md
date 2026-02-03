 **Problem**
  - When LLMs encounter an issue with an installed depdendency and need more information they have a tendency to try to search the internet or github with poor search semantics and often going on wild goose chases (adding the year in the search parameters, or github fails to return the response) and even when the response returns the web docs are incomplte or don't match the installed dependency version.
  - llm-tldr/tldr intentionally ignores heavy dependency dirs (node_modules/, .venv/) by default; indexing them wholesale is slow and produces noisy mixed results.
  - TLDR currently conflates “what you scan” with “where caches live”: indexing a dep in-place tends to write .tldr//.tldrignore into the dep directory, and semantic indexing tends to collide in a single repo-level cache unless you isolate it.

  **Solution (today)**
  - Keep one “main” index for your repo (deps excluded), then allowlist only the specific dep(s) you need via .tldrignore (e.g. prefer node_modules/* + !node_modules/pkg/** instead of ignoring node_modules/ entirely).
  - For deep “how does this library work?” questions, build a separate per-dependency index keyed by name@version and query that (often by mirroring the dep into a dedicated analysis root to avoid writing into node_modules/ / site-packages/).
  - In agent/MCP usage, prefer TLDR results first and only web-search if TLDR can’t answer.

  **Solution (feature needed in llm-tldr to make this clean/lazy without copying)**

  - Add a first-class Index concept: --scan-root (dep path) + --cache-root (repo root) + --index (module@version or hash), with all caches namespaced under repo/.tldr/ indexes/<index_id>/... (call graph + semantic FAISS + metadata + index-scoped ignore), plus daemon/MCP multi-index support and index management commands (list/info/rm/gc).


Goal
  Enable “index this dependency in-place, but store all TLDR state under the repo root” with one index per module@version, without copying.

  1. Add A First-Class “Index” Concept

  1. Define an index as {cache_root, scan_root, index_id, language(s), model, ignore_spec}.
  2. Make index_id deterministic and user-overridable (e.g. node:zod@3.23.8), and store meta.json alongside caches.

  2. Decouple Scan Root From Cache Root (Core Feature)

  1. Add global CLI flags (or equivalent env vars):
      1. --cache-root <path>: where .tldr/… lives (repo root).
      2. --scan-root <path>: what directory to analyze (dep directory).
      3. --index <id>: selects/creates an isolated namespace under cache_root.
  2. Ensure all commands accept these flags (at least: warm, semantic index/search, tree, structure, search, context, calls, impact, imports, importers).

  3. Namespaced Cache Layout

  1. Move from single global cache files to per-index directories:
      1. cache_root/.tldr/indexes/<index_id>/meta.json
      2. cache_root/.tldr/indexes/<index_id>/call_graph.json
      3. cache_root/.tldr/indexes/<index_id>/languages.json
      4. cache_root/.tldr/indexes/<index_id>/semantic/index.faiss
      5. cache_root/.tldr/indexes/<index_id>/semantic/metadata.json
  2. Keep current behavior as the default “index” for backward compatibility.

  4. Fix Semantic Indexing Root Assumptions

  1. Stop auto-walking upward to find “project root” for semantic cache placement when --cache-root/--index is provided.
  2. Make semantic build/search always read/write from the per-index semantic directory.
  3. Store and enforce the embedding model in meta.json/semantic metadata to prevent accidental mismatch.

  5. Make Ignore Specs Index-Scoped

  1. Add --ignore-file <path> and/or “index-local ignore file” support:
      1. Default to cache_root/.tldr/indexes/<index_id>/.tldrignore (not scan_root/.tldrignore).
  2. Add --use-gitignore/--no-gitignore because dependencies are often not in the repo’s gitignore context.

  6. Update Daemon + MCP To Support Multiple Indexes

  1. Include index_id (and scan_root) in daemon startup/config.
  2. Make the daemon socket path keyed by (cache_root, index_id) (simplest “one daemon per index”).
  3. Update daemon commands to accept --index and route requests to the correct index config.
  4. Update tldr-mcp server to accept --index and to set env vars/args accordingly.

  7. Add Index Management Commands (Needed For A Skill)

  1. tldr index list --cache-root …
  2. tldr index info <index_id>
  3. tldr index rm <index_id>
  4. tldr index gc (optional; based on age/size)

  8. Tests / Validation

  1. Unit tests for index path resolution (no collisions across two deps).
  2. Integration test that builds two indexes under one repo root and verifies:
      1. semantic search results differ and load the right FAISS files.
      2. warm caches don’t overwrite each other.
      3. daemon queries for --index A never read caches for --index B.

  With those features, the skill becomes straightforward: resolve dep path + version, derive index_id, then run tldr … --cache-root <repo> --scan-root <dep> --index <id>
  for warm/search without copying or web search

  Why these steps:

• - Index ID / per-module namespace: prevents “semantic search for auth” from mixing results across multiple dependencies, and prevents cache files (call graph, FAISS
    index, metadata) from overwriting each other when you index more than one package.
  - Decouple scan_root from cache_root: lets you analyze code where it lives (node_modules/foo, site-packages/bar) while keeping all TLDR state in one predictable,
    writable place (repo/.tldr/...). This avoids writing into dependency directories (often ephemeral, read-only, or managed by package managers).
  - Namespaced cache layout: makes multiple indexes coexist safely. Without this, TLDR has “one cache location per repo” semantics, so the last indexed thing wins.
  - Fix semantic “project root” assumptions: TLDR currently decides where to put semantic caches by walking up to .git/package.json. If you point it at a dependency
    inside a repo, it tends to store into the repo’s single semantic cache, which collides with other indexes. Explicit cache_root/index_id makes it deterministic.
  - Index-scoped ignore configuration: deps have very different “what matters” than your app. You want ignore rules that are specific to that dependency (and don’t rely
    on the repo’s .gitignore), while still not creating .tldrignore inside node_modules.
  - Daemon/MCP multi-index support: the “fast 100ms queries” daemon is only useful if it can serve the correct index. If it’s only keyed by project path, you can’t
    safely have zod@x and auth0@y loaded simultaneously.
  - Index management commands: a skill needs to “lazy load” and clean up. list/info/rm/gc are how you avoid silently growing .tldr/ forever and how you can detect
    “already indexed for this version”.
  - Tests: this feature is mostly about avoiding subtle collisions (wrong index loaded, wrong model used, stale cache). Tests are what keep it from regressing as TLDR
    evolves.