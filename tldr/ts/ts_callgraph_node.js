#!/usr/bin/env node
/* eslint-disable no-console */
/*
  TypeScript compiler API callgraph builder.

  Contract: prints a single JSON payload to stdout.
*/

const fs = require("fs");
const path = require("path");
const childProcess = require("child_process");

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    if (key === "trace") {
      out.trace = true;
      continue;
    }
    const val = argv[i + 1];
    i++;
    out[key] = val;
  }
  return out;
}

function jsonOut(payload, exitCode) {
  // Important: don't call process.exit() synchronously after a large write, or
  // Node may truncate stdout (~64KB). Always exit from the write callback.
  let out = "";
  try {
    out = JSON.stringify(payload);
  } catch (e) {
    // Last-ditch: ensure we always print something.
    out = JSON.stringify({ status: "error", code: "json_serialize_failed", error: String(e) });
  }
  process.stdout.write(`${out}\n`, () => {
    process.exit(exitCode);
  });
}

function findTypescript(rootAbs) {
  // 1) Prefer repo-local TypeScript.
  try {
    const resolved = require.resolve("typescript", { paths: [rootAbs] });
    return { ts: require(resolved), resolvedPath: resolved, source: "project" };
  } catch (_) {
    // ignore
  }

  // 2) Try global npm root.
  try {
    const globalRoot = childProcess.execSync("npm root -g", { encoding: "utf8" }).trim();
    const resolved = require.resolve("typescript", { paths: [globalRoot] });
    return { ts: require(resolved), resolvedPath: resolved, source: "global" };
  } catch (_) {
    // ignore
  }

  // 3) Fall back to regular resolution (may work in dev).
  try {
    const resolved = require.resolve("typescript");
    return { ts: require(resolved), resolvedPath: resolved, source: "node" };
  } catch (e) {
    return { ts: null, resolvedPath: null, source: null, error: String(e) };
  }
}

function realAbs(p) {
  try {
    return fs.realpathSync.native(p);
  } catch (_) {
    return path.resolve(p);
  }
}

function toRel(rootAbs, fileAbs) {
  const rel = path.relative(rootAbs, fileAbs);
  return rel.split(path.sep).join("/");
}

function isUnderRoot(rootAbs, fileAbs) {
  const rel = path.relative(rootAbs, fileAbs);
  return !!rel && !rel.startsWith("..") && !path.isAbsolute(rel);
}

function isProbablyWorkspaceSource(rootAbs, fileAbs) {
  if (!isUnderRoot(rootAbs, fileAbs)) return false;
  const rel = toRel(rootAbs, fileAbs);
  if (rel.includes("/node_modules/") || rel.startsWith("node_modules/")) return false;
  if (rel.includes("/.tldr/") || rel.startsWith(".tldr/")) return false;
  if (rel.endsWith(".d.ts")) return false;
  return rel.endsWith(".ts") || rel.endsWith(".tsx") || rel.endsWith(".js") || rel.endsWith(".jsx");
}

function mapWorkspaceDistDtsToSrcTs(rootAbs, fileAbs) {
  // Deterministic + conservative mapping:
  //   **/dist/src/**/*.d.ts -> **/src/**/*.ts (only when the source file exists)
  //
  // This helps monorepos where module resolution lands on built declaration outputs,
  // but we want workspace-relative source paths in the call graph.
  if (!isUnderRoot(rootAbs, fileAbs)) return null;
  const rel = toRel(rootAbs, fileAbs);
  if (rel.includes("/node_modules/") || rel.startsWith("node_modules/")) return null;
  if (rel.includes("/.tldr/") || rel.startsWith(".tldr/")) return null;
  if (!rel.endsWith(".d.ts")) return null;

  let mappedRel = null;
  if (rel.startsWith("dist/src/")) {
    mappedRel = `src/${rel.slice("dist/src/".length)}`;
  } else if (rel.includes("/dist/src/")) {
    mappedRel = rel.replace("/dist/src/", "/src/");
  }
  if (!mappedRel) return null;
  mappedRel = mappedRel.replace(/\.d\.ts$/, ".ts");

  const candidateAbs = path.resolve(rootAbs, mappedRel);
  try {
    if (fs.existsSync(candidateAbs)) return realAbs(candidateAbs);
  } catch (_) {
    // ignore
  }
  return null;
}

function compareLoc(a, b) {
  if (a.line !== b.line) return a.line - b.line;
  return a.col - b.col;
}

function endpoint(file, symbol, line, col) {
  const out = { file, symbol };
  if (typeof line === "number") out.line = line;
  if (typeof col === "number") out.col = col;
  return out;
}

function nodeLoc(ts, sourceFile, node) {
  const start = node.getStart(sourceFile, false);
  const lc = sourceFile.getLineAndCharacterOfPosition(start);
  // 1-based for user-facing
  return { line: lc.line + 1, col: lc.character + 1 };
}

function getClassName(ts, node) {
  let p = node.parent;
  while (p) {
    if (ts.isClassDeclaration(p) || ts.isClassExpression(p)) {
      if (p.name && ts.isIdentifier(p.name)) return p.name.text;
      return null;
    }
    p = p.parent;
  }
  return null;
}

function symbolIdFromDecl(ts, decl) {
  if (!decl) return null;

  // Many real-world TS callables are exported as:
  //   export const foo = () => {}
  //   export const foo = function () {}
  // The TS checker often resolves call signatures to the arrow/function expression
  // itself. Derive a stable symbol id from the containing named declaration.
  if (ts.isArrowFunction(decl) || ts.isFunctionExpression(decl)) {
    let cur = decl;
    // Walk up a few frames to find a stable, user-facing name.
    for (let i = 0; i < 6 && cur && cur.parent; i++) {
      const p = cur.parent;

      if (ts.isVariableDeclaration(p) && p.name && ts.isIdentifier(p.name)) {
        return p.name.text;
      }

      if (ts.isPropertyDeclaration(p)) {
        if (p.name && ts.isIdentifier(p.name)) {
          const cls = getClassName(ts, p);
          if (cls) return `${cls}.${p.name.text}`;
          return p.name.text;
        }
        return null;
      }

      if (ts.isPropertyAssignment(p)) {
        const n = p.name;
        if (n && ts.isIdentifier(n)) return n.text;
        if (n && ts.isStringLiteral(n)) return n.text;
        if (n && ts.isNumericLiteral(n)) return n.text;
        return null;
      }

      if (ts.isExportAssignment(p)) {
        return "default";
      }

      // Unwrap common wrappers that can sit between the function-like node and
      // its named container (e.g., parentheses, type assertions).
      const isTypeAssertion =
        typeof ts.isTypeAssertionExpression === "function" && ts.isTypeAssertionExpression(p);
      const isNonNull =
        typeof ts.isNonNullExpression === "function" && ts.isNonNullExpression(p);
      const isSatisfies =
        typeof ts.isSatisfiesExpression === "function" && ts.isSatisfiesExpression(p);
      if (ts.isParenthesizedExpression(p) || ts.isAsExpression(p) || isTypeAssertion || isNonNull || isSatisfies) {
        cur = p;
        continue;
      }

      break;
    }

    if (ts.isFunctionExpression(decl) && decl.name && ts.isIdentifier(decl.name)) {
      return decl.name.text;
    }

    return null;
  }

  if (ts.isFunctionDeclaration(decl)) {
    return decl.name && ts.isIdentifier(decl.name) ? decl.name.text : null;
  }

  if (ts.isMethodDeclaration(decl)) {
    if (!decl.name) return null;
    if (!ts.isIdentifier(decl.name)) return null;
    const cls = getClassName(ts, decl);
    if (!cls) return decl.name.text;
    return `${cls}.${decl.name.text}`;
  }

  if (ts.isConstructorDeclaration(decl)) {
    const cls = getClassName(ts, decl);
    if (!cls) return null;
    return `${cls}.constructor`;
  }

  if (ts.isVariableDeclaration(decl)) {
    return ts.isIdentifier(decl.name) ? decl.name.text : null;
  }

  if (ts.isPropertyDeclaration(decl)) {
    if (!decl.name || !ts.isIdentifier(decl.name)) return null;
    const cls = getClassName(ts, decl);
    if (!cls) return decl.name.text;
    return `${cls}.${decl.name.text}`;
  }

  return null;
}

function isConstVariableDecl(ts, varDecl) {
  // varDecl is VariableDeclaration
  const list = varDecl.parent;
  if (!list || !ts.isVariableDeclarationList(list)) return false;
  return (list.flags & ts.NodeFlags.Const) !== 0;
}

function tryResolveInterfaceDispatch(ts, checker, receiverIdent, methodName) {
  const sym = checker.getSymbolAtLocation(receiverIdent);
  if (!sym) return null;
  const decl = sym.valueDeclaration || (sym.declarations && sym.declarations[0]);
  if (!decl || !ts.isVariableDeclaration(decl)) return null;
  if (!isConstVariableDecl(ts, decl)) return null;
  const init = decl.initializer;
  if (!init || !ts.isNewExpression(init)) return null;

  const initType = checker.getTypeAtLocation(init);
  if (!initType) return null;

  const prop = checker.getPropertyOfType(initType, methodName);
  if (!prop) return null;
  const propDecl = prop.valueDeclaration || (prop.declarations && prop.declarations[0]);
  if (!propDecl) return null;
  return propDecl;
}

function resolveCalleeDecl(ts, checker, callExpr) {
  const expr = callExpr.expression;

  // Disallow dynamic property access.
  if (ts.isElementAccessExpression(expr)) {
    return { decl: null, reason: "dynamic_element_access" };
  }

  // Only handle identifier and property access calls (incl. optional chain).
  let sym = null;
  let propAccess = null;
  if (ts.isIdentifier(expr)) {
    sym = checker.getSymbolAtLocation(expr);
  } else if (ts.isPropertyAccessExpression(expr) || ts.isPropertyAccessChain(expr)) {
    propAccess = expr;
    if (expr.name && ts.isIdentifier(expr.name)) {
      sym = checker.getSymbolAtLocation(expr.name);
    }
  } else {
    return { decl: null, reason: "unsupported_callee_expression" };
  }

  // If callee is `any`, skip (unsound).
  try {
    const t = checker.getTypeAtLocation(expr);
    if (t && (t.flags & ts.TypeFlags.Any) !== 0) {
      return { decl: null, reason: "any_dispatch" };
    }
  } catch (_) {
    // ignore
  }

  // Use signature as a strong hint when available.
  let sigDecl = null;
  try {
    const sig = checker.getResolvedSignature(callExpr);
    if (sig) sigDecl = sig.getDeclaration();
  } catch (_) {
    // ignore
  }

  if (sigDecl) {
    // Prefer concrete implementation declarations.
    if (ts.isFunctionDeclaration(sigDecl) || ts.isMethodDeclaration(sigDecl) || ts.isVariableDeclaration(sigDecl)) {
      return { decl: sigDecl, reason: null };
    }
  }

  // Follow symbol (de-alias).
  if (sym && (sym.flags & ts.SymbolFlags.Alias) !== 0) {
    try {
      sym = checker.getAliasedSymbol(sym);
    } catch (_) {
      // ignore
    }
  }

  let decl = null;
  if (sym) {
    decl = sym.valueDeclaration || (sym.declarations && sym.declarations[0]) || null;
  }

  // If we landed on an object-literal property, try to follow the initializer.
  if (decl && ts.isShorthandPropertyAssignment(decl)) {
    try {
      const vs = checker.getShorthandAssignmentValueSymbol(decl);
      if (vs) {
        let vs2 = vs;
        if ((vs2.flags & ts.SymbolFlags.Alias) !== 0) {
          vs2 = checker.getAliasedSymbol(vs2);
        }
        const vd = vs2.valueDeclaration || (vs2.declarations && vs2.declarations[0]);
        if (vd) return { decl: vd, reason: null };
      }
    } catch (_) {
      // ignore
    }
  }

  if (decl && ts.isPropertyAssignment(decl)) {
    const init = decl.initializer;
    if (init && ts.isIdentifier(init)) {
      const vs = checker.getSymbolAtLocation(init);
      if (vs) {
        let vs2 = vs;
        if ((vs2.flags & ts.SymbolFlags.Alias) !== 0) {
          vs2 = checker.getAliasedSymbol(vs2);
        }
        const vd = vs2.valueDeclaration || (vs2.declarations && vs2.declarations[0]);
        if (vd) return { decl: vd, reason: null };
      }
    }
  }

  // If signature/symbol is interface-only, attempt a safe const-new narrowing.
  if (
    propAccess &&
    (decl && (ts.isMethodSignature(decl) || ts.isPropertySignature(decl)))
  ) {
    const recv = propAccess.expression;
    if (ts.isIdentifier(recv) && ts.isIdentifier(propAccess.name)) {
      const implDecl = tryResolveInterfaceDispatch(ts, checker, recv, propAccess.name.text);
      if (implDecl) return { decl: implDecl, reason: null };
    }
  }

  // As a last resort, use the signature declaration if present.
  if (sigDecl) {
    if (ts.isMethodSignature(sigDecl) || ts.isPropertySignature(sigDecl)) {
      return { decl: null, reason: "interface_signature_only" };
    }
    // Only prefer signature declarations when they can be named stably; otherwise
    // fall back to the symbol's declaration (which is often the variable/property).
    if (symbolIdFromDecl(ts, sigDecl)) {
      return { decl: sigDecl, reason: null };
    }
  }

  if (!decl) return { decl: null, reason: "unresolved_symbol" };
  if (ts.isMethodSignature(decl) || ts.isPropertySignature(decl)) {
    return { decl: null, reason: "interface_signature_only" };
  }

  return { decl, reason: null };
}

function getCallerContext(ts, node) {
  // Walk up to the nearest named callable.
  let cur = node;
  while (cur) {
    if (ts.isFunctionDeclaration(cur) && cur.name && ts.isIdentifier(cur.name)) {
      return { decl: cur, symbol: cur.name.text };
    }

    if (ts.isMethodDeclaration(cur) && cur.name && ts.isIdentifier(cur.name)) {
      const cls = getClassName(ts, cur);
      const sym = cls ? `${cls}.${cur.name.text}` : cur.name.text;
      return { decl: cur, symbol: sym };
    }

    if (ts.isConstructorDeclaration(cur)) {
      const cls = getClassName(ts, cur);
      if (cls) return { decl: cur, symbol: `${cls}.constructor` };
    }

    // const foo = () => {} / function() {}
    if (ts.isArrowFunction(cur) || ts.isFunctionExpression(cur)) {
      const p = cur.parent;
      if (p && ts.isVariableDeclaration(p) && ts.isIdentifier(p.name)) {
        return { decl: p, symbol: p.name.text };
      }
    }

    cur = cur.parent;
  }

  return { decl: null, symbol: "<module>" };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const root = args.root ? realAbs(args.root) : null;
  const tsconfig = args.tsconfig ? realAbs(args.tsconfig) : null;
  const allowlistPath = args.allowlist ? realAbs(args.allowlist) : null;
  const trace = !!args.trace;
  const traceLimit = trace ? 2000 : 0;

  if (!root || !tsconfig) {
    jsonOut({ status: "error", code: "bad_args", error: "Missing --root or --tsconfig" }, 2);
    return;
  }

  const tsRes = findTypescript(root);
  if (!tsRes.ts) {
    jsonOut(
      {
        status: "error",
        code: "typescript_missing",
        error: "TypeScript module not found (install typescript in the project or globally).",
        details: tsRes.error,
      },
      3,
    );
    return;
  }
  const ts = tsRes.ts;

  let allowSet = null;
  if (allowlistPath) {
    try {
      const raw = JSON.parse(fs.readFileSync(allowlistPath, "utf8"));
      if (Array.isArray(raw)) {
        allowSet = new Set(raw.map((p) => realAbs(String(p))));
      }
    } catch (_) {
      // ignore
    }
  }

  let configFile;
  try {
    configFile = ts.readConfigFile(tsconfig, ts.sys.readFile);
    if (configFile.error) {
      jsonOut({ status: "error", code: "tsconfig_read_failed", error: String(configFile.error.messageText) }, 4);
      return;
    }
  } catch (e) {
    jsonOut({ status: "error", code: "tsconfig_read_failed", error: String(e) }, 4);
    return;
  }

  const configDir = path.dirname(tsconfig);
  let parsed;
  try {
    parsed = ts.parseJsonConfigFileContent(
      configFile.config,
      ts.sys,
      configDir,
      undefined,
      tsconfig,
    );
  } catch (e) {
    jsonOut({ status: "error", code: "tsconfig_parse_failed", error: String(e) }, 5);
    return;
  }

  let program;
  try {
    program = ts.createProgram({
      rootNames: parsed.fileNames,
      options: parsed.options,
      projectReferences: parsed.projectReferences,
    });
  } catch (e) {
    jsonOut({ status: "error", code: "program_create_failed", error: String(e) }, 6);
    return;
  }

  let checker;
  try {
    checker = program.getTypeChecker();
  } catch (e) {
    jsonOut({ status: "error", code: "checker_failed", error: String(e) }, 7);
    return;
  }

  const edgesByKey = new Map();
  const skipped = [];
  let skippedCount = 0;
  let processedFiles = 0;

  function maybeAddEdge(edge) {
    const key = `${edge.caller.file}\t${edge.caller.symbol}\t${edge.callee.file}\t${edge.callee.symbol}`;
    const existing = edgesByKey.get(key);
    if (!existing) {
      edgesByKey.set(key, edge);
      return;
    }

    // Keep the smallest callsite (deterministic).
    if (edge.callsite && existing.callsite) {
      if (compareLoc(edge.callsite, existing.callsite) < 0) {
        edgesByKey.set(key, edge);
      }
    } else if (edge.callsite && !existing.callsite) {
      edgesByKey.set(key, edge);
    }
  }

  function walk(node, sourceFile) {
    if (ts.isCallExpression(node)) {
      const callerCtx = getCallerContext(ts, node);

      const callee = resolveCalleeDecl(ts, checker, node);
      if (!callee.decl) {
        if (trace) {
          const loc = nodeLoc(ts, sourceFile, node);
          skippedCount++;
          if (skipped.length < traceLimit) {
            skipped.push({
              callsite: { file: toRel(root, sourceFile.fileName), line: loc.line, col: loc.col },
              reason: callee.reason || "unresolved",
            });
          }
        }
      } else {
        const calleeDecl = callee.decl;
        const calleeFile = calleeDecl.getSourceFile();
        const calleeAbs = realAbs(calleeFile.fileName);

        // If a call resolves into a workspace declaration output, prefer mapping the
        // callee path back to the workspace source path when possible.
        let calleeOutAbs = calleeAbs;
        let calleeMapped = false;
        if (calleeAbs.endsWith(".d.ts")) {
          const mapped = mapWorkspaceDistDtsToSrcTs(root, calleeAbs);
          if (mapped) {
            calleeOutAbs = mapped;
            calleeMapped = true;
          }
        }

        if (!isProbablyWorkspaceSource(root, calleeOutAbs)) {
          if (trace) {
            const loc = nodeLoc(ts, sourceFile, node);
            skippedCount++;
            if (skipped.length < traceLimit) {
              skipped.push({
                callsite: { file: toRel(root, sourceFile.fileName), line: loc.line, col: loc.col },
                reason: "callee_outside_workspace",
              });
            }
          }
        } else {
          const calleeId = symbolIdFromDecl(ts, calleeDecl);
          if (!calleeId) {
            if (trace) {
              const loc = nodeLoc(ts, sourceFile, node);
              skippedCount++;
              if (skipped.length < traceLimit) {
                skipped.push({
                  callsite: { file: toRel(root, sourceFile.fileName), line: loc.line, col: loc.col },
                  reason: "callee_unnamed",
                });
              }
            }
          } else {
            const callerDecl = callerCtx.decl;
            let callerLineCol = null;
            if (callerDecl) callerLineCol = nodeLoc(ts, callerDecl.getSourceFile(), callerDecl);
            const calleeLineCol = calleeMapped ? null : nodeLoc(ts, calleeFile, calleeDecl);
            const callsiteLineCol = nodeLoc(ts, sourceFile, node);

            const edge = {
              caller: endpoint(
                toRel(root, sourceFile.fileName),
                callerCtx.symbol,
                callerLineCol ? callerLineCol.line : undefined,
                callerLineCol ? callerLineCol.col : undefined,
              ),
              callee: endpoint(
                toRel(root, calleeOutAbs),
                calleeId,
                calleeLineCol ? calleeLineCol.line : undefined,
                calleeLineCol ? calleeLineCol.col : undefined,
              ),
              callsite: endpoint(
                toRel(root, sourceFile.fileName),
                "",
                callsiteLineCol.line,
                callsiteLineCol.col,
              ),
            };
            maybeAddEdge(edge);
          }
        }
      }
    }

    ts.forEachChild(node, (child) => walk(child, sourceFile));
  }

  for (const sf of program.getSourceFiles()) {
    const abs = realAbs(sf.fileName);
    if (!isProbablyWorkspaceSource(root, abs)) continue;
    if (allowSet && !allowSet.has(abs)) continue;
    processedFiles++;
    walk(sf, sf);
  }

  if (allowSet && allowSet.size > 0 && processedFiles === 0) {
    jsonOut(
      {
        status: "error",
        code: "tsconfig_no_workspace_inputs",
        error:
          "tsconfig produced no workspace inputs (processed_files=0 while an allowlist was provided).",
        meta: {
          resolver: "ts-compiler-api",
          root,
          tsconfig,
          typescript_source: tsRes.source,
          typescript_path: tsRes.resolvedPath,
          typescript_version: typeof ts.version === "string" ? ts.version : null,
          processed_files: processedFiles,
          allowlist_count: allowSet.size,
        },
      },
      10,
    );
    return;
  }

  const edges = Array.from(edgesByKey.values());
  edges.sort((a, b) => {
    const ak = `${a.caller.file}\t${a.caller.symbol}\t${a.callee.file}\t${a.callee.symbol}\t${a.callsite.line}\t${a.callsite.col}`;
    const bk = `${b.caller.file}\t${b.caller.symbol}\t${b.callee.file}\t${b.callee.symbol}\t${b.callsite.line}\t${b.callsite.col}`;
    return ak < bk ? -1 : ak > bk ? 1 : 0;
  });

  const meta = {
    resolver: "ts-compiler-api",
    root,
    tsconfig,
    typescript_source: tsRes.source,
    typescript_path: tsRes.resolvedPath,
    typescript_version: typeof ts.version === "string" ? ts.version : null,
    processed_files: processedFiles,
    allowlist_count: allowSet ? allowSet.size : null,
  };
  if (trace) {
    skipped.sort((a, b) => {
      const af = (a.callsite && a.callsite.file) || "";
      const bf = (b.callsite && b.callsite.file) || "";
      if (af !== bf) return af < bf ? -1 : 1;
      const al = (a.callsite && a.callsite.line) || 0;
      const bl = (b.callsite && b.callsite.line) || 0;
      if (al !== bl) return al - bl;
      const ac = (a.callsite && a.callsite.col) || 0;
      const bc = (b.callsite && b.callsite.col) || 0;
      if (ac !== bc) return ac - bc;
      const ar = a.reason || "";
      const br = b.reason || "";
      if (ar !== br) return ar < br ? -1 : 1;
      return 0;
    });
    meta.skipped_count = skippedCount;
    meta.skipped_limit = traceLimit;
    meta.skipped_truncated = skippedCount > skipped.length;
  }

  jsonOut(
    {
      status: "ok",
      meta,
      edges,
      skipped: trace ? skipped : undefined,
    },
    0,
  );
}

main();
