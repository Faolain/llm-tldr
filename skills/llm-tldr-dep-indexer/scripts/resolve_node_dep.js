#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const NATIVE_EXTS = new Set(['.node', '.wasm']);

function parseArgs(argv) {
  const args = { pkg: null, from: null };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (!args.pkg && !arg.startsWith('--')) {
      args.pkg = arg;
      continue;
    }
    if (arg === '--from') {
      args.from = argv[i + 1];
      i++;
      continue;
    }
  }
  return args;
}

function resolvePackageJson(pkg, fromDir) {
  const opts = fromDir ? { paths: [fromDir] } : undefined;
  return require.resolve(`${pkg}/package.json`, opts);
}

function scanNative(root, maxFiles = 20000) {
  let nativeCount = 0;
  let scanned = 0;
  let truncated = false;

  const stack = [root];
  while (stack.length) {
    const current = stack.pop();
    let entries;
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
        continue;
      }
      scanned += 1;
      if (scanned > maxFiles) {
        truncated = true;
        return { nativeCount, truncated };
      }
      const ext = path.extname(entry.name).toLowerCase();
      if (NATIVE_EXTS.has(ext)) {
        nativeCount += 1;
      }
    }
  }
  return { nativeCount, truncated };
}

function main() {
  const args = parseArgs(process.argv);
  if (!args.pkg) {
    console.error('Usage: resolve_node_dep.js <pkg> [--from <dir>]');
    process.exit(1);
  }

  const pkgJsonPath = resolvePackageJson(args.pkg, args.from);
  const pkgRoot = path.dirname(pkgJsonPath);
  const pkgJson = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));

  const hasSrcDir = fs.existsSync(path.join(pkgRoot, 'src'));
  const hasDistDir = fs.existsSync(path.join(pkgRoot, 'dist'));

  const { nativeCount, truncated } = scanNative(pkgRoot);
  const hasNativeExt = nativeCount > 0;

  const result = {
    name: pkgJson.name,
    version: pkgJson.version,
    package_json_path: pkgJsonPath,
    package_root: pkgRoot,
    entrypoints: {
      main: pkgJson.main || null,
      module: pkgJson.module || null,
      types: pkgJson.types || pkgJson.typings || null,
      exports: pkgJson.exports || null,
    },
    has_src_dir: hasSrcDir,
    has_dist_dir: hasDistDir,
    has_native_or_wasm: hasNativeExt,
    native_file_count: nativeCount,
    scan_truncated: truncated,
  };

  console.log(JSON.stringify(result, null, 2));
}

main();
