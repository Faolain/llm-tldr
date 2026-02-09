export type TypedFn = (x: number) => number;

// Contextually typed function values are common in real TS code. The TS checker
// often resolves call signatures to the type-level signature node; our callgraph
// resolver must still emit a stable edge to the value symbol (`typedFn`).
export const typedFn: TypedFn = (x: number) => x + 1;

export function callTypedFn(): number {
  return typedFn(1);
}

