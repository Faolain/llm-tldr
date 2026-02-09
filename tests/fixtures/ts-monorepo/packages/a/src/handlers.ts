export function onFoo(): string {
  return "onFoo";
}

// Supported: object-literal property shorthand, then property call.
export const handlers = { onFoo };

export function callSupported(): string {
  return handlers.onFoo();
}

// Not supported: dynamic element access call.
export const dynHandlers: Record<string, Function> = { onFoo };

export function callUnsupported(name: string): unknown {
  return dynHandlers[name]();
}

