export interface I {
  m(): string;
}

export class C implements I {
  m(): string {
    return "m";
  }
}

export function dispatch(): string {
  const x: I = new C();
  return x.m();
}

