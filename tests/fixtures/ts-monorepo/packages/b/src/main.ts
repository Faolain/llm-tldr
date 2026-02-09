import FooDefault, { createCache, createLibp2pExtended, foo as f, dispatch, callSupported } from "@scope/a";
import { foo } from "@scope/a";
import { foo as fooAlias } from "@lib/foo";

export function main(): void {
  foo();
  f();
  fooAlias();
  FooDefault();
  createCache();
  createLibp2pExtended();
  dispatch();
  callSupported();
}
