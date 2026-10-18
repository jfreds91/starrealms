// Loaded lazily: only local games and the replay viewer need the engine.
import init, * as engine from "./wasm/engine.js";
import wasmUrl from "./wasm/engine_bg.wasm?url";

let ready: Promise<typeof engine> | null = null;

export function loadEngine(): Promise<typeof engine> {
  ready ??= init({ module_or_path: wasmUrl }).then(() => engine);
  return ready;
}

export function randomSeed(): bigint {
  const a = new BigUint64Array(1);
  crypto.getRandomValues(a);
  return a[0];
}
