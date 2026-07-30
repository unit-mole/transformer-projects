export function nowMs(): number {
  return performance.now();
}

export function elapsedMs(start: number): number {
  return Number((performance.now() - start).toFixed(1));
}
