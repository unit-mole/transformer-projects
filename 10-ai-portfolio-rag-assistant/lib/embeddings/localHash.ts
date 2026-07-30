import { tokenize } from "@/lib/utils/text";

function fnv1a(value: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

export function localHashEmbedding(text: string, dimension: number): number[] {
  const vector = Array.from({ length: dimension }, () => 0);
  const counts = new Map<string, number>();

  for (const token of tokenize(text)) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }

  for (const [token, count] of counts.entries()) {
    const hash = fnv1a(token);
    const index = hash % dimension;
    const sign = (hash & 1) === 0 ? 1 : -1;
    vector[index] += sign * (1 + Math.log(count));
  }

  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return norm === 0 ? vector : vector.map((value) => value / norm);
}
