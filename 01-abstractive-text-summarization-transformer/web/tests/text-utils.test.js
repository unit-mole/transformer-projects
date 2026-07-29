import test from 'node:test';
import assert from 'node:assert/strict';
import {
  compressionRatio,
  countCharacters,
  countWords,
  normalizeText,
  splitIntoSentences,
  validateGenerationSettings,
} from '../src/text-utils.js';

test('normalizeText preserves Unicode facts and removes repeated whitespace', () => {
  assert.equal(normalizeText('  José\treported 42 cases.  '), 'José reported 42 cases.');
});

test('word and character counts handle empty and populated strings', () => {
  assert.equal(countWords(''), 0);
  assert.equal(countWords('one two three'), 3);
  assert.equal(countCharacters('A B'), 3);
});

test('compressionRatio uses summary words divided by source words', () => {
  assert.equal(compressionRatio('one two three four', 'one two'), 0.5);
});

test('splitIntoSentences returns usable sentence chunks', () => {
  assert.deepEqual(splitIntoSentences('First sentence. Second sentence!'), ['First sentence.', 'Second sentence!']);
});

test('validateGenerationSettings accepts the portfolio defaults', () => {
  assert.deepEqual(validateGenerationSettings({
    minNewTokens: 30,
    maxNewTokens: 120,
    numBeams: 4,
    lengthPenalty: 2,
    noRepeatNgramSize: 3,
  }), {
    minNewTokens: 30,
    maxNewTokens: 120,
    numBeams: 4,
    lengthPenalty: 2,
    noRepeatNgramSize: 3,
    earlyStopping: true,
  });
});

test('validateGenerationSettings rejects invalid length ranges', () => {
  assert.throws(() => validateGenerationSettings({
    minNewTokens: 120,
    maxNewTokens: 100,
    numBeams: 4,
    lengthPenalty: 2,
    noRepeatNgramSize: 3,
  }), /greater than minimum/);
});
