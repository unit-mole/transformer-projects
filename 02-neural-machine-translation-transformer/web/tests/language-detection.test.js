import test from 'node:test';
import assert from 'node:assert/strict';
import { detectLanguage, resolveDirection } from '../src/language-detection.js';

test('detects English', () => assert.equal(detectLanguage('This is a test sentence.').language, 'english'));
test('detects Hindi', () => assert.equal(detectLanguage('यह एक परीक्षण वाक्य है।').language, 'hindi'));
test('detects mixed scripts', () => assert.equal(detectLanguage('यह model अच्छा है').language, 'mixed'));
test('resolves automatic direction', () => assert.equal(resolveDirection('Quality matters.', 'auto').direction, 'en-hi'));
