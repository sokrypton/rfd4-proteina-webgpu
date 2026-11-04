/**
 * The condition vocabulary is what makes an annotated CIF the whole input: every condition the model understands
 * has to be reachable by name, with a stable column name, or a document can carry a condition nothing reads back.
 * Nothing covered ALL_CONDITIONS before this, so a condition could be added to the list and never wired.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import { ALL_CONDITIONS, resolveConditionSpecName } from '../shared/rfd4p/conditions.js';

test('every condition resolves by its alias and by its full column name', () => {
  assert.ok(ALL_CONDITIONS.length >= 19, `expected the full vocabulary, got ${ALL_CONDITIONS.length}`);
  for (const condition of ALL_CONDITIONS) {
    assert.ok(condition.alias, `a condition has no alias: ${JSON.stringify(condition)}`);
    assert.ok(condition.full_name, `${condition.alias} has no full_name`);
    assert.equal(resolveConditionSpecName(condition.alias).full_name, condition.full_name,
      `${condition.alias} does not resolve by alias`);
    assert.equal(resolveConditionSpecName(condition.full_name).alias, condition.alias,
      `${condition.full_name} does not resolve by full name`);
  }
});

test('aliases and column names are unique, so neither shadows another', () => {
  const aliases = ALL_CONDITIONS.map((c) => c.alias);
  const columns = ALL_CONDITIONS.map((c) => c.full_name);
  assert.equal(new Set(aliases).size, aliases.length,
    `duplicate alias: ${aliases.filter((a, i) => aliases.indexOf(a) !== i)}`);
  assert.equal(new Set(columns).size, columns.length,
    `duplicate column: ${columns.filter((c, i) => columns.indexOf(c) !== i)}`);
});

test('column names follow the annotated-CIF convention, and enums carry values', () => {
  for (const condition of ALL_CONDITIONS) {
    // Deliberately loose in the middle: the upstream vocabulary is not uniformly lower-case
    // (`condition_InternalCoordinate_1_atom` is CamelCase and is the real column, used throughout output.js).
    // What has to hold is the prefix and the level suffix, which is what readers key on.
    assert.match(condition.full_name, /^condition_\S+_(atom|residue|token)$/,
      `${condition.alias} has an off-convention column name: ${condition.full_name}`);
    if (condition.enum !== undefined) {
      assert.ok(Object.keys(condition.enum).length > 0, `${condition.alias} has an empty enum`);
    }
  }
});

test('an unknown condition name is rejected rather than silently ignored', () => {
  assert.throws(() => resolveConditionSpecName('condition_not_a_real_one'), /Unknown condition name/);
  assert.throws(() => resolveConditionSpecName(''), /Unknown condition name/);
});
