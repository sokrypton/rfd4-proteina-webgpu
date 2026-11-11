/**
 * A GPU recording cut short is not kept (webgpu/rfd4p trunk, decoder, encoder `record`). It used to set the size it
 * was recording for first, so after a failure (a shader an iPad could not compile) the next Generate saw that size,
 * skipped recording and ran programs that were never made: "undefined is not an object (this.passPrograms[...])".
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { Rfd4pDecoderGpu } from '../webgpu/rfd4p/decoder.js';
import { Rfd4pEncoderGpu } from '../webgpu/rfd4p/encoder.js';
import { Rfd4pTrunkGpu } from '../webgpu/rfd4p/trunk.js';

/** A stand-in whose first recording fails after it has started, as a pipeline that does not compile does. */
function flaky(cls, key) {
  const self = Object.create(cls.prototype);
  self[key] = 0;
  let calls = 0;
  self._record = async function (size) {
    calls += 1;
    if (this[key] === size) return 'skipped';
    this[key] = size;
    if (calls === 1) throw new Error('pipeline failed');
    return 'recorded';
  };
  return self;
}

for (const [name, cls, key] of [['trunk', Rfd4pTrunkGpu, 'n'], ['decoder', Rfd4pDecoderGpu, 'na'], ['encoder', Rfd4pEncoderGpu, 'na']]) {
  test(`${name}: a failed recording is recorded again on the next call`, async () => {
    const m = flaky(cls, key);
    await assert.rejects(m.record(120, {}), /pipeline failed/);
    assert.equal(m[key], 0);
    assert.equal(await m.record(120, {}), 'recorded');
    assert.equal(await m.record(120, {}), 'skipped');
  });
}
