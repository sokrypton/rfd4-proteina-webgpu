/**
 * Superposition never mirrors (shared/rfd4p/output.js superimposeCoords3D): a reflection showed right-handed designs
 * as their mirror images. Handedness is read from the Cα dihedral of helical quadruplets.
 */
import test from 'node:test';
import assert from 'node:assert/strict';


/** An ideal α-helix in nm: 100° per residue, 0.15 nm rise, 0.23 nm radius; right-handed going up. */
const helix = (n) => Float32Array.from({ length: n * 3 }, (_, k) => {
  const i = Math.floor(k / 3);
  const t = (i * 100 * Math.PI) / 180;
  return [0.23 * Math.cos(t), 0.23 * Math.sin(t), 0.15 * i][k % 3];
});

/** Helical Cα quadruplets (Cα i→i+3 under maxSpan) by the sign of their dihedral: right (+50°) and left (−50°). */
function helixHandedness(ca, n, { maxSpan = 0.58, keep = null } = {}) {
  let right = 0;
  let left = 0;
  const p = (i, k) => ca[i * 3 + k];
  for (let i = 0; i + 3 < n; i += 1) {
    if (keep && !keep(i)) continue;
    const span = Math.hypot(p(i + 3, 0) - p(i, 0), p(i + 3, 1) - p(i, 1), p(i + 3, 2) - p(i, 2));
    if (!(span < maxSpan)) continue;
    const b1 = [p(i + 1, 0) - p(i, 0), p(i + 1, 1) - p(i, 1), p(i + 1, 2) - p(i, 2)];
    const b2 = [p(i + 2, 0) - p(i + 1, 0), p(i + 2, 1) - p(i + 1, 1), p(i + 2, 2) - p(i + 1, 2)];
    const b3 = [p(i + 3, 0) - p(i + 2, 0), p(i + 3, 1) - p(i + 2, 1), p(i + 3, 2) - p(i + 2, 2)];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const n1 = cross(b1, b2);
    const n2 = cross(b2, b3);
    const m = cross(n1, b2);
    const angle = (Math.atan2(-dot(m, n2) / Math.hypot(...b2), dot(n1, n2)) * 180) / Math.PI;
    if (angle > 20 && angle < 80) right += 1;
    else if (angle < -20 && angle > -80) left += 1;
  }
  return { right, left };
}


/** Mirror a Cα trace in place (x → −x). */
function mirrorTrace(ca, n) {
  for (let i = 0; i < n; i += 1) ca[i * 3] = -ca[i * 3];
  return ca;
}

test('an ideal helix is right-handed, its mirror left-handed', () => {
  const ca = helix(20);
  assert.deepEqual(helixHandedness(ca, 20), { right: 17, left: 0 });
  mirrorTrace(ca, 20);
  assert.deepEqual(helixHandedness(ca, 20), { right: 0, left: 17 });
});

test('an extended strand counts neither way', () => {
  const ca = Float32Array.from({ length: 30 }, (_, k) => (k % 3 === 0 ? 0.34 * Math.floor(k / 3) : k % 3 === 1 ? 0.05 * (Math.floor(k / 3) % 2) : 0));
  assert.deepEqual(helixHandedness(ca, 10), { right: 0, left: 0 });
});

test('superposition never mirrors: a right-handed helix stays right-handed whatever it is fitted onto', async () => {
  const { superimposeCoords3D } = await import('../shared/rfd4p/output.js');
  const mobile = helix(20);
  let rng = 7;
  const rand = () => { rng = (rng * 16807) % 2147483647; return rng / 2147483647 - 0.5; };
  for (let trial = 0; trial < 40; trial += 1) {
    // Fitted onto noise, as a design with nothing held is onto its first frame.
    const fixed = Float32Array.from({ length: 60 }, rand);
    const { rotation: R } = superimposeCoords3D(fixed, mobile);
    const det = R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
    assert.ok(det > 0, `trial ${trial}: det ${det}`);
  }
});
