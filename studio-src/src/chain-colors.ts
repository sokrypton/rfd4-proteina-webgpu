// rfd4p: one chain palette for the viewer, the sequence bar and the RFD4-Proteina panel (window.__rfd4pChainColor).
// Keyed by chain id, not chain order, so a chain keeps its colour when others are added or removed.
const PALETTE = [
  0x3fbf8f, 0xf4a259, 0x6d9eff, 0xe56b9f, 0xd7c45a, 0xa78bfa,
  0x4fc3e8, 0xff7f6e, 0x9bc53d, 0xc9a0dc, 0x80cbc4, 0xffb74d,
];

/** Palette slot of a chain id: A, B, C… in order, anything else (e.g. "A_2" assembly copies) by hash. */
export function chainColorIndex(chainId: string): number {
  const id = String(chainId);
  if (/^[A-Z]$/.test(id)) return (id.charCodeAt(0) - 65) % PALETTE.length;
  let h = 0;
  for (let i = 0; i < id.length; i += 1) h = (h * 31 + id.charCodeAt(i)) >>> 0;
  return h % PALETTE.length;
}

export function chainColor(chainId: string): number {
  return PALETTE[chainColorIndex(chainId)];
}

export function chainColorCss(chainId: string): string {
  return `#${chainColor(chainId).toString(16).padStart(6, "0")}`;
}
