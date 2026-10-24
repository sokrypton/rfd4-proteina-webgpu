// rfd4p: Mol* colour theme using the shared chain palette (src/chain-colors.ts) instead of Mol*'s order-based one.
// Carbons (and the cartoon, traced through carbon atoms) take their chain's colour; with `carbonOnly`, other
// elements keep standard element colours, otherwise every atom takes the chain colour.
import { Bond, StructureElement, StructureProperties } from "molstar/lib/mol-model/structure";
import type { Location } from "molstar/lib/mol-model/location";
import type { ColorTheme } from "molstar/lib/mol-theme/color";
import { ElementSymbolColors, elementSymbolColor } from "molstar/lib/mol-theme/color/element-symbol";
import type { ThemeDataContext } from "molstar/lib/mol-theme/theme";
import { Color } from "molstar/lib/mol-util/color";
import { ParamDefinition as PD } from "molstar/lib/mol-util/param-definition";
import { chainColor } from "../chain-colors";

const Params = { carbonOnly: PD.Boolean(true) };
type Params = typeof Params;

function StudioChainColorTheme(_ctx: ThemeDataContext, props: PD.Values<Params>): ColorTheme<Params> {
  const cache = new Map<string, Color>();
  const byChain = (loc: StructureElement.Location): Color => {
    // label_asym_id: the chain id the document, sequence bar and design panel use (auth ids can differ in RCSB files).
    const id = StructureProperties.chain.label_asym_id(loc);
    let c = cache.get(id);
    if (c === undefined) cache.set(id, (c = Color(chainColor(id))));
    return c;
  };
  const atom = (loc: StructureElement.Location): Color => {
    if (!props.carbonOnly) return byChain(loc);
    const symbol = StructureProperties.atom.type_symbol(loc);
    return symbol === "C" ? byChain(loc) : elementSymbolColor(ElementSymbolColors, symbol);
  };
  const l = StructureElement.Location.create();
  const color = (location: Location): Color => {
    if (StructureElement.Location.is(location)) return atom(location);
    if (Bond.isLocation(location)) {
      l.structure = location.aStructure;
      l.unit = location.aUnit;
      l.element = location.aUnit.elements[location.aIndex];
      return atom(l);
    }
    return Color(0xbbbbbb);
  };
  return {
    factory: StudioChainColorTheme,
    granularity: "group",
    color,
    props,
    description: "Chain colours shared with the sequence bar and the design panel.",
  };
}

export const StudioChainColorThemeProvider: ColorTheme.Provider<Params, "studio-chain"> = {
  name: "studio-chain",
  label: "Chain (Studio)",
  category: "Chain Property",
  factory: StudioChainColorTheme,
  getParams: () => Params,
  defaultValues: PD.getDefaultValues(Params),
  isApplicable: (ctx: ThemeDataContext) => !!ctx.structure,
};
