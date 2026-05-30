# Generic Projections

Projection syntax is universal and material-agnostic. The core package does not
ship built-in aliases for any element family, material family, defect chemistry,
or valley convention.

## Selector Syntax

Supported selector forms include:

- `species:<element-or-*>`
- `atom:<index-or-*>`
- `orbital:s`, `orbital:p`, `orbital:d`, `orbital:f`, or a resolved orbital
  label such as `orbital:px`, `orbital:dz2`, or `orbital:dx2-y2`
- `layer:<index-or-*>`
- `sublattice:<label-or-*>`
- `region:<name-or-*>`
- `defect_shell:<n-or-*>`
- `surface:top` or `surface:bottom`
- `interface:<name-or-*>`
- `valley:<label-or-*>`
- `spin:x`, `spin:y`, or `spin:z`
- standalone role tags: `adsorbate`, `substrate`, and `molecule`

Material-specific aliases can be supplied by user configuration as projection
groups, but they are not built into the core parser.

## User-Defined Groups

```yaml
projection_groups:
  example_d_orbitals:
    selector:
      species: ["A", "B"]
      orbitals: ["d"]
  top_layer:
    selector:
      layer: 0
  defect_neighborhood:
    selector:
      distance_from_defect: 4.0
```

These groups are labels for user intent. The exact meaning of regions, layers,
interfaces, valleys, and defect neighborhoods depends on the structures and
metadata provided by the workflow.
