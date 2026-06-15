"""Core numerical and data-model primitives."""

from unfoldlab.core.kpoints import KPoint, KPointMapping, fold_kpoints_to_supercell
from unfoldlab.core.plane_waves import matching_g_mask, weights_from_coefficients
from unfoldlab.core.projections import (
    ProjectionGroup,
    ProjectionSelector,
    parse_projection_selectors,
)
from unfoldlab.core.spectral import EffectiveBandStructure
from unfoldlab.core.structures import Structure
from unfoldlab.core.transformations import TransformationMatrix, detect_transformation
from unfoldlab.core.unfolding import (
    BandUnfoldingData,
    PlaneWaveKPointData,
    WeightDiagnostics,
    compare_effective_band_structures,
    compute_plane_wave_unfolding_weights,
    diagnose_weights,
)
from unfoldlab.core.valleys import ValleyDefinition

__all__ = [
    "BandUnfoldingData",
    "EffectiveBandStructure",
    "KPoint",
    "KPointMapping",
    "PlaneWaveKPointData",
    "ProjectionGroup",
    "ProjectionSelector",
    "Structure",
    "TransformationMatrix",
    "ValleyDefinition",
    "WeightDiagnostics",
    "compare_effective_band_structures",
    "compute_plane_wave_unfolding_weights",
    "detect_transformation",
    "diagnose_weights",
    "fold_kpoints_to_supercell",
    "matching_g_mask",
    "parse_projection_selectors",
    "weights_from_coefficients",
]
