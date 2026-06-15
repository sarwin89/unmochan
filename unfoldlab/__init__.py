"""UnfoldLab public API."""

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
from unfoldlab.io.qe import (
    QEKMap,
    QEPath,
    build_qe_path,
    qe_effective_band_structure,
    read_pw_input_structure,
)
from unfoldlab.twist import TwistedUnfoldingProblem
from unfoldlab.workflows.backend import (
    compute_backend_weights,
    unfold_backend_bands,
    write_backend_path_files,
)
from unfoldlab.workflows.problem import UnfoldingProblem
from unfoldlab.workflows.qe import (
    QEUnfoldResult,
    build_qe_effective_band_structure,
    unfold_qe_bands,
)
from unfoldlab.workflows.vasp import (
    VaspUnfoldResult,
    build_vasp_effective_band_structure,
    unfold_vasp_bands,
)

__all__ = [
    "EffectiveBandStructure",
    "BandUnfoldingData",
    "KPoint",
    "KPointMapping",
    "PlaneWaveKPointData",
    "QEKMap",
    "QEPath",
    "QEUnfoldResult",
    "ProjectionGroup",
    "ProjectionSelector",
    "Structure",
    "TransformationMatrix",
    "TwistedUnfoldingProblem",
    "UnfoldingProblem",
    "ValleyDefinition",
    "VaspUnfoldResult",
    "WeightDiagnostics",
    "build_qe_effective_band_structure",
    "build_qe_path",
    "build_vasp_effective_band_structure",
    "compare_effective_band_structures",
    "compute_backend_weights",
    "compute_plane_wave_unfolding_weights",
    "detect_transformation",
    "diagnose_weights",
    "fold_kpoints_to_supercell",
    "matching_g_mask",
    "parse_projection_selectors",
    "qe_effective_band_structure",
    "read_pw_input_structure",
    "unfold_backend_bands",
    "unfold_qe_bands",
    "unfold_vasp_bands",
    "weights_from_coefficients",
    "write_backend_path_files",
]

__version__ = "0.1.0"
