from __future__ import annotations

import unmochan

EXPECTED_PUBLIC_API = {
    "EffectiveBandStructure",
    "ProjectionSelector",
    "QEUnfoldResult",
    "Structure",
    "TransformationMatrix",
    "TwistedUnfoldingProblem",
    "UnfoldingProblem",
    "VaspUnfoldResult",
    "__version__",
    "build_qe_effective_band_structure",
    "build_vasp_effective_band_structure",
    "compute_backend_weights",
    "unfold_backend_bands",
    "unfold_qe_bands",
    "unfold_vasp_bands",
}


def test_root_public_api_is_deliberately_curated() -> None:
    assert set(unmochan.__all__) == EXPECTED_PUBLIC_API
    assert all(hasattr(unmochan, name) for name in EXPECTED_PUBLIC_API)


def test_specialized_algorithms_are_not_root_exports() -> None:
    assert not hasattr(unmochan, "diagnose_augmentation")
    assert not hasattr(unmochan, "shared_weights_from_coefficients")
    assert not hasattr(unmochan, "degeneracy_report")
