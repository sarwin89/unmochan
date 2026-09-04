"""Unfolding of phonon supercells.

Why this is the same problem
----------------------------

A phonon of a supercell is a displacement pattern with one complex amplitude per
``(atom, Cartesian direction, primitive cell)``, and unfolding it onto the
primitive Brillouin zone is the *same* discrete Fourier duality as unfolding a
tight-binding state: the "orbital" index is the pair ``(atom, direction)`` and
the cell index is the same transversal of ``Z^3 / Z^3 @ T``.  So the weights,
the two sum rules, and their proofs in
``RequestProject/Unfolding/TightBinding.lean`` carry over verbatim, and this
module is a thin layer that mass-weights the force constants and turns
eigenvalues into frequencies.

Conventions
-----------

``force_constants[D][3a + i, 3b + j]`` is the force constant
``Phi_{a i, b j}(0, D)`` between direction ``i`` of atom ``a`` in the cell at
the origin and direction ``j`` of atom ``b`` in the cell at the primitive
translation ``D``.  The dynamical matrix is

``D(k)[3a + i, 3b + j] = sum_D exp(2 pi i k . D) Phi_{a i, b j}(D)
   / sqrt(m_a m_b)``,

whose eigenvalues are ``omega^2``.  Frequencies are reported as
``sign(lambda) * sqrt(|lambda|)``, so an unstable mode appears as a negative
frequency instead of vanishing into a ``NaN`` — that convention is what makes a
soft mode visible in a plot of a distorted supercell, which is one of the
things phonon unfolding is for.

Units are whatever the force constants and masses are given in; no conversion
factor is applied.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.tight_binding import (
    TightBindingModel,
    supercell_bloch_hamiltonian,
    supercell_cells,
    unfold_tight_binding_model,
    unfold_tight_binding_path,
)
from unmochan.core.transformations import TransformationMatrix

__all__ = [
    "ForceConstantModel",
    "eigenvalues_to_frequencies",
    "mass_site_scaling",
    "supercell_dynamical_matrix",
    "supercell_site_masses",
    "unfold_phonon_fiber",
    "unfold_phonon_path",
]


def eigenvalues_to_frequencies(eigenvalues: ArrayLike) -> NDArray[np.float64]:
    """``sign(lambda) sqrt(|lambda|)``: unstable modes become negative.

    The eigenvalues of a dynamical matrix are ``omega^2``, and a supercell that
    is distorted, or simply not at a local minimum, has negative ones.  Taking
    the signed square root keeps those modes on the plot instead of turning them
    into ``NaN``, which is the convention every phonon code uses.
    """

    values = np.asarray(eigenvalues, dtype=float)
    return np.sign(values) * np.sqrt(np.abs(values))


@dataclass(frozen=True)
class ForceConstantModel:
    """Primitive-cell force constants and masses.

    ``masses`` has one entry per atom of the primitive cell and
    ``force_constants`` maps an integer primitive translation ``D`` to the
    ``(3 n_atoms, 3 n_atoms)`` block ``Phi(D)``.
    """

    masses: NDArray[np.float64]
    force_constants: dict[tuple[int, int, int], NDArray[np.float64]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        masses = np.asarray(self.masses, dtype=float).reshape(-1)
        if masses.size == 0:
            raise ValueError("masses must have at least one atom")
        if np.any(masses <= 0.0):
            raise ValueError("masses must be positive")
        size = 3 * masses.size
        normalized: dict[tuple[int, int, int], NDArray[np.float64]] = {}
        for cell, block in self.force_constants.items():
            key = tuple(int(component) for component in cell)
            if len(key) != 3:
                raise ValueError("force-constant keys must be integer 3-vectors")
            matrix = np.asarray(block, dtype=float)
            if matrix.shape != (size, size):
                raise ValueError(f"force-constant blocks must have shape ({size}, {size})")
            normalized[key] = normalized.get(key, 0.0) + matrix  # type: ignore[assignment]
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "force_constants", normalized)

    @property
    def n_atoms(self) -> int:
        return int(self.masses.size)

    @property
    def n_modes(self) -> int:
        """The number of phonon branches, ``3 n_atoms``."""

        return 3 * self.n_atoms

    def symmetrized(self) -> ForceConstantModel:
        """Impose ``Phi(-D) = Phi(D)^T``, which Hermiticity of ``D(k)`` needs.

        A block listed in only one direction gets its partner at full strength;
        a pair listed in both directions is replaced by its symmetric part.  The
        operation is idempotent.
        """

        blocks: dict[tuple[int, int, int], NDArray[np.float64]] = {
            cell: np.array(block, dtype=float) for cell, block in self.force_constants.items()
        }
        for cell, block in self.force_constants.items():
            opposite = (-cell[0], -cell[1], -cell[2])
            if opposite not in blocks:
                blocks[opposite] = block.T
        symmetric = {
            cell: 0.5 * (block + blocks[(-cell[0], -cell[1], -cell[2])].T)
            for cell, block in blocks.items()
        }
        return ForceConstantModel(self.masses, symmetric)

    def acoustic_sum_rule_deviation(self) -> float:
        """``max |sum_{D, b} Phi_{a i, b j}(D)|``, which must vanish.

        Translating the whole crystal costs no energy, so the force constants of
        one atom must add up to zero over all its partners.  A force-constant
        set fitted to finite displacements never satisfies this exactly, and the
        error shows up as acoustic branches that do not go to zero at Gamma —
        the phonon analogue of a violated unfolding sum rule.
        """

        if not self.force_constants:
            return 0.0
        total = sum(self.force_constants.values())
        reshaped = np.asarray(total).reshape(self.n_atoms, 3, self.n_atoms, 3)
        return float(np.max(np.abs(reshaped.sum(axis=2))))

    def enforce_acoustic_sum_rule(self) -> ForceConstantModel:
        """Return a model with the sum rule imposed on the on-site block.

        The standard repair: the row sum of each atom is subtracted from its own
        self-interaction block ``Phi(0)``, which is the only block that can
        absorb it without changing any inter-atomic force constant.
        """

        blocks = {
            cell: np.array(block, dtype=float) for cell, block in self.force_constants.items()
        }
        size = self.n_modes
        onsite = blocks.get((0, 0, 0), np.zeros((size, size)))
        total = sum(blocks.values(), np.zeros((size, size)))
        rows = np.asarray(total).reshape(self.n_atoms, 3, self.n_atoms, 3).sum(axis=2)
        correction = np.zeros((size, size), dtype=float)
        for atom in range(self.n_atoms):
            correction[3 * atom : 3 * atom + 3, 3 * atom : 3 * atom + 3] = rows[atom]
        blocks[(0, 0, 0)] = onsite - correction
        return ForceConstantModel(self.masses, blocks)

    def mass_weighted_model(self) -> TightBindingModel:
        """The mass-weighted force constants, as a tight-binding model.

        ``Phi(D) / sqrt(m_a m_b)`` is exactly the "hopping" whose Bloch sum is
        the dynamical matrix, so the whole tight-binding machinery — supercell
        construction, unfolding, and both sum rules — applies unchanged.
        """

        inverse_root = np.repeat(1.0 / np.sqrt(self.masses), 3)
        scaling = np.outer(inverse_root, inverse_root)
        return TightBindingModel(
            self.n_modes,
            {
                cell: (block * scaling).astype(np.complex128)
                for cell, block in self.force_constants.items()
            },
        )

    def dynamical_matrix(self, kpoint: ArrayLike) -> NDArray[np.complex128]:
        """``D(k)``, the mass-weighted Bloch sum of the force constants."""

        return self.mass_weighted_model().bloch_hamiltonian(kpoint)

    def frequencies(self, kpoints: ArrayLike) -> NDArray[np.float64]:
        """Signed phonon frequencies, shape ``(n_kpoints, 3 n_atoms)``."""

        return eigenvalues_to_frequencies(self.mass_weighted_model().bands(kpoints))


def supercell_site_masses(
    model: ForceConstantModel,
    transform: TransformationMatrix,
    substitutions: Iterable[tuple[int, int, float]] | None = None,
) -> NDArray[np.float64]:
    """The ``(n_cells, n_atoms)`` mass of every atom of the supercell.

    Every cell starts from the primitive masses; ``substitutions`` are
    ``(cell_index, atom_index, mass)`` triples that replace one of them, which
    is how an isotope, a substitutional impurity, or a mass-disordered alloy
    configuration is described.  ``cell_index`` indexes
    :func:`unmochan.core.tight_binding.supercell_cells` of the same transform,
    the same ordering the on-site ``perturbations`` use.
    """

    n_cells = supercell_cells(transform).shape[0]
    masses = np.tile(model.masses.reshape(1, -1), (n_cells, 1))
    for cell_index, atom_index, mass in substitutions or ():
        cell = int(cell_index)
        atom = int(atom_index)
        if not (0 <= cell < n_cells):
            raise ValueError(f"substitution cell index {cell} is outside the supercell")
        if not (0 <= atom < model.n_atoms):
            raise ValueError(f"substitution atom index {atom} is outside the cell")
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("a substituted mass must be finite and positive")
        masses[cell, atom] = float(mass)
    return masses


def mass_site_scaling(
    model: ForceConstantModel,
    transform: TransformationMatrix,
    site_masses: ArrayLike,
) -> NDArray[np.float64]:
    """The congruence factors ``sqrt(m_primitive / m_site)`` of a mass pattern.

    The supercell dynamical matrix built from the primitive masses is
    ``D_ref = Phi / sqrt(m_a m_b)``; with site-dependent masses it is
    ``D = S D_ref S`` with ``S = diag(sqrt(m_a / m'_a))``
    (``UnfoldLab.dynMatrix_massScale``).  Returns the
    ``(n_cells, 3 n_atoms)`` table of factors, one per degree of freedom, in the
    ``3 * atom + direction`` ordering the supercell basis uses.
    """

    n_cells = supercell_cells(transform).shape[0]
    masses = np.asarray(site_masses, dtype=float)
    if masses.shape != (n_cells, model.n_atoms):
        raise ValueError(
            f"site_masses must have shape ({n_cells}, {model.n_atoms}), got {masses.shape}"
        )
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("site masses must be finite and positive")
    ratios = np.sqrt(model.masses.reshape(1, -1) / masses)
    return np.repeat(ratios, 3, axis=1)


def _scaling_or_none(
    model: ForceConstantModel,
    transform: TransformationMatrix,
    site_masses: ArrayLike | None,
) -> NDArray[np.float64] | None:
    if site_masses is None:
        return None
    return mass_site_scaling(model, transform, site_masses)


def supercell_dynamical_matrix(
    model: ForceConstantModel,
    transform: TransformationMatrix,
    supercell_kpoint: ArrayLike,
    *,
    site_masses: ArrayLike | None = None,
    perturbations: Iterable[tuple[int, int, ArrayLike]] | None = None,
) -> NDArray[np.complex128]:
    """``D(q)`` of the supercell, with optional site-dependent masses.

    Ordered as ``(cell, 3 * atom + direction)``.  A mass defect enters as the
    congruence ``D -> S D S``, *not* as an on-site term: changing one mass
    rescales a whole row and column of the matrix, and
    ``UnfoldLab.mass_change_not_diagonal_shift`` shows that no on-site shift
    reproduces it as soon as the substituted atom is coupled to anything.
    Hermiticity survives, because the congruence is real and diagonal
    (``UnfoldLab.dynMatrix_isHermitian``).

    ``perturbations`` are added *before* the mass rescaling, so they are read as
    changes of the force constants divided by the *primitive* masses; a bond
    stiffened next to a heavy isotope therefore gets the isotope's mass factor
    as it should.
    """

    matrix, _ = supercell_bloch_hamiltonian(
        model.mass_weighted_model(),
        transform,
        supercell_kpoint,
        perturbations=perturbations,
        site_scaling=_scaling_or_none(model, transform, site_masses),
    )
    return matrix


def unfold_phonon_path(
    model: ForceConstantModel,
    transform: TransformationMatrix,
    primitive_kpoints: Sequence[ArrayLike] | ArrayLike,
    *,
    site_masses: ArrayLike | None = None,
    perturbations: Iterable[tuple[int, int, ArrayLike]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unfold a phonon supercell along a path in the primitive zone.

    Returns ``(frequencies, weights)``, both of shape
    ``(n_kpoints, |det T| * 3 n_atoms)``.  A perfect crystal reproduces its
    primitive branches with weight one; a distortion, an isotope pattern, or a
    defect spreads that weight, and the total at each k-point is still
    ``3 n_atoms`` (``UnfoldLab.sum_tbWeight_states``).

    ``perturbations`` are ``(cell_index, mode_index, value)`` additions to the
    mass-weighted on-site terms, with ``mode_index = 3 * atom + direction``.

    ``site_masses`` is an optional ``(|det T|, n_atoms)`` table of masses, from
    :func:`supercell_site_masses`, describing isotopes or substituted atoms.  A
    mass defect is a congruence of the dynamical matrix rather than an on-site
    term, so it cannot be expressed through ``perturbations``; see
    :func:`supercell_dynamical_matrix`.
    """

    eigenvalues, weights = unfold_tight_binding_path(
        model.mass_weighted_model(),
        transform,
        primitive_kpoints,
        perturbations=perturbations,
        site_scaling=_scaling_or_none(model, transform, site_masses),
    )
    return eigenvalues_to_frequencies(eigenvalues), weights


def unfold_phonon_fiber(
    model: ForceConstantModel,
    transform: TransformationMatrix,
    supercell_kpoints: Sequence[ArrayLike] | ArrayLike,
    *,
    site_masses: ArrayLike | None = None,
    perturbations: Iterable[tuple[int, int, ArrayLike]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Unfold phonons of a supercell k-point onto its whole fiber.

    Returns ``(kpoints, frequencies, weights)`` exactly as
    :func:`unmochan.core.tight_binding.unfold_tight_binding_model` does, with
    the eigenvalues converted to signed frequencies.  ``site_masses`` describes
    a mass defect as in :func:`unfold_phonon_path`.
    """

    kpoints, eigenvalues, weights = unfold_tight_binding_model(
        model.mass_weighted_model(),
        transform,
        supercell_kpoints,
        perturbations=perturbations,
        site_scaling=_scaling_or_none(model, transform, site_masses),
    )
    return kpoints, eigenvalues_to_frequencies(eigenvalues), weights
