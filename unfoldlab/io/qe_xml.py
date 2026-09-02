"""Reader for the Quantum ESPRESSO ``data-file-schema.xml`` of a ``.save``.

The XML that ``pw.x`` writes next to the wavefunctions carries everything the
unfolding machinery previously had to be told by hand:

* the cell and ``alat``, hence QE's ``at`` matrix, which is what turns the
  ``xk`` recorded in a wavefunction file into fractional reciprocal
  coordinates (``UnfoldLab.xkToFrac_of_reciprocal``).  The cell is written in
  bohr and ``alat`` in bohr, and ``at = cell / alat`` is dimensionless, so the
  unit the rest of the library works in is irrelevant
  (``UnfoldLab.latticeAlat_unit_invariant``);
* the k-point list in the same order as the ``wfc*`` files, so the file order
  can be checked rather than assumed;
* the Kohn-Sham eigenvalues, so a separate bands file is optional;
* ``uspp`` / ``paw`` flags, which say whether the plane-wave weights are those
  of a pseudo wavefunction rather than the all-electron one.

Only the elements listed above are parsed; anything else in the document is
ignored, and a missing optional element yields ``None`` rather than an error.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from numpy.typing import NDArray

#: Hartree in electronvolt (CODATA 2018), the unit QE writes energies in.
HARTREE_EV = 27.211386245988

#: Bohr radius in angstrom (CODATA 2018).
BOHR_ANGSTROM = 0.529177210903

XML_BASENAMES = ("data-file-schema.xml", "data-file.xml")


class QEXMLError(ValueError):
    """Raised when a QE XML document cannot be interpreted."""


@dataclass(frozen=True)
class QESaveMetadata:
    """Everything unfoldlab reads from a QE ``data-file-schema.xml``."""

    path: Path
    alat_bohr: float
    #: Direct lattice in bohr, rows = lattice vectors.
    lattice_bohr: NDArray[np.float64]
    #: QE's ``at``: the direct lattice in units of ``alat`` (dimensionless).
    lattice_alat: NDArray[np.float64]
    #: QE's ``bg`` if the document records it, rows in units of ``2*pi/alat``.
    reciprocal_alat: NDArray[np.float64] | None
    #: Cartesian k-points in units of ``2*pi/alat``, in wavefunction-file order.
    kpoints_cart_alat: NDArray[np.float64]
    #: The same k-points in fractional reciprocal coordinates.
    kpoints_frac: NDArray[np.float64]
    kpoint_weights: NDArray[np.float64]
    #: Kohn-Sham eigenvalues in eV, shape ``(nks, n_values)``.  For a
    #: spin-polarized (LSDA) calculation ``n_values`` is ``nbnd_up + nbnd_dw``.
    eigenvalues_ev: NDArray[np.float64]
    occupations: NDArray[np.float64] | None
    nbnd: int
    nbnd_up: int | None
    nbnd_dw: int | None
    n_electrons: float | None
    lsda: bool
    noncolin: bool
    spinorbit: bool
    gamma_only: bool
    uspp: bool
    paw: bool
    fermi_energy_ev: float | None
    #: ``(species name, pseudopotential file)`` pairs.
    species: tuple[tuple[str, str], ...]
    n_atoms: int | None

    @property
    def n_kpoints(self) -> int:
        return int(self.kpoints_cart_alat.shape[0])

    @property
    def lattice_angstrom(self) -> NDArray[np.float64]:
        """Direct lattice in angstrom, rows = lattice vectors."""

        return self.lattice_bohr * BOHR_ANGSTROM

    def eigenvalues_by_spin(self) -> tuple[NDArray[np.float64], ...]:
        """Eigenvalues split per spin channel.

        Returns a one-element tuple for an unpolarized or noncollinear run and
        a ``(up, down)`` pair for LSDA.
        """

        if not self.lsda:
            return (self.eigenvalues_ev,)
        up = self.nbnd_up if self.nbnd_up is not None else self.eigenvalues_ev.shape[1] // 2
        return (self.eigenvalues_ev[:, :up], self.eigenvalues_ev[:, up:])

    def pseudo_warning(self) -> str | None:
        """Message about pseudo-wavefunction weights, or ``None`` if exact."""

        kinds = [name for name, flag in (("PAW", self.paw), ("ultrasoft", self.uspp)) if flag]
        if not kinds:
            return None
        return (
            f"this calculation uses {' and '.join(kinds)} pseudopotentials: the "
            "plane-wave unfolding weights are those of the pseudo wavefunction, "
            "the augmentation charge inside the spheres is not represented.  The "
            "fiber sum rule still holds, so it does not detect this; the error is "
            "in how weight is distributed between primitive k-points and is "
            "largest for localized d/f states"
        )


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find(element: ElementTree.Element | None, *names: str) -> ElementTree.Element | None:
    """Find a descendant by a path of namespace-insensitive tag names."""

    current = element
    for name in names:
        if current is None:
            return None
        nxt = None
        for child in current:
            if _strip_ns(child.tag) == name:
                nxt = child
                break
        current = nxt
    return current


def _findall(element: ElementTree.Element | None, name: str) -> list[ElementTree.Element]:
    if element is None:
        return []
    return [child for child in element if _strip_ns(child.tag) == name]


def _text_str(element: ElementTree.Element | None) -> str:
    """Return the stripped text of ``element``, or the empty string when absent."""

    if element is None or element.text is None:
        return ""
    return element.text.strip()


def _text_floats(element: ElementTree.Element | None) -> NDArray[np.float64]:
    if element is None or element.text is None:
        return np.zeros(0, dtype=float)
    return np.asarray([float(token) for token in element.text.split()], dtype=float)


def _text_float(element: ElementTree.Element | None) -> float | None:
    if element is None or element.text is None or not element.text.strip():
        return None
    return float(element.text.split()[0])


def _text_int(element: ElementTree.Element | None) -> int | None:
    value = _text_float(element)
    return None if value is None else int(round(value))


def _text_bool(element: ElementTree.Element | None, default: bool = False) -> bool:
    if element is None or element.text is None:
        return default
    return element.text.strip().lower() in {"true", ".true.", "t", "1"}


def _three_rows(
    parent: ElementTree.Element | None, names: tuple[str, str, str]
) -> NDArray[np.float64] | None:
    if parent is None:
        return None
    rows = []
    for name in names:
        values = _text_floats(_find(parent, name))
        if values.size != 3:
            return None
        rows.append(values)
    return np.asarray(rows, dtype=float)


def find_qe_xml(path: str | Path) -> Path:
    """Locate a QE XML document.

    ``path`` may be the XML file itself, a ``.save`` directory, or a directory
    that contains one (an ``outdir``).
    """

    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        for basename in XML_BASENAMES:
            direct = candidate / basename
            if direct.is_file():
                return direct
        for save in sorted(candidate.glob("*.save")):
            for basename in XML_BASENAMES:
                nested = save / basename
                if nested.is_file():
                    return nested
    raise FileNotFoundError(f"no QE XML ({' or '.join(XML_BASENAMES)}) found for {candidate}")


def read_qe_xml(path: str | Path) -> QESaveMetadata:
    """Parse a QE ``data-file-schema.xml``.

    ``path`` is anything :func:`find_qe_xml` accepts.
    """

    xml_path = find_qe_xml(path)
    try:
        root = ElementTree.parse(xml_path).getroot()
    except ElementTree.ParseError as exc:  # pragma: no cover - message passthrough
        raise QEXMLError(f"could not parse {xml_path}: {exc}") from exc

    output = _find(root, "output")
    source = output if output is not None else _find(root, "input")
    if source is None:
        raise QEXMLError(f"{xml_path} has neither an <output> nor an <input> block")

    structure = _find(source, "atomic_structure")
    if structure is None:
        structure = _find(root, "input", "atomic_structure")
    if structure is None:
        raise QEXMLError(f"{xml_path} has no <atomic_structure>")
    alat_text = structure.get("alat")
    if alat_text is None:
        raise QEXMLError(f"{xml_path} has no alat attribute on <atomic_structure>")
    alat = float(alat_text)
    if not np.isfinite(alat) or alat <= 0.0:
        raise QEXMLError(f"{xml_path} has a non-positive alat {alat}")

    lattice = _three_rows(_find(structure, "cell"), ("a1", "a2", "a3"))
    if lattice is None:
        raise QEXMLError(f"{xml_path} has no complete <cell> block")
    if abs(float(np.linalg.det(lattice))) < 1e-12:
        raise QEXMLError(f"{xml_path} has a singular cell")

    basis = _find(source, "basis_set")
    reciprocal = _three_rows(_find(basis, "reciprocal_lattice"), ("b1", "b2", "b3"))
    gamma_only = _text_bool(_find(basis, "gamma_only"))

    band_structure = _find(source, "band_structure")
    kpoints, weights, eigenvalues, occupations = _read_ks_energies(band_structure)

    algorithmic = _find(source, "algorithmic_info")
    species = tuple(
        (
            child.get("name", f"species{index + 1}"),
            _text_str(_find(child, "pseudo_file")),
        )
        for index, child in enumerate(_findall(_find(source, "atomic_species"), "species"))
    )

    nbnd = _text_int(_find(band_structure, "nbnd"))
    nbnd_up = _text_int(_find(band_structure, "nbnd_up"))
    nbnd_dw = _text_int(_find(band_structure, "nbnd_dw"))
    if nbnd is None:
        if nbnd_up is not None and nbnd_dw is not None:
            nbnd = nbnd_up + nbnd_dw
        else:
            nbnd = int(eigenvalues.shape[1]) if eigenvalues.size else 0

    lattice_alat = lattice / alat
    kpoints_frac = kpoints @ lattice_alat.T if kpoints.size else kpoints.reshape(0, 3)
    fermi = _text_float(_find(band_structure, "fermi_energy"))
    if fermi is None:
        fermi = _text_float(_find(band_structure, "highestOccupiedLevel"))

    nat = structure.get("nat")
    return QESaveMetadata(
        path=xml_path,
        alat_bohr=alat,
        lattice_bohr=lattice,
        lattice_alat=lattice_alat,
        reciprocal_alat=reciprocal,
        kpoints_cart_alat=kpoints,
        kpoints_frac=kpoints_frac,
        kpoint_weights=weights,
        eigenvalues_ev=eigenvalues,
        occupations=occupations,
        nbnd=int(nbnd),
        nbnd_up=nbnd_up,
        nbnd_dw=nbnd_dw,
        n_electrons=_text_float(_find(band_structure, "nelec")),
        lsda=_text_bool(_find(band_structure, "lsda")) or nbnd_up is not None,
        noncolin=_text_bool(_find(band_structure, "noncolin")),
        spinorbit=_text_bool(_find(band_structure, "spinorbit")),
        gamma_only=gamma_only,
        uspp=_text_bool(_find(algorithmic, "uspp")),
        paw=_text_bool(_find(algorithmic, "paw")),
        fermi_energy_ev=None if fermi is None else fermi * HARTREE_EV,
        species=species,
        n_atoms=None if nat is None else int(nat),
    )


def _read_ks_energies(
    band_structure: ElementTree.Element | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64] | None,
]:
    blocks = _findall(band_structure, "ks_energies")
    if not blocks:
        empty = np.zeros((0, 3), dtype=float)
        return empty, np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float), None

    kpoints: list[NDArray[np.float64]] = []
    weights: list[float] = []
    eigenvalues: list[NDArray[np.float64]] = []
    occupations: list[NDArray[np.float64]] = []
    for block in blocks:
        kpoint_element = _find(block, "k_point")
        values = _text_floats(kpoint_element)
        if values.size != 3:
            raise QEXMLError("a <ks_energies> block has no three-component <k_point>")
        kpoints.append(values)
        weight = None if kpoint_element is None else kpoint_element.get("weight")
        weights.append(float(weight) if weight is not None else float("nan"))
        eigenvalues.append(_text_floats(_find(block, "eigenvalues")))
        occupations.append(_text_floats(_find(block, "occupations")))

    sizes = {values.size for values in eigenvalues}
    if len(sizes) > 1:
        raise QEXMLError(f"<ks_energies> blocks disagree on the band count: {sorted(sizes)}")
    energies = np.asarray(eigenvalues, dtype=float) * HARTREE_EV
    occupation_sizes = {values.size for values in occupations}
    occupation_array: NDArray[np.float64] | None = None
    if len(occupation_sizes) == 1 and occupation_sizes != {0}:
        occupation_array = np.asarray(occupations, dtype=float)
    return (
        np.asarray(kpoints, dtype=float),
        np.asarray(weights, dtype=float),
        energies,
        occupation_array,
    )


def eigenvalues_for_spin(
    metadata: QESaveMetadata,
    spin: int | None = None,
) -> NDArray[np.float64]:
    """Eigenvalues of one spin channel, in eV.

    For an unpolarized or noncollinear calculation there is one channel and
    ``spin`` is ignored.  For LSDA a channel must be chosen, because the two
    carry different bands.
    """

    blocks = metadata.eigenvalues_by_spin()
    if len(blocks) == 1:
        return blocks[0]
    if spin not in {1, 2}:
        raise ValueError(
            "this is a spin-polarized calculation; choose spin=1 (up) or spin=2 (down) "
            "to take the eigenvalues from the XML"
        )
    return blocks[spin - 1]


def read_qe_xml_eigenvalues(
    path: str | Path,
    *,
    spin: int | None = None,
) -> NDArray[np.float64]:
    """Kohn-Sham eigenvalues in eV from a QE ``.save``, shape ``(nks, nbnd)``."""

    metadata = read_qe_xml(path)
    energies = eigenvalues_for_spin(metadata, spin)
    if energies.size == 0:
        raise QEXMLError(f"{metadata.path} records no eigenvalues")
    return energies


def read_qe_lattice_alat(path: str | Path) -> NDArray[np.float64]:
    """QE's ``at`` matrix (direct lattice in units of ``alat``) from the XML."""

    return read_qe_xml(path).lattice_alat


def try_read_qe_metadata(path: str | Path) -> QESaveMetadata | None:
    """Read the XML if there is one and it parses; otherwise return ``None``.

    This is what the wavefunction readers use to fill in information the caller
    did not supply: a missing or unreadable XML must not turn a working
    calculation into an error.
    """

    try:
        return read_qe_xml(path)
    except (FileNotFoundError, QEXMLError, OSError):
        return None


def warn_about_pseudopotentials(metadata: QESaveMetadata | None) -> None:
    """Emit the PAW/ultrasoft normalization warning when it applies."""

    if metadata is None:
        return
    message = metadata.pseudo_warning()
    if message is not None:
        warnings.warn(message, stacklevel=2)


def check_xml_kpoint_order(
    metadata: QESaveMetadata,
    kpoints_cart_alat: NDArray[np.float64],
    *,
    atol: float = 1e-6,
) -> None:
    """Check wavefunction-file k-points against the XML list, in order.

    ``kpoints_cart_alat`` holds the ``xk`` of the wavefunction files, sorted by
    their k-point index, in the same units as the XML.  Entries that are
    ``None``/NaN are skipped, so a reader that could not recover ``xk`` still
    passes.
    """

    stored = np.asarray(kpoints_cart_alat, dtype=float)
    if stored.ndim != 2 or stored.shape[1] != 3:
        raise ValueError("kpoints_cart_alat must have shape (n, 3)")
    if metadata.n_kpoints == 0:
        return
    if stored.shape[0] > metadata.n_kpoints:
        raise ValueError(
            f"{stored.shape[0]} wavefunction files but the XML lists {metadata.n_kpoints} k-points"
        )
    for index, xk in enumerate(stored):
        if not np.all(np.isfinite(xk)):
            continue
        reference = metadata.kpoints_cart_alat[index]
        if not np.allclose(xk, reference, atol=atol):
            raise ValueError(
                f"wavefunction file {index + 1} has xk {tuple(xk)} but the XML "
                f"lists {tuple(reference)} at that index; the file order does "
                "not match the calculation"
            )
