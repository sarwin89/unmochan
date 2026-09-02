"""Input/output adapters."""

from unfoldlab.io.force_constants import read_force_constant_model
from unfoldlab.io.model_hamiltonian import read_tight_binding_model
from unfoldlab.io.plot_bands import (
    color_from_weight,
    plot_unfolded,
    plot_unfolded_svg,
)
from unfoldlab.io.plot_spectral import (
    plot_spectral_function,
    plot_spectral_function_svg,
    spectral_energy_grid,
    spectral_intensity_bound,
    spectral_map,
)
from unfoldlab.io.procar import ProcarData, read_procar
from unfoldlab.io.qe import (
    QEKMap,
    QEPath,
    build_qe_path,
    qe_effective_band_structure,
    read_gnu_blocks,
    read_kmap,
    read_pw_input_structure,
    read_spin_texture_table,
    read_weight_table,
    write_qe_path_files,
    write_spin_texture_table,
    write_unfolded,
    write_weight_table,
)
from unfoldlab.io.qe_wfc import compute_spin_texture_from_qe_save
from unfoldlab.io.qe_xml import (
    QESaveMetadata,
    QEXMLError,
    find_qe_xml,
    read_qe_lattice_alat,
    read_qe_xml,
)
from unfoldlab.io.serialization import (
    RunManifest,
    SourceFile,
    build_run_manifest,
    read_ebs_json,
    read_run_manifest,
    write_ebs_json,
    write_run_manifest,
)
from unfoldlab.io.vasp import EigenvalData, read_eigenval, read_poscar
from unfoldlab.io.vasp_wfc import (
    WavecarHeader,
    WavecarReader,
    compute_spin_texture_from_wavecar,
    compute_weights_from_wavecar,
)
from unfoldlab.io.wannier90 import (
    Wannier90TB,
    read_wannier90_hr,
    read_wannier90_tb,
)

__all__ = [
    "EigenvalData",
    "ProcarData",
    "QEKMap",
    "QEPath",
    "QESaveMetadata",
    "QEXMLError",
    "RunManifest",
    "SourceFile",
    "Wannier90TB",
    "WavecarHeader",
    "WavecarReader",
    "build_qe_path",
    "build_run_manifest",
    "find_qe_xml",
    "compute_spin_texture_from_wavecar",
    "compute_spin_texture_from_qe_save",
    "compute_weights_from_wavecar",
    "qe_effective_band_structure",
    "read_ebs_json",
    "read_force_constant_model",
    "read_eigenval",
    "read_gnu_blocks",
    "read_kmap",
    "read_poscar",
    "color_from_weight",
    "plot_unfolded",
    "plot_unfolded_svg",
    "plot_spectral_function",
    "plot_spectral_function_svg",
    "read_procar",
    "spectral_energy_grid",
    "spectral_intensity_bound",
    "spectral_map",
    "read_qe_lattice_alat",
    "read_qe_xml",
    "read_pw_input_structure",
    "read_run_manifest",
    "read_spin_texture_table",
    "read_tight_binding_model",
    "read_wannier90_hr",
    "read_wannier90_tb",
    "read_weight_table",
    "write_ebs_json",
    "write_qe_path_files",
    "write_run_manifest",
    "write_spin_texture_table",
    "write_unfolded",
    "write_weight_table",
]
