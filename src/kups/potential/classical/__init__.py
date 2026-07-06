# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Classical molecular mechanics force fields.

This module provides standard force field components used in molecular simulations:
non-bonded interactions (Lennard-Jones, Coulomb, Ewald) and bonded terms (harmonic
bonds, angles, dihedrals). All potentials support tail corrections, cutoffs, and
neighbor lists.

## Available Potentials

- **[Lennard-Jones][kups.potential.classical.lennard_jones]**: Van der Waals interactions with optional tail corrections
- **[Coulomb][kups.potential.classical.coulomb]**: Electrostatic interactions
- **[Ewald][kups.potential.classical.ewald]**: Long-range electrostatics via Ewald summation
- **[Harmonic][kups.potential.classical.harmonic]**: Bonded interactions (bonds, angles)
- **[Cosine Angle][kups.potential.classical.uff_cosine_angle]**: UFF-style cosine angle bending
- **[Morse][kups.potential.classical.morse]**: Anharmonic bond stretching with proper dissociation
- **[Dihedral][kups.potential.classical.uff_dihedral]**: Torsion potentials (UFF-style)
- **[Inversion][kups.potential.classical.uff_inversion]**: Out-of-plane/improper potentials (UFF-style)
- **[Repulsion][kups.potential.classical.repulsion]**: Power-law and exponential excluded-volume repulsion

Each potential provides a `make_*_potential` factory function that constructs a
configured [Potential][kups.core.potential.Potential] instance.
"""

from .uff_cosine_angle import (
    UFFCosineAngleParameters,
    uff_cosine_angle_energy,
    make_uff_cosine_angle_potential,
)
from .uff_dihedral import UFFDihedralParameters, make_uff_dihedral_potential
from .ewald import EwaldParameters, make_ewald_potential
from .harmonic import (
    HarmonicAngleParameters,
    HarmonicBondParameters,
    HarmonicImproperParameters,
    harmonic_improper_energy,
    make_harmonic_angle_potential,
    make_harmonic_bond_potential,
    make_harmonic_improper_potential,
)
from .uff_inversion import (
    UFFInversionParameters,
    uff_inversion_energy,
    make_uff_inversion_potential,
)
from .lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
    LennardJonesParameters,
    PairTailCorrectedLennardJonesParameters,
    make_global_lennard_jones_tail_correction_potential,
    make_lennard_jones_potential,
    make_pair_tail_corrected_lennard_jones_potential,
)
from .fourier_series import (
    FourierDihedralParameters,
    fourier_dihedral_energy,
    make_fourier_dihedral_potential,
)
from .morse import MorseBondParameters, make_morse_bond_potential
from .polynomial import (
    PolynomialAngleParameters,
    make_polynomial_angle_potential,
    polynomial_angle_energy,
)
from .repulsion import (
    CutoffExpRepulsionParameters,
    CutoffRepulsionParameters,
    ExpRepulsionParameters,
    RepulsionParameters,
    cutoff_exp_repulsion_energy,
    cutoff_repulsion_energy,
    exp_repulsion_energy,
    make_cutoff_exp_repulsion_potential,
    make_cutoff_repulsion_potential,
    make_exp_repulsion_potential,
    make_repulsion_potential,
    repulsion_energy,
)

__all__ = [
    "make_uff_cosine_angle_potential",
    "make_uff_dihedral_potential",
    "make_ewald_potential",
    "make_harmonic_angle_potential",
    "make_harmonic_bond_potential",
    "make_harmonic_improper_potential",
    "make_uff_inversion_potential",
    "make_fourier_dihedral_potential",
    "make_morse_bond_potential",
    "make_polynomial_angle_potential",
    "make_repulsion_potential",
    "make_cutoff_repulsion_potential",
    "make_exp_repulsion_potential",
    "make_cutoff_exp_repulsion_potential",
    "make_lennard_jones_potential",
    "make_global_lennard_jones_tail_correction_potential",
    "make_pair_tail_corrected_lennard_jones_potential",
    "UFFCosineAngleParameters",
    "uff_cosine_angle_energy",
    "UFFDihedralParameters",
    "EwaldParameters",
    "HarmonicAngleParameters",
    "HarmonicBondParameters",
    "HarmonicImproperParameters",
    "harmonic_improper_energy",
    "UFFInversionParameters",
    "uff_inversion_energy",
    "FourierDihedralParameters",
    "fourier_dihedral_energy",
    "MorseBondParameters",
    "PolynomialAngleParameters",
    "polynomial_angle_energy",
    "RepulsionParameters",
    "repulsion_energy",
    "CutoffRepulsionParameters",
    "cutoff_repulsion_energy",
    "ExpRepulsionParameters",
    "exp_repulsion_energy",
    "CutoffExpRepulsionParameters",
    "cutoff_exp_repulsion_energy",
    "LennardJonesParameters",
    "GlobalTailCorrectedLennardJonesParameters",
    "PairTailCorrectedLennardJonesParameters",
]
