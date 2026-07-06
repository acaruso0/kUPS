# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Fourier-series dihedral prior.

A per-quadruplet truncated Fourier series in the torsion angle $\phi$,

$$
V(\phi) = v_0 + \sum_{n=1}^{N} \left[ k^{(1)}_n \sin(n\phi) + k^{(2)}_n \cos(n\phi) \right]
$$

where the number of terms $N$ is inferred from the coefficient arrays. The
torsion angle of the quadruplet $(i, j, k, l)$ is computed the same way as the
UFF dihedral (``arctan2(sin\phi, cos\phi)``), in radians.
"""

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

from kups.core.cell import AnyPeriodicity
from kups.core.data import Index, Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import FixedEdgesNeighborList
from kups.core.patch import IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    Energy,
    Potential,
    PotentialOut,
)
from kups.core.typing import (
    HasCell,
    HasPositionsAndLabels,
    Label,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field
from kups.potential.common.energy import (
    EnergyFunction,
    PotentialFromEnergy,
)
from kups.potential.common.graph import (
    GraphConstructor,
    GraphPotentialInput,
    IsGraphProbe,
    IsRadiusGraphPoints,
    LocalGraphSumComposer,
)


@runtime_checkable
class IsBondedParticles(HasPositionsAndLabels, IsRadiusGraphPoints, Protocol):
    """Particle data with positions, labels, and system index."""

    ...


@dataclass
class FourierDihedralParameters:
    r"""Fourier-series dihedral parameters.

    Energy $V(\phi) = v_0 + \sum_{n=1}^{N} [k^{(1)}_n \sin(n\phi) + k^{(2)}_n \cos(n\phi)]$
    over the torsion angle $\phi$ of the quadruplet $(i, j, k, l)$.

    Attributes:
        labels: Species labels, shape `(n_species,)`.
        k1s: Sine coefficients $k^{(1)}_1, \dots, k^{(1)}_N$,
            shape `(N, n_species, n_species, n_species, n_species)`.
        k2s: Cosine coefficients $k^{(2)}_1, \dots, k^{(2)}_N$,
            shape `(N, n_species, n_species, n_species, n_species)`.
        v0: Constant offsets,
            shape `(n_species, n_species, n_species, n_species)`.
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    k1s: Array  # (N, n_species, n_species, n_species, n_species)
    k2s: Array  # (N, n_species, n_species, n_species, n_species)
    v0: Array  # (n_species, n_species, n_species, n_species)


type FourierDihedralInput = GraphPotentialInput[
    FourierDihedralParameters, IsBondedParticles, HasCell[AnyPeriodicity], Literal[4]
]


def fourier_dihedral_energy(
    inp: FourierDihedralInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    r"""Compute Fourier-series dihedral energy for all quadruplets.

    Args:
        inp: Graph potential input with Fourier dihedral parameters.

    Returns:
        Total dihedral energy per system.
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 4, (
        "Fourier dihedral potential only supports quadruplet interactions (order=4)."
    )
    edg_species = graph.particles[graph.edges.indices].labels.indices_in(
        inp.parameters.labels
    )
    s0, s1, s2, s3 = (
        edg_species[:, 0],
        edg_species[:, 1],
        edg_species[:, 2],
        edg_species[:, 3],
    )
    k1s = inp.parameters.k1s[:, s0, s1, s2, s3]  # (N, n_edges)
    k2s = inp.parameters.k2s[:, s0, s1, s2, s3]  # (N, n_edges)
    v0 = inp.parameters.v0[s0, s1, s2, s3]  # (n_edges,)

    # Torsion angle (same convention as the UFF dihedral).
    r_ij = graph.edge_shifts[:, 0]  # j - i
    r_jk = graph.edge_shifts[:, 1] - r_ij  # k - j
    r_kl = graph.edge_shifts[:, 2] - graph.edge_shifts[:, 1]  # l - k

    n1 = jnp.cross(r_ij, r_jk)
    n2 = jnp.cross(r_jk, r_kl)

    eps = 1e-10
    n1n = n1 / (jnp.linalg.norm(n1, axis=-1, keepdims=True) + eps)
    n2n = n2 / (jnp.linalg.norm(n2, axis=-1, keepdims=True) + eps)
    r_jk_n = r_jk / (jnp.linalg.norm(r_jk, axis=-1, keepdims=True) + eps)

    cos_phi = jnp.sum(n1n * n2n, axis=-1)
    sin_phi = jnp.sum(jnp.cross(n1n, n2n) * r_jk_n, axis=-1)
    phi = jnp.arctan2(sin_phi, cos_phi)

    n_degs = inp.parameters.k1s.shape[0]
    orders = jnp.arange(1, n_degs + 1)  # (N,)
    n_phi = orders[:, None] * phi[None, :]  # (N, n_edges)
    edge_energy = v0 + jnp.sum(
        k1s * jnp.sin(n_phi) + k2s * jnp.cos(n_phi), axis=0
    )
    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch[Any]())


def make_fourier_dihedral_potential[
    State,
    P: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edge_indices_view: View[State, Index[ParticleId]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, FourierDihedralParameters],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]] | None,
    gradient_lens: Lens[FourierDihedralInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create a Fourier-series dihedral potential for explicitly defined dihedrals.

    Args:
        particles_view: Extracts particle data (positions, species) with system index.
        edge_indices_view: Extracts dihedral connectivity (quadruplets).
        systems_view: Extracts indexed system data (cell).
        parameter_view: Extracts [FourierDihedralParameters][kups.potential.classical.fourier_series.FourierDihedralParameters].
        probe: Graph probe for incremental particle and neighbor-list updates.
        gradient_lens: Specifies gradients to compute.
        hessian_lens: Specifies Hessians to compute.
        hessian_idx_view: Hessian index structure.
        patch_idx_view: Cached output index structure.
        out_cache_lens: Cache location lens.

    Returns:
        Fourier dihedral [Potential][kups.core.potential.Potential].
    """
    graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=lambda state: FixedEdgesNeighborList[Literal[4]](
            edge_indices_view(state)
        ),
        probe=probe,
    )
    composer = LocalGraphSumComposer(
        graph_constructor=graph_fn,
        parameter_view=parameter_view,
    )
    return PotentialFromEnergy(
        composer=composer,
        energy_fn=fourier_dihedral_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


if TYPE_CHECKING:
    _fd: EnergyFunction[Any, FourierDihedralInput] = fourier_dihedral_energy
