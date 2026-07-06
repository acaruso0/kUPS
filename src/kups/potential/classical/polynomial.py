# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Polynomial angle prior.

A per-triplet polynomial in an angular feature,

$$
V(x) = v_0 + \sum_{n=1}^{N} k_n\, x^n
$$

where the feature $x$ is either $\cos\theta$ (``feature="cosine"``) or the angle
$\theta$ in degrees (``feature="angle"``). The degree $N$ is inferred from the
coefficient array. The vertex is the **first** atom of the triplet.
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
class PolynomialAngleParameters:
    r"""Polynomial angle potential parameters, $V(x) = v_0 + \sum_{n=1}^{N} k_n x^n$.

    Attributes:
        labels: Species labels, shape `(n_species,)`.
        ks: Polynomial coefficients $k_1, \dots, k_N$,
            shape `(N, n_species, n_species, n_species)`; the leading axis is the
            degree (``ks[0]`` multiplies $x^1$).
        v0: Constant offsets, shape `(n_species, n_species, n_species)`.
        feature: Angular feature ``x``: ``"cosine"`` for $\cos\theta$ or
            ``"angle"`` for $\theta$ in degrees.
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    ks: Array  # (N, n_species, n_species, n_species)
    v0: Array  # (n_species, n_species, n_species)
    feature: Literal["cosine", "angle"] = field(static=True, default="cosine")


type PolynomialAngleInput = GraphPotentialInput[
    PolynomialAngleParameters, IsBondedParticles, HasCell[AnyPeriodicity], Literal[3]
]


def polynomial_angle_energy(
    inp: PolynomialAngleInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    r"""Compute polynomial angle energy $v_0 + \sum_n k_n x^n$ for all triplets.

    The vertex is the **first** atom of the triplet: the angle is formed by the
    displacements ``edge_shifts[:, 0]`` and ``edge_shifts[:, 1]``.

    Args:
        inp: Graph potential input with polynomial angle parameters.

    Returns:
        Total angle energy per system.
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 3, (
        "Polynomial angle potential only supports triplet interactions (order=3)."
    )
    edg_species = graph.particles[graph.edges.indices].labels.indices_in(
        inp.parameters.labels
    )
    s0, s1, s2 = edg_species[:, 0], edg_species[:, 1], edg_species[:, 2]
    ks = inp.parameters.ks[:, s0, s1, s2]  # (N, n_edges)
    v0 = inp.parameters.v0[s0, s1, s2]  # (n_edges,)

    v1, v2 = graph.edge_shifts[:, 0], graph.edge_shifts[:, 1]
    cos_angle = jnp.einsum("ij,ij->i", v1, v2) / (
        jnp.linalg.norm(v1, axis=-1) * jnp.linalg.norm(v2, axis=-1)
    )
    if inp.parameters.feature == "cosine":
        x = cos_angle
    else:
        x = jnp.rad2deg(jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0)))

    n_degs = inp.parameters.ks.shape[0]
    powers = jnp.arange(1, n_degs + 1)  # (N,)
    x_pow = x[None, :] ** powers[:, None]  # (N, n_edges)
    edge_energy = v0 + jnp.sum(ks * x_pow, axis=0)
    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch[Any]())


def make_polynomial_angle_potential[
    State,
    P: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edge_indices_view: View[State, Index[ParticleId]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, PolynomialAngleParameters],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]] | None,
    gradient_lens: Lens[PolynomialAngleInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create a polynomial angle potential for explicitly defined angles.

    Applies a per-triplet polynomial in the angular feature to specified atom
    triplets (i-j-k). Angles must be provided explicitly via the
    ``edge_indices_view`` edge set as triplets with the vertex first.

    Args:
        particles_view: Extracts particle data (positions, species) with system index.
        edge_indices_view: Extracts angle connectivity (triplets).
        systems_view: Extracts indexed system data (cell).
        parameter_view: Extracts [PolynomialAngleParameters][kups.potential.classical.polynomial.PolynomialAngleParameters].
        probe: Graph probe for incremental particle and neighbor-list updates.
        gradient_lens: Specifies gradients to compute.
        hessian_lens: Specifies Hessians to compute.
        hessian_idx_view: Hessian index structure.
        patch_idx_view: Cached output index structure.
        out_cache_lens: Cache location lens.

    Returns:
        Polynomial angle [Potential][kups.core.potential.Potential].
    """
    graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=lambda state: FixedEdgesNeighborList[Literal[3]](
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
        energy_fn=polynomial_angle_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


if TYPE_CHECKING:
    _pa: EnergyFunction[Any, PolynomialAngleInput] = polynomial_angle_energy
