# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Restricted-quartic angle prior.

A per-triplet quartic polynomial in $\cos\theta$ together with a $1/\sin^2\theta$
barrier that prevents the angle from approaching $0$ or $\pi$:

$$
V(\theta) = a\cos^4\theta + b\cos^3\theta + c\cos^2\theta + d\cos\theta
          + \frac{k}{\sin^2\theta} + v_0
$$

The vertex is the **first** atom of the triplet. Since the energy depends only on
$\cos\theta$ and $\sin\theta$, all coefficients are plain energies.

Reference: Bulacu et al., J. Chem. Theory Comput. 2013, 9 (8), 3282-3292.
DOI: 10.1021/ct400219n
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
class RestrictedQuarticParameters:
    r"""Restricted-quartic angle potential parameters.

    Energy $V(\theta) = a\cos^4\theta + b\cos^3\theta + c\cos^2\theta + d\cos\theta
    + k/\sin^2\theta + v_0$.

    Attributes:
        labels: Species labels, shape `(n_species,)`.
        a: $\cos^4\theta$ coefficients, shape `(n_species, n_species, n_species)`.
        b: $\cos^3\theta$ coefficients, shape `(n_species, n_species, n_species)`.
        c: $\cos^2\theta$ coefficients, shape `(n_species, n_species, n_species)`.
        d: $\cos\theta$ coefficients, shape `(n_species, n_species, n_species)`.
        k: $1/\sin^2\theta$ barrier coefficients, shape `(n_species, n_species, n_species)`.
        v0: Constant offsets, shape `(n_species, n_species, n_species)`.
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    a: Array  # (n_species, n_species, n_species)
    b: Array  # (n_species, n_species, n_species)
    c: Array  # (n_species, n_species, n_species)
    d: Array  # (n_species, n_species, n_species)
    k: Array  # (n_species, n_species, n_species)
    v0: Array  # (n_species, n_species, n_species)


type RestrictedQuarticInput = GraphPotentialInput[
    RestrictedQuarticParameters, IsBondedParticles, HasCell[AnyPeriodicity], Literal[3]
]


def restricted_quartic_energy(
    inp: RestrictedQuarticInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    r"""Compute restricted-quartic angle energy for all triplets.

    The vertex is the **first** atom of the triplet: the angle is formed by the
    displacements ``edge_shifts[:, 0]`` and ``edge_shifts[:, 1]``.

    Args:
        inp: Graph potential input with restricted-quartic parameters.

    Returns:
        Total angle energy per system.
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 3, (
        "Restricted-quartic potential only supports triplet interactions (order=3)."
    )
    edg_species = graph.particles[graph.edges.indices].labels.indices_in(
        inp.parameters.labels
    )
    s0, s1, s2 = edg_species[:, 0], edg_species[:, 1], edg_species[:, 2]
    a = inp.parameters.a[s0, s1, s2]
    b = inp.parameters.b[s0, s1, s2]
    c = inp.parameters.c[s0, s1, s2]
    d = inp.parameters.d[s0, s1, s2]
    k = inp.parameters.k[s0, s1, s2]
    v0 = inp.parameters.v0[s0, s1, s2]

    v1, v2 = graph.edge_shifts[:, 0], graph.edge_shifts[:, 1]
    cos = jnp.einsum("ij,ij->i", v1, v2) / (
        jnp.linalg.norm(v1, axis=-1) * jnp.linalg.norm(v2, axis=-1)
    )
    sin2 = 1.0 - cos**2

    edge_energy = (
        a * cos**4 + b * cos**3 + c * cos**2 + d * cos + k / sin2 + v0
    )
    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch[Any]())


def make_restricted_quartic_potential[
    State,
    P: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edge_indices_view: View[State, Index[ParticleId]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, RestrictedQuarticParameters],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]] | None,
    gradient_lens: Lens[RestrictedQuarticInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create a restricted-quartic angle potential for explicitly defined angles.

    Args:
        particles_view: Extracts particle data (positions, species) with system index.
        edge_indices_view: Extracts angle connectivity (triplets).
        systems_view: Extracts indexed system data (cell).
        parameter_view: Extracts [RestrictedQuarticParameters][kups.potential.classical.restricted_bending.RestrictedQuarticParameters].
        probe: Graph probe for incremental particle and neighbor-list updates.
        gradient_lens: Specifies gradients to compute.
        hessian_lens: Specifies Hessians to compute.
        hessian_idx_view: Hessian index structure.
        patch_idx_view: Cached output index structure.
        out_cache_lens: Cache location lens.

    Returns:
        Restricted-quartic angle [Potential][kups.core.potential.Potential].
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
        energy_fn=restricted_quartic_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


if TYPE_CHECKING:
    _rq: EnergyFunction[Any, RestrictedQuarticInput] = restricted_quartic_energy
