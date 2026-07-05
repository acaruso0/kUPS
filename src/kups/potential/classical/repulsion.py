# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Repulsion priors for non-bonded excluded-volume interactions.

This module provides power-law and exponential repulsion terms together with
cutoff-shifted variants that are $C^1$-continuous at a cutoff radius. They are
typically applied to an explicit (e.g. fully connected) set of non-bonded pairs
to enforce excluded volume in coarse-grained models.

Power-law repulsion:

$$
U(r) = \left(\frac{\sigma}{r}\right)^6
$$

Exponential (exp-6) repulsion:

$$
U(r) = \frac{6}{\alpha}\,\exp\!\left[\alpha\left(1 - \frac{r}{r_0}\right)\right]
$$

The cutoff variants subtract the value and slope at $r_\mathrm{c}$ so that both
the energy and its first derivative vanish at the cutoff; the energy is zero for
$r \ge r_\mathrm{c}$.
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


# --------------------------------------------------------------------------- #
# Power-law repulsion: U(r) = (sigma / r)^6
# --------------------------------------------------------------------------- #
@dataclass
class RepulsionParameters:
    r"""Power-law repulsion parameters, $U(r) = (\sigma / r)^6$.

    Attributes:
        labels: Species labels, shape `(n_species,)`.
        sigma: Excluded-volume radii [Å], shape `(n_species, n_species)`.
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    sigma: Array  # (n_species, n_species)


type RepulsionInput = GraphPotentialInput[
    RepulsionParameters, IsBondedParticles, HasCell[AnyPeriodicity], Literal[2]
]


def repulsion_energy(
    inp: RepulsionInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    r"""Compute power-law repulsion energy $(\sigma/r)^6$ for all pairs.

    Args:
        inp: Graph potential input with repulsion parameters.

    Returns:
        Total repulsion energy per system.
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 2, (
        "Repulsion potential only supports pairwise interactions (order=2)."
    )
    edg_species = graph.particles[graph.edges.indices].labels.indices_in(
        inp.parameters.labels
    )
    sigma = inp.parameters.sigma[edg_species[:, 0], edg_species[:, 1]]
    r = jnp.linalg.norm(graph.edge_shifts[:, 0], axis=-1)
    edge_energy = (sigma / r) ** 6
    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch[Any]())


# --------------------------------------------------------------------------- #
# Cutoff-shifted power-law repulsion
# --------------------------------------------------------------------------- #
@dataclass
class CutoffRepulsionParameters:
    r"""Cutoff-shifted power-law repulsion, $C^1$-continuous at ``cutoff``.

    For $r < r_\mathrm{c}$ the energy is
    $U(r) = (\sigma/r)^6 - U(r_\mathrm{c}) - (r - r_\mathrm{c})\,U'(r_\mathrm{c})$
    with $U'(r_\mathrm{c}) = -6\,(\sigma/r_\mathrm{c})^6 / r_\mathrm{c}$, and zero
    for $r \ge r_\mathrm{c}$.

    Attributes:
        labels: Species labels, shape `(n_species,)`.
        sigma: Excluded-volume radii [Å], shape `(n_species, n_species)`.
        cutoff: Cutoff radius [Å] beyond which the energy vanishes.
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    sigma: Array  # (n_species, n_species)
    cutoff: float = field(static=True)


type CutoffRepulsionInput = GraphPotentialInput[
    CutoffRepulsionParameters, IsBondedParticles, HasCell[AnyPeriodicity], Literal[2]
]


def cutoff_repulsion_energy(
    inp: CutoffRepulsionInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    r"""Compute cutoff-shifted power-law repulsion energy for all pairs.

    Args:
        inp: Graph potential input with cutoff repulsion parameters.

    Returns:
        Total repulsion energy per system.
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 2, (
        "Repulsion potential only supports pairwise interactions (order=2)."
    )
    edg_species = graph.particles[graph.edges.indices].labels.indices_in(
        inp.parameters.labels
    )
    sigma = inp.parameters.sigma[edg_species[:, 0], edg_species[:, 1]]
    rc = inp.parameters.cutoff
    r = jnp.linalg.norm(graph.edge_shifts[:, 0], axis=-1)

    u = (sigma / r) ** 6
    uc = (sigma / rc) ** 6
    duc = -6.0 * uc / rc
    shifted = u - uc - (r - rc) * duc
    edge_energy = jnp.where(r < rc, shifted, 0.0)
    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch[Any]())


# --------------------------------------------------------------------------- #
# Exponential (exp-6) repulsion
# --------------------------------------------------------------------------- #
@dataclass
class ExpRepulsionParameters:
    r"""Exponential repulsion, $U(r) = (6/\alpha)\,\exp[\alpha(1 - r/r_0)]$.

    Attributes:
        labels: Species labels, shape `(n_species,)`.
        alpha: Repulsion strengths, shape `(n_species, n_species)`.
        r0: Excluded-volume radii [Å], shape `(n_species, n_species)`.
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    alpha: Array  # (n_species, n_species)
    r0: Array  # (n_species, n_species)


type ExpRepulsionInput = GraphPotentialInput[
    ExpRepulsionParameters, IsBondedParticles, HasCell[AnyPeriodicity], Literal[2]
]


def exp_repulsion_energy(
    inp: ExpRepulsionInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    r"""Compute exponential repulsion energy for all pairs.

    Args:
        inp: Graph potential input with exponential repulsion parameters.

    Returns:
        Total repulsion energy per system.
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 2, (
        "Repulsion potential only supports pairwise interactions (order=2)."
    )
    edg_species = graph.particles[graph.edges.indices].labels.indices_in(
        inp.parameters.labels
    )
    alpha = inp.parameters.alpha[edg_species[:, 0], edg_species[:, 1]]
    r0 = inp.parameters.r0[edg_species[:, 0], edg_species[:, 1]]
    r = jnp.linalg.norm(graph.edge_shifts[:, 0], axis=-1)
    edge_energy = (6.0 / alpha) * jnp.exp(alpha * (1.0 - r / r0))
    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch[Any]())


# --------------------------------------------------------------------------- #
# Cutoff-shifted exponential repulsion
# --------------------------------------------------------------------------- #
@dataclass
class CutoffExpRepulsionParameters:
    r"""Cutoff-shifted exponential repulsion, $C^1$-continuous at ``cutoff``.

    For $r < r_\mathrm{c}$ the energy is
    $U(r) = U_\mathrm{exp}(r) - U_\mathrm{exp}(r_\mathrm{c})
    - (r - r_\mathrm{c})\,U_\mathrm{exp}'(r_\mathrm{c})$ with
    $U_\mathrm{exp}'(r_\mathrm{c}) = -(\alpha / r_0)\,U_\mathrm{exp}(r_\mathrm{c})$,
    and zero for $r \ge r_\mathrm{c}$.

    Attributes:
        labels: Species labels, shape `(n_species,)`.
        alpha: Repulsion strengths, shape `(n_species, n_species)`.
        r0: Excluded-volume radii [Å], shape `(n_species, n_species)`.
        cutoff: Cutoff radius [Å] beyond which the energy vanishes.
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    alpha: Array  # (n_species, n_species)
    r0: Array  # (n_species, n_species)
    cutoff: float = field(static=True)


type CutoffExpRepulsionInput = GraphPotentialInput[
    CutoffExpRepulsionParameters,
    IsBondedParticles,
    HasCell[AnyPeriodicity],
    Literal[2],
]


def cutoff_exp_repulsion_energy(
    inp: CutoffExpRepulsionInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    r"""Compute cutoff-shifted exponential repulsion energy for all pairs.

    Args:
        inp: Graph potential input with cutoff exponential repulsion parameters.

    Returns:
        Total repulsion energy per system.
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 2, (
        "Repulsion potential only supports pairwise interactions (order=2)."
    )
    edg_species = graph.particles[graph.edges.indices].labels.indices_in(
        inp.parameters.labels
    )
    alpha = inp.parameters.alpha[edg_species[:, 0], edg_species[:, 1]]
    r0 = inp.parameters.r0[edg_species[:, 0], edg_species[:, 1]]
    rc = inp.parameters.cutoff
    r = jnp.linalg.norm(graph.edge_shifts[:, 0], axis=-1)

    u = (6.0 / alpha) * jnp.exp(alpha * (1.0 - r / r0))
    uc = (6.0 / alpha) * jnp.exp(alpha * (1.0 - rc / r0))
    duc = -(alpha / r0) * uc
    shifted = u - uc - (r - rc) * duc
    edge_energy = jnp.where(r < rc, shifted, 0.0)
    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch[Any]())


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def make_repulsion_potential[
    State,
    P: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edge_indices_view: View[State, Index[ParticleId]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, RepulsionParameters],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]] | None,
    gradient_lens: Lens[RepulsionInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create a power-law repulsion potential for explicitly defined pairs.

    Applies $(\\sigma/r)^6$ repulsion to specified atom pairs. Pairs must be
    provided explicitly via the ``edge_indices_view`` edge set (e.g. a fully
    connected non-bonded set).

    Args:
        particles_view: Extracts particle data (positions, species) with system index.
        edge_indices_view: Extracts pair connectivity.
        systems_view: Extracts indexed system data (cell).
        parameter_view: Extracts [RepulsionParameters][kups.potential.classical.repulsion.RepulsionParameters].
        probe: Graph probe for incremental particle and neighbor-list updates.
        gradient_lens: Specifies gradients to compute.
        hessian_lens: Specifies Hessians to compute.
        hessian_idx_view: Hessian index structure.
        patch_idx_view: Cached output index structure.
        out_cache_lens: Cache location lens.

    Returns:
        Repulsion [Potential][kups.core.potential.Potential].
    """
    graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=lambda state: FixedEdgesNeighborList[Literal[2]](
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
        energy_fn=repulsion_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


def make_cutoff_repulsion_potential[
    State,
    P: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edge_indices_view: View[State, Index[ParticleId]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, CutoffRepulsionParameters],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]] | None,
    gradient_lens: Lens[CutoffRepulsionInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create a cutoff-shifted power-law repulsion potential for explicit pairs.

    Args:
        particles_view: Extracts particle data (positions, species) with system index.
        edge_indices_view: Extracts pair connectivity.
        systems_view: Extracts indexed system data (cell).
        parameter_view: Extracts [CutoffRepulsionParameters][kups.potential.classical.repulsion.CutoffRepulsionParameters].
        probe: Graph probe for incremental particle and neighbor-list updates.
        gradient_lens: Specifies gradients to compute.
        hessian_lens: Specifies Hessians to compute.
        hessian_idx_view: Hessian index structure.
        patch_idx_view: Cached output index structure.
        out_cache_lens: Cache location lens.

    Returns:
        Cutoff repulsion [Potential][kups.core.potential.Potential].
    """
    graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=lambda state: FixedEdgesNeighborList[Literal[2]](
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
        energy_fn=cutoff_repulsion_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


def make_exp_repulsion_potential[
    State,
    P: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edge_indices_view: View[State, Index[ParticleId]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, ExpRepulsionParameters],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]] | None,
    gradient_lens: Lens[ExpRepulsionInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create an exponential repulsion potential for explicitly defined pairs.

    Args:
        particles_view: Extracts particle data (positions, species) with system index.
        edge_indices_view: Extracts pair connectivity.
        systems_view: Extracts indexed system data (cell).
        parameter_view: Extracts [ExpRepulsionParameters][kups.potential.classical.repulsion.ExpRepulsionParameters].
        probe: Graph probe for incremental particle and neighbor-list updates.
        gradient_lens: Specifies gradients to compute.
        hessian_lens: Specifies Hessians to compute.
        hessian_idx_view: Hessian index structure.
        patch_idx_view: Cached output index structure.
        out_cache_lens: Cache location lens.

    Returns:
        Exponential repulsion [Potential][kups.core.potential.Potential].
    """
    graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=lambda state: FixedEdgesNeighborList[Literal[2]](
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
        energy_fn=exp_repulsion_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


def make_cutoff_exp_repulsion_potential[
    State,
    P: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edge_indices_view: View[State, Index[ParticleId]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, CutoffExpRepulsionParameters],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]] | None,
    gradient_lens: Lens[CutoffExpRepulsionInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create a cutoff-shifted exponential repulsion potential for explicit pairs.

    Args:
        particles_view: Extracts particle data (positions, species) with system index.
        edge_indices_view: Extracts pair connectivity.
        systems_view: Extracts indexed system data (cell).
        parameter_view: Extracts [CutoffExpRepulsionParameters][kups.potential.classical.repulsion.CutoffExpRepulsionParameters].
        probe: Graph probe for incremental particle and neighbor-list updates.
        gradient_lens: Specifies gradients to compute.
        hessian_lens: Specifies Hessians to compute.
        hessian_idx_view: Hessian index structure.
        patch_idx_view: Cached output index structure.
        out_cache_lens: Cache location lens.

    Returns:
        Cutoff exponential repulsion [Potential][kups.core.potential.Potential].
    """
    graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=lambda state: FixedEdgesNeighborList[Literal[2]](
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
        energy_fn=cutoff_exp_repulsion_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


if TYPE_CHECKING:
    _r: EnergyFunction[Any, RepulsionInput] = repulsion_energy
    _cr: EnergyFunction[Any, CutoffRepulsionInput] = cutoff_repulsion_energy
    _er: EnergyFunction[Any, ExpRepulsionInput] = exp_repulsion_energy
    _cer: EnergyFunction[Any, CutoffExpRepulsionInput] = cutoff_exp_repulsion_energy
