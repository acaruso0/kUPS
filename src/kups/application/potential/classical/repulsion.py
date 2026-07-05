# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for repulsion priors.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, non-bonded pair indices, and parameters
into the state-agnostic factories in
[kups.potential.classical.repulsion][].

All variants share a common ``repulsion_edge_indices`` connectivity attribute
(the non-bonded pair set). Parameters may live on the state (e.g.
``state.repulsion_parameters``) or be passed directly via ``parameters=``; in the
latter case they are bound with a constant lens and the state need not carry a
parameter field. For incremental (probe) updates with constant parameters, the
cache is read from a conventional ``*_cache`` attribute.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, overload

from kups.core.cell import AnyPeriodicity
from kups.core.data import Index
from kups.core.lens import Lens, const_lens
from kups.core.patch import Patch, Probe
from kups.core.potential import (
    EMPTY_LENS,
    EmptyType,
    Potential,
    PotentialOut,
    empty_patch_idx_view,
)
from kups.core.typing import HasCache, HasCell, IsState, MaybeCached, ParticleId
from kups.potential.classical.repulsion import (
    CutoffExpRepulsionParameters,
    CutoffRepulsionParameters,
    ExpRepulsionParameters,
    IsBondedParticles,
    RepulsionParameters,
    make_cutoff_exp_repulsion_potential,
    make_cutoff_repulsion_potential,
    make_exp_repulsion_potential,
    make_repulsion_potential,
)
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    position_and_cell_idx_view,
)
from kups.potential.common.graph import GRAPH_GEOMETRY, IsGraphProbe


class HasRepulsionParticlesAndSystems(
    IsState[IsBondedParticles, HasCell[AnyPeriodicity]], Protocol
): ...


class IsRepulsionGraphState(HasRepulsionParticlesAndSystems, Protocol):
    """Particles, systems, and non-bonded pair indices (no parameters)."""

    @property
    def repulsion_edge_indices(self) -> Index[ParticleId]: ...


# --------------------------------------------------------------------------- #
# Power-law repulsion
# --------------------------------------------------------------------------- #
class IsRepulsionState[Params](IsRepulsionGraphState, Protocol):
    """:class:`IsRepulsionGraphState` that also carries repulsion parameters."""

    @property
    def repulsion_parameters(self) -> Params: ...


class IsCachedRepulsionState[Cache](IsRepulsionGraphState, Protocol):
    """:class:`IsRepulsionGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def repulsion_cache(self) -> Cache: ...


@overload
def make_repulsion_from_state[State](
    state: Lens[State, IsRepulsionState[MaybeCached[RepulsionParameters, Any]]],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_repulsion_from_state[State](
    state: Lens[State, IsRepulsionState[MaybeCached[RepulsionParameters, Any]]],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsRepulsionState[
            HasCache[RepulsionParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsRepulsionState[
            HasCache[RepulsionParameters, PotentialOut[PositionsAndCell, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


@overload
def make_repulsion_from_state[State](
    state: Lens[State, IsRepulsionGraphState],
    probe: None = None,
    *,
    parameters: RepulsionParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_repulsion_from_state[State](
    state: Lens[State, IsRepulsionGraphState],
    probe: None = None,
    *,
    parameters: RepulsionParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCachedRepulsionState[PotentialOut[EmptyType, EmptyType]]],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: RepulsionParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedRepulsionState[PotentialOut[PositionsAndCell, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: RepulsionParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


def make_repulsion_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: RepulsionParameters | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
) -> Any:
    """Create a power-law repulsion potential, optionally with incremental updates.

    Convenience wrapper around
    [make_repulsion_potential][kups.potential.classical.repulsion.make_repulsion_potential].

    Args:
        state: Lens into the sub-state providing particles, cell, and
            ``repulsion_edge_indices`` (plus ``repulsion_parameters`` when
            ``parameters`` is not given).
        probe: If provided, detects particle changes and supplies the
            before/after fixed-edge neighbor lists for incremental updates.
        parameters: Constant repulsion parameters. When given they are bound
            with a constant lens and the state need not carry
            ``repulsion_parameters``; with a ``probe``, the cache is read from
            ``state.repulsion_cache``.
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting the
            optimizer DOFs. ``None`` computes no gradients (the default). Composed with
            ``GRAPH_GEOMETRY`` into the potential's gradient lens.

    Returns:
        Configured repulsion [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if gradient is not None:
        gradient_lens = GRAPH_GEOMETRY.nest(gradient)
        patch_idx_view = position_and_cell_idx_view
    if parameters is not None:
        param_view = const_lens(parameters)
    else:
        param_view = state.focus(
            lambda x: (
                x.repulsion_parameters.data
                if isinstance(x.repulsion_parameters, HasCache)
                else x.repulsion_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.repulsion_parameters.data)
            cache_view = state.focus(lambda x: x.repulsion_parameters.cache)
        else:
            cache_view = state.focus(lambda x: x.repulsion_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_repulsion_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.repulsion_edge_indices),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        out_cache_lens=cache_view,
    )


# --------------------------------------------------------------------------- #
# Cutoff-shifted power-law repulsion
# --------------------------------------------------------------------------- #
class IsCutoffRepulsionState[Params](IsRepulsionGraphState, Protocol):
    """:class:`IsRepulsionGraphState` that also carries cutoff repulsion parameters."""

    @property
    def cutoff_repulsion_parameters(self) -> Params: ...


class IsCachedCutoffRepulsionState[Cache](IsRepulsionGraphState, Protocol):
    """:class:`IsRepulsionGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def cutoff_repulsion_cache(self) -> Cache: ...


@overload
def make_cutoff_repulsion_from_state[State](
    state: Lens[
        State, IsCutoffRepulsionState[MaybeCached[CutoffRepulsionParameters, Any]]
    ],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_cutoff_repulsion_from_state[State](
    state: Lens[
        State, IsCutoffRepulsionState[MaybeCached[CutoffRepulsionParameters, Any]]
    ],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_cutoff_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsCutoffRepulsionState[
            HasCache[CutoffRepulsionParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_cutoff_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsCutoffRepulsionState[
            HasCache[
                CutoffRepulsionParameters, PotentialOut[PositionsAndCell, EmptyType]
            ]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


@overload
def make_cutoff_repulsion_from_state[State](
    state: Lens[State, IsRepulsionGraphState],
    probe: None = None,
    *,
    parameters: CutoffRepulsionParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_cutoff_repulsion_from_state[State](
    state: Lens[State, IsRepulsionGraphState],
    probe: None = None,
    *,
    parameters: CutoffRepulsionParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_cutoff_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedCutoffRepulsionState[PotentialOut[EmptyType, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: CutoffRepulsionParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_cutoff_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedCutoffRepulsionState[PotentialOut[PositionsAndCell, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: CutoffRepulsionParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


def make_cutoff_repulsion_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: CutoffRepulsionParameters | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
) -> Any:
    """Create a cutoff-shifted power-law repulsion potential.

    Convenience wrapper around
    [make_cutoff_repulsion_potential][kups.potential.classical.repulsion.make_cutoff_repulsion_potential].

    Args:
        state: Lens into the sub-state providing particles, cell, and
            ``repulsion_edge_indices`` (plus ``cutoff_repulsion_parameters`` when
            ``parameters`` is not given).
        probe: If provided, detects particle changes and supplies the
            before/after fixed-edge neighbor lists for incremental updates.
        parameters: Constant cutoff repulsion parameters. When given they are
            bound with a constant lens and the state need not carry
            ``cutoff_repulsion_parameters``; with a ``probe``, the cache is read
            from ``state.cutoff_repulsion_cache``.
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting the
            optimizer DOFs. ``None`` computes no gradients (the default).

    Returns:
        Configured cutoff repulsion [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if gradient is not None:
        gradient_lens = GRAPH_GEOMETRY.nest(gradient)
        patch_idx_view = position_and_cell_idx_view
    if parameters is not None:
        param_view = const_lens(parameters)
    else:
        param_view = state.focus(
            lambda x: (
                x.cutoff_repulsion_parameters.data
                if isinstance(x.cutoff_repulsion_parameters, HasCache)
                else x.cutoff_repulsion_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.cutoff_repulsion_parameters.data)
            cache_view = state.focus(lambda x: x.cutoff_repulsion_parameters.cache)
        else:
            cache_view = state.focus(lambda x: x.cutoff_repulsion_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_cutoff_repulsion_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.repulsion_edge_indices),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        out_cache_lens=cache_view,
    )


# --------------------------------------------------------------------------- #
# Exponential repulsion
# --------------------------------------------------------------------------- #
class IsExpRepulsionState[Params](IsRepulsionGraphState, Protocol):
    """:class:`IsRepulsionGraphState` that also carries exponential repulsion parameters."""

    @property
    def exp_repulsion_parameters(self) -> Params: ...


class IsCachedExpRepulsionState[Cache](IsRepulsionGraphState, Protocol):
    """:class:`IsRepulsionGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def exp_repulsion_cache(self) -> Cache: ...


@overload
def make_exp_repulsion_from_state[State](
    state: Lens[State, IsExpRepulsionState[MaybeCached[ExpRepulsionParameters, Any]]],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_exp_repulsion_from_state[State](
    state: Lens[State, IsExpRepulsionState[MaybeCached[ExpRepulsionParameters, Any]]],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_exp_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsExpRepulsionState[
            HasCache[ExpRepulsionParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_exp_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsExpRepulsionState[
            HasCache[ExpRepulsionParameters, PotentialOut[PositionsAndCell, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


@overload
def make_exp_repulsion_from_state[State](
    state: Lens[State, IsRepulsionGraphState],
    probe: None = None,
    *,
    parameters: ExpRepulsionParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_exp_repulsion_from_state[State](
    state: Lens[State, IsRepulsionGraphState],
    probe: None = None,
    *,
    parameters: ExpRepulsionParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_exp_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCachedExpRepulsionState[PotentialOut[EmptyType, EmptyType]]],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: ExpRepulsionParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_exp_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedExpRepulsionState[PotentialOut[PositionsAndCell, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: ExpRepulsionParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


def make_exp_repulsion_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: ExpRepulsionParameters | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
) -> Any:
    """Create an exponential repulsion potential, optionally with incremental updates.

    Convenience wrapper around
    [make_exp_repulsion_potential][kups.potential.classical.repulsion.make_exp_repulsion_potential].

    Args:
        state: Lens into the sub-state providing particles, cell, and
            ``repulsion_edge_indices`` (plus ``exp_repulsion_parameters`` when
            ``parameters`` is not given).
        probe: If provided, detects particle changes and supplies the
            before/after fixed-edge neighbor lists for incremental updates.
        parameters: Constant exponential repulsion parameters. When given they
            are bound with a constant lens and the state need not carry
            ``exp_repulsion_parameters``; with a ``probe``, the cache is read
            from ``state.exp_repulsion_cache``.
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting the
            optimizer DOFs. ``None`` computes no gradients (the default).

    Returns:
        Configured exponential repulsion [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if gradient is not None:
        gradient_lens = GRAPH_GEOMETRY.nest(gradient)
        patch_idx_view = position_and_cell_idx_view
    if parameters is not None:
        param_view = const_lens(parameters)
    else:
        param_view = state.focus(
            lambda x: (
                x.exp_repulsion_parameters.data
                if isinstance(x.exp_repulsion_parameters, HasCache)
                else x.exp_repulsion_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.exp_repulsion_parameters.data)
            cache_view = state.focus(lambda x: x.exp_repulsion_parameters.cache)
        else:
            cache_view = state.focus(lambda x: x.exp_repulsion_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_exp_repulsion_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.repulsion_edge_indices),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        out_cache_lens=cache_view,
    )


# --------------------------------------------------------------------------- #
# Cutoff-shifted exponential repulsion
# --------------------------------------------------------------------------- #
class IsCutoffExpRepulsionState[Params](IsRepulsionGraphState, Protocol):
    """:class:`IsRepulsionGraphState` that also carries cutoff exponential repulsion parameters."""

    @property
    def cutoff_exp_repulsion_parameters(self) -> Params: ...


class IsCachedCutoffExpRepulsionState[Cache](IsRepulsionGraphState, Protocol):
    """:class:`IsRepulsionGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def cutoff_exp_repulsion_cache(self) -> Cache: ...


@overload
def make_cutoff_exp_repulsion_from_state[State](
    state: Lens[
        State,
        IsCutoffExpRepulsionState[MaybeCached[CutoffExpRepulsionParameters, Any]],
    ],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_cutoff_exp_repulsion_from_state[State](
    state: Lens[
        State,
        IsCutoffExpRepulsionState[MaybeCached[CutoffExpRepulsionParameters, Any]],
    ],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_cutoff_exp_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsCutoffExpRepulsionState[
            HasCache[CutoffExpRepulsionParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_cutoff_exp_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsCutoffExpRepulsionState[
            HasCache[
                CutoffExpRepulsionParameters, PotentialOut[PositionsAndCell, EmptyType]
            ]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


@overload
def make_cutoff_exp_repulsion_from_state[State](
    state: Lens[State, IsRepulsionGraphState],
    probe: None = None,
    *,
    parameters: CutoffExpRepulsionParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_cutoff_exp_repulsion_from_state[State](
    state: Lens[State, IsRepulsionGraphState],
    probe: None = None,
    *,
    parameters: CutoffExpRepulsionParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_cutoff_exp_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedCutoffExpRepulsionState[PotentialOut[EmptyType, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: CutoffExpRepulsionParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_cutoff_exp_repulsion_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsCachedCutoffExpRepulsionState[PotentialOut[PositionsAndCell, EmptyType]],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: CutoffExpRepulsionParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


def make_cutoff_exp_repulsion_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: CutoffExpRepulsionParameters | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
) -> Any:
    """Create a cutoff-shifted exponential repulsion potential.

    Convenience wrapper around
    [make_cutoff_exp_repulsion_potential][kups.potential.classical.repulsion.make_cutoff_exp_repulsion_potential].

    Args:
        state: Lens into the sub-state providing particles, cell, and
            ``repulsion_edge_indices`` (plus ``cutoff_exp_repulsion_parameters``
            when ``parameters`` is not given).
        probe: If provided, detects particle changes and supplies the
            before/after fixed-edge neighbor lists for incremental updates.
        parameters: Constant cutoff exponential repulsion parameters. When given
            they are bound with a constant lens and the state need not carry
            ``cutoff_exp_repulsion_parameters``; with a ``probe``, the cache is
            read from ``state.cutoff_exp_repulsion_cache``.
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting the
            optimizer DOFs. ``None`` computes no gradients (the default).

    Returns:
        Configured cutoff exponential repulsion [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if gradient is not None:
        gradient_lens = GRAPH_GEOMETRY.nest(gradient)
        patch_idx_view = position_and_cell_idx_view
    if parameters is not None:
        param_view = const_lens(parameters)
    else:
        param_view = state.focus(
            lambda x: (
                x.cutoff_exp_repulsion_parameters.data
                if isinstance(x.cutoff_exp_repulsion_parameters, HasCache)
                else x.cutoff_exp_repulsion_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.cutoff_exp_repulsion_parameters.data)
            cache_view = state.focus(
                lambda x: x.cutoff_exp_repulsion_parameters.cache
            )
        else:
            cache_view = state.focus(lambda x: x.cutoff_exp_repulsion_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_cutoff_exp_repulsion_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.repulsion_edge_indices),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        out_cache_lens=cache_view,
    )
