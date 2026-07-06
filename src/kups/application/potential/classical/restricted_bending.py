# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for the restricted-quartic angle prior.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, angle indices, and parameters into the
state-agnostic factory in [kups.potential.classical.restricted_bending][].

The angle connectivity is read from the shared ``angle_edge_indices`` attribute.
Parameters may live on the state (``state.restricted_quartic_parameters``) or be
passed directly via ``parameters=``; in the latter case they are bound with a
constant lens and the state need not carry a parameter field. For incremental
(probe) updates with constant parameters, the cache is read from a conventional
``restricted_quartic_cache`` attribute.
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
from kups.potential.classical.restricted_bending import (
    IsBondedParticles,
    RestrictedQuarticParameters,
    make_restricted_quartic_potential,
)
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    position_and_cell_idx_view,
)
from kups.potential.common.graph import GRAPH_GEOMETRY, IsGraphProbe


class IsRestrictedQuarticGraphState(
    IsState[IsBondedParticles, HasCell[AnyPeriodicity]], Protocol
):
    """Particles, systems, and angle indices for a restricted-quartic graph (no parameters)."""

    @property
    def angle_edge_indices(self) -> Index[ParticleId]: ...


class IsRestrictedQuarticState[Params](IsRestrictedQuarticGraphState, Protocol):
    """:class:`IsRestrictedQuarticGraphState` that also carries parameters on the state."""

    @property
    def restricted_quartic_parameters(self) -> Params: ...


class IsCachedRestrictedQuarticState[Cache](IsRestrictedQuarticGraphState, Protocol):
    """:class:`IsRestrictedQuarticGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def restricted_quartic_cache(self) -> Cache: ...


@overload
def make_restricted_quartic_from_state[State](
    state: Lens[
        State, IsRestrictedQuarticState[MaybeCached[RestrictedQuarticParameters, Any]]
    ],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_restricted_quartic_from_state[State](
    state: Lens[
        State, IsRestrictedQuarticState[MaybeCached[RestrictedQuarticParameters, Any]]
    ],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_restricted_quartic_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsRestrictedQuarticState[
            HasCache[RestrictedQuarticParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_restricted_quartic_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsRestrictedQuarticState[
            HasCache[
                RestrictedQuarticParameters, PotentialOut[PositionsAndCell, EmptyType]
            ]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


@overload
def make_restricted_quartic_from_state[State](
    state: Lens[State, IsRestrictedQuarticGraphState],
    probe: None = None,
    *,
    parameters: RestrictedQuarticParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_restricted_quartic_from_state[State](
    state: Lens[State, IsRestrictedQuarticGraphState],
    probe: None = None,
    *,
    parameters: RestrictedQuarticParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_restricted_quartic_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedRestrictedQuarticState[PotentialOut[EmptyType, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    parameters: RestrictedQuarticParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_restricted_quartic_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsCachedRestrictedQuarticState[PotentialOut[PositionsAndCell, EmptyType]],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    parameters: RestrictedQuarticParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


def make_restricted_quartic_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: RestrictedQuarticParameters | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
) -> Any:
    """Create a restricted-quartic angle potential, optionally with incremental updates.

    Convenience wrapper around
    [make_restricted_quartic_potential][kups.potential.classical.restricted_bending.make_restricted_quartic_potential].

    Args:
        state: Lens into the sub-state providing particles, cell, and
            ``angle_edge_indices`` (plus ``restricted_quartic_parameters`` when
            ``parameters`` is not given).
        probe: If provided, detects particle changes and supplies the
            before/after fixed-edge neighbor lists for incremental updates.
        parameters: Constant restricted-quartic parameters. When given they are
            bound with a constant lens and the state need not carry
            ``restricted_quartic_parameters``; with a ``probe``, the cache is read
            from ``state.restricted_quartic_cache``.
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting the
            optimizer DOFs. ``None`` computes no gradients (the default). Composed with
            ``GRAPH_GEOMETRY`` into the potential's gradient lens.

    Returns:
        Configured restricted-quartic angle [Potential][kups.core.potential.Potential].
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
                x.restricted_quartic_parameters.data
                if isinstance(x.restricted_quartic_parameters, HasCache)
                else x.restricted_quartic_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.restricted_quartic_parameters.data)
            cache_view = state.focus(
                lambda x: x.restricted_quartic_parameters.cache
            )
        else:
            cache_view = state.focus(lambda x: x.restricted_quartic_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_restricted_quartic_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.angle_edge_indices),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        out_cache_lens=cache_view,
    )
