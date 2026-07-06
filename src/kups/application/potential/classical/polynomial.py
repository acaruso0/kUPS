# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for the polynomial angle prior.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, angle indices, and parameters into the
state-agnostic factory in [kups.potential.classical.polynomial][].

The angle connectivity is read from the shared ``angle_edge_indices`` attribute
(the same one used by the harmonic angle). Parameters may live on the state
(``state.polynomial_angle_parameters``) or be passed directly via ``parameters=``;
in the latter case they are bound with a constant lens and the state need not
carry a parameter field. For incremental (probe) updates with constant
parameters, the cache is read from a conventional ``polynomial_angle_cache``
attribute.
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
from kups.potential.classical.polynomial import (
    IsBondedParticles,
    PolynomialAngleParameters,
    make_polynomial_angle_potential,
)
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    position_and_cell_idx_view,
)
from kups.potential.common.graph import GRAPH_GEOMETRY, IsGraphProbe


class IsPolynomialAngleGraphState(
    IsState[IsBondedParticles, HasCell[AnyPeriodicity]], Protocol
):
    """Particles, systems, and angle indices for a polynomial angle graph (no parameters)."""

    @property
    def angle_edge_indices(self) -> Index[ParticleId]: ...


class IsPolynomialAngleState[Params](IsPolynomialAngleGraphState, Protocol):
    """:class:`IsPolynomialAngleGraphState` that also carries parameters on the state."""

    @property
    def polynomial_angle_parameters(self) -> Params: ...


class IsCachedPolynomialAngleState[Cache](IsPolynomialAngleGraphState, Protocol):
    """:class:`IsPolynomialAngleGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def polynomial_angle_cache(self) -> Cache: ...


@overload
def make_polynomial_angle_from_state[State](
    state: Lens[
        State, IsPolynomialAngleState[MaybeCached[PolynomialAngleParameters, Any]]
    ],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_polynomial_angle_from_state[State](
    state: Lens[
        State, IsPolynomialAngleState[MaybeCached[PolynomialAngleParameters, Any]]
    ],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_polynomial_angle_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsPolynomialAngleState[
            HasCache[PolynomialAngleParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_polynomial_angle_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsPolynomialAngleState[
            HasCache[
                PolynomialAngleParameters, PotentialOut[PositionsAndCell, EmptyType]
            ]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


@overload
def make_polynomial_angle_from_state[State](
    state: Lens[State, IsPolynomialAngleGraphState],
    probe: None = None,
    *,
    parameters: PolynomialAngleParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_polynomial_angle_from_state[State](
    state: Lens[State, IsPolynomialAngleGraphState],
    probe: None = None,
    *,
    parameters: PolynomialAngleParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_polynomial_angle_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedPolynomialAngleState[PotentialOut[EmptyType, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    parameters: PolynomialAngleParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_polynomial_angle_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedPolynomialAngleState[PotentialOut[PositionsAndCell, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    parameters: PolynomialAngleParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


def make_polynomial_angle_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: PolynomialAngleParameters | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
) -> Any:
    """Create a polynomial angle potential, optionally with incremental updates.

    Convenience wrapper around
    [make_polynomial_angle_potential][kups.potential.classical.polynomial.make_polynomial_angle_potential].

    Args:
        state: Lens into the sub-state providing particles, cell, and
            ``angle_edge_indices`` (plus ``polynomial_angle_parameters`` when
            ``parameters`` is not given).
        probe: If provided, detects particle changes and supplies the
            before/after fixed-edge neighbor lists for incremental updates.
        parameters: Constant polynomial angle parameters. When given they are
            bound with a constant lens and the state need not carry
            ``polynomial_angle_parameters``; with a ``probe``, the cache is read
            from ``state.polynomial_angle_cache``.
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting the
            optimizer DOFs. ``None`` computes no gradients (the default). Composed with
            ``GRAPH_GEOMETRY`` into the potential's gradient lens.

    Returns:
        Configured polynomial angle [Potential][kups.core.potential.Potential].
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
                x.polynomial_angle_parameters.data
                if isinstance(x.polynomial_angle_parameters, HasCache)
                else x.polynomial_angle_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.polynomial_angle_parameters.data)
            cache_view = state.focus(lambda x: x.polynomial_angle_parameters.cache)
        else:
            cache_view = state.focus(lambda x: x.polynomial_angle_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_polynomial_angle_potential(
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
