# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for the Fourier-series dihedral prior.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, dihedral indices, and parameters into the
state-agnostic factory in [kups.potential.classical.fourier_series][].

The dihedral connectivity is read from the shared ``dihedral_edge_indices``
attribute. Parameters may live on the state (``state.fourier_dihedral_parameters``)
or be passed directly via ``parameters=``; in the latter case they are bound with
a constant lens and the state need not carry a parameter field. For incremental
(probe) updates with constant parameters, the cache is read from a conventional
``fourier_dihedral_cache`` attribute.
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
from kups.potential.classical.fourier_series import (
    FourierDihedralParameters,
    IsBondedParticles,
    make_fourier_dihedral_potential,
)
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    position_and_cell_idx_view,
)
from kups.potential.common.graph import GRAPH_GEOMETRY, IsGraphProbe


class IsFourierDihedralGraphState(
    IsState[IsBondedParticles, HasCell[AnyPeriodicity]], Protocol
):
    """Particles, systems, and dihedral indices for a Fourier dihedral graph (no parameters)."""

    @property
    def dihedral_edge_indices(self) -> Index[ParticleId]: ...


class IsFourierDihedralState[Params](IsFourierDihedralGraphState, Protocol):
    """:class:`IsFourierDihedralGraphState` that also carries parameters on the state."""

    @property
    def fourier_dihedral_parameters(self) -> Params: ...


class IsCachedFourierDihedralState[Cache](IsFourierDihedralGraphState, Protocol):
    """:class:`IsFourierDihedralGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def fourier_dihedral_cache(self) -> Cache: ...


@overload
def make_fourier_dihedral_from_state[State](
    state: Lens[
        State, IsFourierDihedralState[MaybeCached[FourierDihedralParameters, Any]]
    ],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_fourier_dihedral_from_state[State](
    state: Lens[
        State, IsFourierDihedralState[MaybeCached[FourierDihedralParameters, Any]]
    ],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_fourier_dihedral_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsFourierDihedralState[
            HasCache[FourierDihedralParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]],
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_fourier_dihedral_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsFourierDihedralState[
            HasCache[
                FourierDihedralParameters, PotentialOut[PositionsAndCell, EmptyType]
            ]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


@overload
def make_fourier_dihedral_from_state[State](
    state: Lens[State, IsFourierDihedralGraphState],
    probe: None = None,
    *,
    parameters: FourierDihedralParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_fourier_dihedral_from_state[State](
    state: Lens[State, IsFourierDihedralGraphState],
    probe: None = None,
    *,
    parameters: FourierDihedralParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_fourier_dihedral_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedFourierDihedralState[PotentialOut[EmptyType, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]],
    *,
    parameters: FourierDihedralParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_fourier_dihedral_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedFourierDihedralState[PotentialOut[PositionsAndCell, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]],
    *,
    parameters: FourierDihedralParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


def make_fourier_dihedral_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: FourierDihedralParameters | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
) -> Any:
    """Create a Fourier-series dihedral potential, optionally with incremental updates.

    Convenience wrapper around
    [make_fourier_dihedral_potential][kups.potential.classical.fourier_series.make_fourier_dihedral_potential].

    Args:
        state: Lens into the sub-state providing particles, cell, and
            ``dihedral_edge_indices`` (plus ``fourier_dihedral_parameters`` when
            ``parameters`` is not given).
        probe: If provided, detects particle changes and supplies the
            before/after fixed-edge neighbor lists for incremental updates.
        parameters: Constant Fourier dihedral parameters. When given they are
            bound with a constant lens and the state need not carry
            ``fourier_dihedral_parameters``; with a ``probe``, the cache is read
            from ``state.fourier_dihedral_cache``.
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting the
            optimizer DOFs. ``None`` computes no gradients (the default). Composed with
            ``GRAPH_GEOMETRY`` into the potential's gradient lens.

    Returns:
        Configured Fourier dihedral [Potential][kups.core.potential.Potential].
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
                x.fourier_dihedral_parameters.data
                if isinstance(x.fourier_dihedral_parameters, HasCache)
                else x.fourier_dihedral_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.fourier_dihedral_parameters.data)
            cache_view = state.focus(lambda x: x.fourier_dihedral_parameters.cache)
        else:
            cache_view = state.focus(lambda x: x.fourier_dihedral_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_fourier_dihedral_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.dihedral_edge_indices),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        out_cache_lens=cache_view,
    )
