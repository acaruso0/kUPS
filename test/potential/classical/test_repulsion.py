# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the repulsion prior implementations."""

import jax
import jax.numpy as jnp
import pytest

from kups.core.data.index import Index
from kups.core.neighborlist import Edges
from kups.potential.classical.repulsion import (
    CutoffExpRepulsionParameters,
    CutoffRepulsionParameters,
    ExpRepulsionParameters,
    RepulsionParameters,
    cutoff_exp_repulsion_energy,
    cutoff_repulsion_energy,
    exp_repulsion_energy,
    repulsion_energy,
)
from kups.potential.common.graph import GraphPotentialInput, HyperGraph

from .conftest import make_particles, make_systems

_LABELS = ("A", "B")
_SPECIES = ["A", "B"]
_SYSTEM_IDS = [0, 0]
_CELLS_LV = jnp.eye(3)[None] * 10.0


def _positions_for_distance(r) -> jax.Array:
    """Positions for two atoms separated by distance ``r`` along the x-axis."""
    return jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])


def _make_graph(positions):
    particles = make_particles(positions, _SPECIES, _SYSTEM_IDS)
    systems = make_systems(_CELLS_LV)
    edges = Edges(
        indices=Index(particles.keys, jnp.array([[0, 1]])),
        shifts=jnp.array([[[0.0, 0.0, 0.0]]]),
    )
    return HyperGraph(particles, systems, edges)


def _energy_fn(energy_fn, params):
    jit_fn = jax.jit(energy_fn)

    def _energy(positions):
        graph = _make_graph(positions)
        return jit_fn(
            GraphPotentialInput(graph=graph, parameters=params)
        ).data.data[0]

    return _energy


class TestRepulsionEnergy:
    """Test the (sigma / r)^6 power-law repulsion."""

    @classmethod
    def setup_class(cls):
        cls.sigma = jnp.array([[1.0, 1.2], [1.2, 1.5]])
        cls.params = RepulsionParameters(labels=_LABELS, sigma=cls.sigma)
        cls.energy = staticmethod(_energy_fn(repulsion_energy, cls.params))
        cls.sigma_ab = cls.sigma[0, 1]

    def test_energy_formula(self):
        for r in (0.8, 1.2, 2.0, 3.5):
            expected = (self.sigma_ab / r) ** 6
            assert jnp.isclose(
                self.energy(_positions_for_distance(r)), expected, rtol=1e-6
            )

    def test_monotonic_decreasing(self):
        e_close = self.energy(_positions_for_distance(0.9))
        e_far = self.energy(_positions_for_distance(2.5))
        assert e_close > e_far > 0.0

    def test_gradient_nonzero_and_attractive_inward(self):
        grad = jax.jit(jax.grad(self.energy))(_positions_for_distance(1.0))
        # Repulsion pushes the two atoms apart: atom 0 forced toward -x, atom 1 +x.
        # Force = -grad, so grad[0, 0] > 0 and grad[1, 0] < 0.
        assert grad[0, 0] > 1e-6
        assert grad[1, 0] < -1e-6

    def test_assertion_wrong_order(self):
        positions = _positions_for_distance(1.0)
        particles = make_particles(positions, _SPECIES, _SYSTEM_IDS)
        systems = make_systems(_CELLS_LV)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 0]])),
            shifts=jnp.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        )
        graph = HyperGraph(particles, systems, edges)
        with pytest.raises(AssertionError, match="pairwise interactions"):
            repulsion_energy(GraphPotentialInput(graph=graph, parameters=self.params))


class TestCutoffRepulsionEnergy:
    """Test the cutoff-shifted power-law repulsion."""

    @classmethod
    def setup_class(cls):
        cls.sigma = jnp.array([[1.0, 1.2], [1.2, 1.5]])
        cls.cutoff = 2.0
        cls.params = CutoffRepulsionParameters(
            labels=_LABELS, sigma=cls.sigma, cutoff=cls.cutoff
        )
        cls.energy = staticmethod(_energy_fn(cutoff_repulsion_energy, cls.params))
        cls.sigma_ab = cls.sigma[0, 1]

    def test_zero_beyond_cutoff(self):
        assert jnp.isclose(
            self.energy(_positions_for_distance(self.cutoff + 0.5)), 0.0, atol=1e-12
        )
        assert jnp.isclose(
            self.energy(_positions_for_distance(3.0)), 0.0, atol=1e-12
        )

    def test_shifted_formula_inside_cutoff(self):
        r = 1.0
        u = (self.sigma_ab / r) ** 6
        uc = (self.sigma_ab / self.cutoff) ** 6
        duc = -6.0 * uc / self.cutoff
        expected = u - uc - (r - self.cutoff) * duc
        assert jnp.isclose(
            self.energy(_positions_for_distance(r)), expected, rtol=1e-6
        )

    def test_c1_continuity_at_cutoff(self):
        # Energy and force both vanish as r -> cutoff from below.
        e_near = self.energy(_positions_for_distance(self.cutoff - 1e-3))
        assert jnp.isclose(e_near, 0.0, atol=1e-4)
        grad = jax.jit(jax.grad(self.energy))(
            _positions_for_distance(self.cutoff - 1e-3)
        )
        assert jnp.linalg.norm(grad) < 1e-2


class TestExpRepulsionEnergy:
    """Test the exponential (exp-6) repulsion."""

    @classmethod
    def setup_class(cls):
        cls.alpha = jnp.array([[2.0, 1.8], [1.8, 2.2]])
        cls.r0 = jnp.array([[1.0, 1.2], [1.2, 1.5]])
        cls.params = ExpRepulsionParameters(
            labels=_LABELS, alpha=cls.alpha, r0=cls.r0
        )
        cls.energy = staticmethod(_energy_fn(exp_repulsion_energy, cls.params))
        cls.alpha_ab = cls.alpha[0, 1]
        cls.r0_ab = cls.r0[0, 1]

    def test_energy_formula(self):
        for r in (0.8, 1.2, 2.0):
            expected = (6.0 / self.alpha_ab) * jnp.exp(
                self.alpha_ab * (1.0 - r / self.r0_ab)
            )
            assert jnp.isclose(
                self.energy(_positions_for_distance(r)), expected, rtol=1e-6
            )

    def test_value_at_r0(self):
        assert jnp.isclose(
            self.energy(_positions_for_distance(self.r0_ab)),
            6.0 / self.alpha_ab,
            rtol=1e-6,
        )

    def test_monotonic_decreasing(self):
        e_close = self.energy(_positions_for_distance(0.9))
        e_far = self.energy(_positions_for_distance(2.5))
        assert e_close > e_far > 0.0


class TestCutoffExpRepulsionEnergy:
    """Test the cutoff-shifted exponential repulsion."""

    @classmethod
    def setup_class(cls):
        cls.alpha = jnp.array([[2.0, 1.8], [1.8, 2.2]])
        cls.r0 = jnp.array([[1.0, 1.2], [1.2, 1.5]])
        cls.cutoff = 2.0
        cls.params = CutoffExpRepulsionParameters(
            labels=_LABELS, alpha=cls.alpha, r0=cls.r0, cutoff=cls.cutoff
        )
        cls.energy = staticmethod(_energy_fn(cutoff_exp_repulsion_energy, cls.params))
        cls.alpha_ab = cls.alpha[0, 1]
        cls.r0_ab = cls.r0[0, 1]

    def test_zero_beyond_cutoff(self):
        assert jnp.isclose(
            self.energy(_positions_for_distance(self.cutoff + 0.5)), 0.0, atol=1e-12
        )

    def test_shifted_formula_inside_cutoff(self):
        r = 1.0
        u = (6.0 / self.alpha_ab) * jnp.exp(self.alpha_ab * (1.0 - r / self.r0_ab))
        uc = (6.0 / self.alpha_ab) * jnp.exp(
            self.alpha_ab * (1.0 - self.cutoff / self.r0_ab)
        )
        duc = -(self.alpha_ab / self.r0_ab) * uc
        expected = u - uc - (r - self.cutoff) * duc
        assert jnp.isclose(
            self.energy(_positions_for_distance(r)), expected, rtol=1e-6
        )

    def test_c1_continuity_at_cutoff(self):
        e_near = self.energy(_positions_for_distance(self.cutoff - 1e-3))
        assert jnp.isclose(e_near, 0.0, atol=1e-4)
        grad = jax.jit(jax.grad(self.energy))(
            _positions_for_distance(self.cutoff - 1e-3)
        )
        assert jnp.linalg.norm(grad) < 1e-2
