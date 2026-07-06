# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the polynomial angle prior."""

import jax
import jax.numpy as jnp

from kups.core.data.index import Index
from kups.core.neighborlist import Edges
from kups.potential.classical.polynomial import (
    PolynomialAngleParameters,
    polynomial_angle_energy,
)
from kups.potential.common.graph import GraphPotentialInput, HyperGraph

from .conftest import make_particles, make_systems

_CELLS_LV = jnp.eye(3)[None] * 20.0


def _positions_for_angle(theta) -> jax.Array:
    """Three atoms with the vertex at atom 0 subtending ``theta``."""
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [jnp.cos(theta), jnp.sin(theta), 0.0],
        ]
    )


def _angle_graph(positions):
    particles = make_particles(positions, ["A", "A", "A"], [0, 0, 0])
    systems = make_systems(_CELLS_LV)
    edges = Edges(
        indices=Index(particles.keys, jnp.array([[0, 1, 2]])),
        shifts=jnp.zeros((1, 2, 3)),
    )
    return HyperGraph(particles, systems, edges)


class TestPolynomialAngleCosine:
    """Cosine feature: V = v0 + sum_n k_n (cos theta)^n."""

    @classmethod
    def setup_class(cls):
        cls.ks = jnp.array([1.0, -2.0, 0.5, 0.25]).reshape(4, 1, 1, 1)
        cls.v0 = jnp.array([[[0.3]]])
        cls.params = PolynomialAngleParameters(
            labels=("A",), ks=cls.ks, v0=cls.v0, feature="cosine"
        )

    def _energy(self, positions):
        graph = _angle_graph(positions)
        return jax.jit(polynomial_angle_energy)(
            GraphPotentialInput(graph=graph, parameters=self.params)
        ).data.data[0]

    def test_formula(self):
        for theta_deg in (60.0, 90.0, 120.0, 150.0):
            theta = jnp.deg2rad(theta_deg)
            c = jnp.cos(theta)
            ks = self.ks.reshape(4)
            expected = (
                self.v0[0, 0, 0]
                + ks[0] * c
                + ks[1] * c**2
                + ks[2] * c**3
                + ks[3] * c**4
            )
            assert jnp.isclose(
                self._energy(_positions_for_angle(theta)), expected, rtol=1e-6
            )

    def test_gradient_finite_and_nonzero(self):
        grad = jax.jit(jax.grad(self._energy))(_positions_for_angle(jnp.deg2rad(100.0)))
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.linalg.norm(grad) > 1e-6

    def test_degree_inferred_from_ks(self):
        # A degree-2 polynomial should agree with the explicit 2-term formula.
        ks2 = jnp.array([1.5, -0.5]).reshape(2, 1, 1, 1)
        params = PolynomialAngleParameters(
            labels=("A",), ks=ks2, v0=self.v0, feature="cosine"
        )
        theta = jnp.deg2rad(100.0)
        c = jnp.cos(theta)
        expected = self.v0[0, 0, 0] + 1.5 * c - 0.5 * c**2
        graph = _angle_graph(_positions_for_angle(theta))
        got = polynomial_angle_energy(
            GraphPotentialInput(graph=graph, parameters=params)
        ).data.data[0]
        assert jnp.isclose(got, expected, rtol=1e-6)


class TestPolynomialAngleRaw:
    """Angle feature: V = v0 + sum_n k_n theta^n, theta in degrees."""

    @classmethod
    def setup_class(cls):
        cls.ks = jnp.array([0.01, -1e-4, 1e-6, 1e-9]).reshape(4, 1, 1, 1)
        cls.v0 = jnp.array([[[0.0]]])
        cls.params = PolynomialAngleParameters(
            labels=("A",), ks=cls.ks, v0=cls.v0, feature="angle"
        )

    def _energy(self, positions):
        graph = _angle_graph(positions)
        return jax.jit(polynomial_angle_energy)(
            GraphPotentialInput(graph=graph, parameters=self.params)
        ).data.data[0]

    def test_formula_degrees(self):
        theta_deg = 110.0
        ks = self.ks.reshape(4)
        expected = sum(ks[n] * theta_deg ** (n + 1) for n in range(4))
        assert jnp.isclose(
            self._energy(_positions_for_angle(jnp.deg2rad(theta_deg))),
            expected,
            rtol=1e-6,
        )
