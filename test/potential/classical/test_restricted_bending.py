# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the restricted-quartic angle prior."""

import jax
import jax.numpy as jnp

from kups.core.data.index import Index
from kups.core.neighborlist import Edges
from kups.potential.classical.restricted_bending import (
    RestrictedQuarticParameters,
    restricted_quartic_energy,
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


class TestRestrictedQuarticEnergy:
    """V = a cos^4 + b cos^3 + c cos^2 + d cos + k/sin^2 + v0."""

    @classmethod
    def setup_class(cls):
        cls.a = 1.0
        cls.b = -0.5
        cls.c = 2.0
        cls.d = 0.3
        cls.k = 0.1
        cls.v0 = 0.4

        def one(v):
            return jnp.array([[[v]]])

        cls.params = RestrictedQuarticParameters(
            labels=("A",),
            a=one(cls.a),
            b=one(cls.b),
            c=one(cls.c),
            d=one(cls.d),
            k=one(cls.k),
            v0=one(cls.v0),
        )

    def _energy(self, positions):
        graph = _angle_graph(positions)
        return jax.jit(restricted_quartic_energy)(
            GraphPotentialInput(graph=graph, parameters=self.params)
        ).data.data[0]

    def _expected(self, theta):
        cos = jnp.cos(theta)
        sin2 = jnp.sin(theta) ** 2
        return (
            self.a * cos**4
            + self.b * cos**3
            + self.c * cos**2
            + self.d * cos
            + self.k / sin2
            + self.v0
        )

    def test_formula(self):
        for theta_deg in (60.0, 90.0, 110.0, 135.0):
            theta = jnp.deg2rad(theta_deg)
            assert jnp.isclose(
                self._energy(_positions_for_angle(theta)),
                self._expected(theta),
                rtol=1e-6,
            )

    def test_barrier_diverges_near_linear(self):
        # The k/sin^2 term blows up as theta -> pi (linear).
        near_linear = self._energy(_positions_for_angle(jnp.deg2rad(179.0)))
        mid = self._energy(_positions_for_angle(jnp.deg2rad(90.0)))
        assert near_linear > mid
        assert near_linear > 100.0

    def test_gradient_finite(self):
        grad = jax.jit(jax.grad(self._energy))(_positions_for_angle(jnp.deg2rad(100.0)))
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.linalg.norm(grad) > 1e-6
