# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cosine-feature harmonic angle and harmonic improper potentials.

The degrees ``feature="angle"`` bond/angle terms are exercised end-to-end via
``test/application/potential/test_potential_update.py``; here we focus on the
newly added cosine-feature angle and the (optionally shifted-periodic) harmonic
improper.
"""

import jax
import jax.numpy as jnp

from kups.core.data.index import Index
from kups.core.neighborlist import Edges
from kups.potential.classical.harmonic import (
    HarmonicAngleParameters,
    HarmonicImproperParameters,
    harmonic_angle_energy,
    harmonic_improper_energy,
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


def _positions_for_dihedral(phi) -> jax.Array:
    """Four atoms with torsion angle ``phi`` (copied convention from the UFF dihedral test)."""
    i = jnp.array([0.0, 0.0, 0.0])
    j = jnp.array([1.0, 0.0, 0.0])
    k = jnp.array([1.5, 1.0, 0.0])

    fourth_trans = jnp.array([2.5, 1.0, 0.0])
    jk = k - j
    jk_norm = jk / jnp.linalg.norm(jk)
    kl_trans = fourth_trans - k

    angle = phi - jnp.pi
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    kl_rot = (
        kl_trans * cos_a
        + jnp.cross(jk_norm, kl_trans) * sin_a
        + jk_norm * jnp.dot(jk_norm, kl_trans) * (1 - cos_a)
    )
    fourth = k + kl_rot
    return jnp.stack([i, j, k, fourth])


def _angle_graph(positions):
    particles = make_particles(positions, ["A", "A", "A"], [0, 0, 0])
    systems = make_systems(_CELLS_LV)
    edges = Edges(
        indices=Index(particles.keys, jnp.array([[0, 1, 2]])),
        shifts=jnp.zeros((1, 2, 3)),
    )
    return HyperGraph(particles, systems, edges)


def _dihedral_graph(positions):
    particles = make_particles(positions, ["A", "A", "A", "A"], [0, 0, 0, 0])
    systems = make_systems(_CELLS_LV)
    edges = Edges(
        indices=Index(particles.keys, jnp.array([[0, 1, 2, 3]])),
        shifts=jnp.zeros((1, 3, 3)),
    )
    return HyperGraph(particles, systems, edges)


class TestHarmonicAngleCosineFeature:
    """Test HarmonicAngle with ``feature="cosine"`` -> k(cos theta - cos theta0)^2."""

    @classmethod
    def setup_class(cls):
        cls.theta0_deg = 109.5
        cls.k = 50.0
        cls.params = HarmonicAngleParameters(
            labels=("A",),
            theta0=jnp.array([[[cls.theta0_deg]]]),
            k=jnp.array([[[cls.k]]]),
            feature="cosine",
        )

    def _energy(self, positions):
        graph = _angle_graph(positions)
        return jax.jit(harmonic_angle_energy)(
            GraphPotentialInput(graph=graph, parameters=self.params)
        ).data.data[0]

    def test_zero_at_equilibrium(self):
        theta0 = jnp.deg2rad(self.theta0_deg)
        assert jnp.isclose(
            self._energy(_positions_for_angle(theta0)), 0.0, atol=1e-10
        )

    def test_cosine_formula(self):
        theta = jnp.deg2rad(90.0)
        expected = self.k * (
            jnp.cos(theta) - jnp.cos(jnp.deg2rad(self.theta0_deg))
        ) ** 2
        assert jnp.isclose(
            self._energy(_positions_for_angle(theta)), expected, rtol=1e-6
        )

    def test_differs_from_angle_feature(self):
        theta = jnp.deg2rad(90.0)
        angle_params = HarmonicAngleParameters(
            labels=("A",),
            theta0=jnp.array([[[self.theta0_deg]]]),
            k=jnp.array([[[self.k]]]),
            feature="angle",
        )
        graph = _angle_graph(_positions_for_angle(theta))
        angle_e = harmonic_angle_energy(
            GraphPotentialInput(graph=graph, parameters=angle_params)
        ).data.data[0]
        assert not jnp.isclose(self._energy(_positions_for_angle(theta)), angle_e)

    def test_gradient_zero_at_equilibrium(self):
        theta0 = jnp.deg2rad(self.theta0_deg)
        grad = jax.jit(jax.grad(self._energy))(_positions_for_angle(theta0))
        assert jnp.allclose(grad, 0.0, atol=1e-6)


class TestHarmonicImproper:
    """Test the harmonic improper (torsion) potential k(phi - phi0)^2."""

    @classmethod
    def setup_class(cls):
        cls.phi0 = 0.5
        cls.k = 30.0
        cls.params = HarmonicImproperParameters(
            labels=("A",),
            phi0=jnp.full((1, 1, 1, 1), cls.phi0),
            k=jnp.full((1, 1, 1, 1), cls.k),
        )

    def _energy(self, positions):
        graph = _dihedral_graph(positions)
        return jax.jit(harmonic_improper_energy)(
            GraphPotentialInput(graph=graph, parameters=self.params)
        ).data.data[0]

    def test_zero_at_equilibrium(self):
        assert jnp.isclose(
            self._energy(_positions_for_dihedral(self.phi0)), 0.0, atol=1e-8
        )

    def test_formula(self):
        phi = 1.3
        expected = self.k * (phi - self.phi0) ** 2
        assert jnp.isclose(
            self._energy(_positions_for_dihedral(phi)), expected, rtol=1e-5
        )

    def test_gradient_zero_at_equilibrium(self):
        grad = jax.jit(jax.grad(self._energy))(_positions_for_dihedral(self.phi0))
        assert jnp.allclose(grad, 0.0, atol=1e-6)


class TestShiftedPeriodicHarmonicImproper:
    """Test the shifted-periodic recentering: where(phi<0, phi+2pi, phi) - pi."""

    @staticmethod
    def _shifted(phi):
        return jnp.where(phi < 0.0, phi + 2.0 * jnp.pi, phi) - jnp.pi

    def _energy(self, positions, phi0, shifted_periodic):
        params = HarmonicImproperParameters(
            labels=("A",),
            phi0=jnp.full((1, 1, 1, 1), phi0),
            k=jnp.full((1, 1, 1, 1), 30.0),
            shifted_periodic=shifted_periodic,
        )
        graph = _dihedral_graph(positions)
        return jax.jit(harmonic_improper_energy)(
            GraphPotentialInput(graph=graph, parameters=params)
        ).data.data[0]

    def test_zero_at_shifted_equilibrium(self):
        phi = -2.0
        phi0 = float(self._shifted(jnp.array(phi)))
        assert jnp.isclose(
            self._energy(_positions_for_dihedral(phi), phi0, True), 0.0, atol=1e-6
        )

    def test_plain_nonzero_at_same_point(self):
        # With the same phi0 but no recentering, the raw torsion (~-2.0) != phi0,
        # so the energy is clearly nonzero.
        phi = -2.0
        phi0 = float(self._shifted(jnp.array(phi)))
        assert self._energy(_positions_for_dihedral(phi), phi0, False) > 1.0
