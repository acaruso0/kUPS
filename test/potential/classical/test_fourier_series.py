# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Fourier-series dihedral prior."""

import jax
import jax.numpy as jnp

from kups.core.data.index import Index
from kups.core.neighborlist import Edges
from kups.potential.classical.fourier_series import (
    FourierDihedralParameters,
    fourier_dihedral_energy,
)
from kups.potential.common.graph import GraphPotentialInput, HyperGraph

from .conftest import make_particles, make_systems

_CELLS_LV = jnp.eye(3)[None] * 20.0


def _positions_for_dihedral(phi) -> jax.Array:
    """Four atoms with torsion angle ``phi``."""
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


def _dihedral_graph(positions):
    particles = make_particles(positions, ["A", "A", "A", "A"], [0, 0, 0, 0])
    systems = make_systems(_CELLS_LV)
    edges = Edges(
        indices=Index(particles.keys, jnp.array([[0, 1, 2, 3]])),
        shifts=jnp.zeros((1, 3, 3)),
    )
    return HyperGraph(particles, systems, edges)


class TestFourierDihedralEnergy:
    """V(phi) = v0 + sum_n [k1_n sin(n phi) + k2_n cos(n phi)]."""

    @classmethod
    def setup_class(cls):
        # N = 3 terms.
        cls.k1s = jnp.array([0.5, -0.2, 0.1]).reshape(3, 1, 1, 1, 1)
        cls.k2s = jnp.array([1.0, 0.3, -0.4]).reshape(3, 1, 1, 1, 1)
        cls.v0 = jnp.array(0.7).reshape(1, 1, 1, 1)
        cls.params = FourierDihedralParameters(
            labels=("A",), k1s=cls.k1s, k2s=cls.k2s, v0=cls.v0
        )

    def _energy(self, positions):
        graph = _dihedral_graph(positions)
        return jax.jit(fourier_dihedral_energy)(
            GraphPotentialInput(graph=graph, parameters=self.params)
        ).data.data[0]

    def _expected(self, phi):
        k1 = self.k1s.reshape(3)
        k2 = self.k2s.reshape(3)
        val = self.v0.reshape(())
        for n in range(1, 4):
            val = val + k1[n - 1] * jnp.sin(n * phi) + k2[n - 1] * jnp.cos(n * phi)
        return val

    def test_formula(self):
        for phi in (-2.0, -0.5, 0.3, 1.7, 2.9):
            phi = jnp.array(phi)
            assert jnp.isclose(
                self._energy(_positions_for_dihedral(phi)),
                self._expected(phi),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_sign_sensitivity(self):
        # The sine terms make the energy sensitive to the sign of phi:
        # E(phi) != E(-phi) unless all k1 vanish.
        phi = jnp.array(1.1)
        e_pos = self._energy(_positions_for_dihedral(phi))
        e_neg = self._energy(_positions_for_dihedral(-phi))
        assert not jnp.isclose(e_pos, e_neg, rtol=1e-4)

    def test_gradient_finite(self):
        grad = jax.jit(jax.grad(self._energy))(_positions_for_dihedral(jnp.array(0.8)))
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.linalg.norm(grad) > 1e-6

    def test_degree_inferred(self):
        # A single-term (N=1) series should match the explicit one-term formula.
        params = FourierDihedralParameters(
            labels=("A",),
            k1s=jnp.array(0.5).reshape(1, 1, 1, 1, 1),
            k2s=jnp.array(1.0).reshape(1, 1, 1, 1, 1),
            v0=jnp.zeros((1, 1, 1, 1)),
        )
        phi = jnp.array(0.9)
        expected = 0.5 * jnp.sin(phi) + 1.0 * jnp.cos(phi)
        graph = _dihedral_graph(_positions_for_dihedral(phi))
        got = fourier_dihedral_energy(
            GraphPotentialInput(graph=graph, parameters=params)
        ).data.data[0]
        assert jnp.isclose(got, expected, rtol=1e-5, atol=1e-6)
