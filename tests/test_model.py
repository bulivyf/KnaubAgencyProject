"""Unit tests for graph construction, conservation, interventions, and convergence."""
import unittest
from dataclasses import replace

import numpy as np
from model import Config, topology, run


class TestGraphDynamics(unittest.TestCase):
    """Verify mathematical properties of the synthetic network model."""

    def test_topology(self):
        # All supported agent counts must produce symmetric nonnegative W.
        for n in (3, 4, 5):
            W, Q = topology(n)
            self.assertTrue(np.allclose(W, W.T))
            self.assertTrue(np.all(W >= 0) and np.all(Q >= 0))

    def test_pure_diffusion_conserves_total(self):
        # With only symmetric diffusion, sum of node states stays constant.
        cfg = Config(n=5, diffusion=1, advection=0, reaction=0,
                     epsilon=0.35, dt=0.01, duration=6)
        _, states, *_ = run(cfg, False)
        self.assertLess(np.max(np.abs(
            states.sum(axis=1) - states[0].sum())), 1e-10)

        # Diffusion should narrow the stance range for this test case.
        self.assertLess(np.ptp(states[-1]), np.ptp(states[0]))

    def test_directed_transport_conserves_total(self):
        # Incoming-minus-outgoing transport conserves the sum of states.
        cfg = Config(n=4, diffusion=0, advection=0.15, reaction=0,
                     dt=0.01, duration=2)
        _, states, *_ = run(cfg, False)
        self.assertLess(np.max(np.abs(
            states.sum(axis=1) - states[0].sum())), 1e-10)

    def test_evidence_has_effect(self):
        # Changing only the evidence switch must measurably change outputs.
        cfg = Config(duration=12)
        _, baseline, *_ = run(cfg, False)
        _, evidence, *_ = run(cfg, True)
        self.assertGreater(
            np.linalg.norm(baseline[-1] - evidence[-1]), 0.01)

    def test_dt_convergence(self):
        # A reasonably refined Euler timestep should approach a finer run.
        cfg = Config(duration=10, dt=0.08)
        final_states = []
        for dt in (0.08, 0.04, 0.02, 0.01):
            _, u, *_ = run(replace(cfg, dt=dt), True)
            final_states.append(u[-1])
        coarse = np.linalg.norm(final_states[0] - final_states[-1])
        finer = np.linalg.norm(final_states[2] - final_states[-1])
        self.assertLess(finer, coarse)


if __name__ == "__main__":
    unittest.main()
