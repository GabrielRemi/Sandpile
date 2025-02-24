import unittest

import numpy as np

from sandpile import *


class AvalancheTestCase(unittest.TestCase):
    def test_not_avalanches(self):
        system = SandpileND(dimension=1, critical_slope=2, linear_grid_size=5)

        start_cfg = np.array([1, 1, 1, 1, 2])
        a = check_create_avalanche(system, start_cfg)
        self.assertIsNone(a)

        start_cfg = np.array([1, 1, 1, 4, 4])
        self.assertRaises(Exception, check_create_avalanche, system, start_cfg)

    def test_avalanches_1d(self):
        system = SandpileND(dimension=1, critical_slope=1, linear_grid_size=5)

        cfg = np.array([0, 0, 0, 2, 0])
        a = check_create_avalanche(system, cfg)
        self.assertIsNotNone(a)

        self.assertTrue(np.asarray(cfg == np.array([0, 0, 1, 0, 1])).all())
        self.assertEqual(a.reach, 0)
        self.assertEqual(a.time, 1)
        self.assertEqual(a.size, 1)

        cfg = np.array([0, 0, 1, 2, 0])
        a = check_create_avalanche(system, cfg)
        self.assertIsNotNone(a)

        self.assertTrue(np.asarray(cfg == np.array([0, 1, 0, 1, 1])).all())
        self.assertEqual(a.reach, 1)
        self.assertEqual(a.time, 2)
        self.assertEqual(a.size, 2)


class SystemTestCase(unittest.TestCase):
    def test_conservative_perturbation_1d(self):
        system = SandpileND(dimension=1, critical_slope=1, linear_grid_size=5)

        values = [
            ([0, 0, 0, 0, 0], 1, [-1, 1, 0, 0, 0]),
            ([0, 0, 0, 0, 0], 0, [1, 0, 0, 0, 0]),
            ([1, 1, 1, 3, 2], 2, [1, 0, 2, 3, 2])
        ]

        for setup in values:
            cfg = np.array(setup[0])
            system._conservative_perturbation(cfg, [setup[1]])
            self.assertTrue(np.asarray(cfg == np.array(setup[2])).all(), f"at array {setup[0]}")

if __name__ == '__main__':
    unittest.main()
