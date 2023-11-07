#  PINTS: Peak Identifier for Nascent Transcripts Sequencing
#  Copyright (c) 2019-2023 Yu Lab.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
import unittest
import numpy as np
from pints.stats_engine import ZIP


class ZeroInflatedPoissonTestCase(unittest.TestCase):
    def test_em(self):
        np.random.seed(299)
        n = 1000
        mu = 2.5  # Poisson rate
        pi = 0.55  # probability of extra-zeros (pi = 1-psi)

        # Simulate some data
        z = ZIP()
        counts = np.array([(np.random.random() > pi) *
                           np.random.poisson(mu) for _ in range(n)])
        mu_hat, _, pi_hat, likelihood, convergence = z.fit(counts)
        # Check if the alg converges
        self.assertTrue(convergence)
        # Check if the estimated mu is close enough to the true value
        self.assertAlmostEqual(mu_hat, mu, delta=0.05)
        # Check if the estimated pi is close enough to the true value
        self.assertAlmostEqual(pi_hat, pi, delta=0.05)
