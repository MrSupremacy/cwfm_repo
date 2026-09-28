#!/usr/bin/env python
"""Run the production tiny-model fullFT and routing tests."""
import unittest


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromNames([
        "tests.unit.test_routers",
        "tests.integration.test_fullft_tiny_t5",
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
