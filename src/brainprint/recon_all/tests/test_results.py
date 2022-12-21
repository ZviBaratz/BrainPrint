from unittest.case import TestCase

from brainprint.recon_all.results import ReconAllResults

METRICS = [
    "Average Thickness",
    "Folding Index",
    "Gray Matter Volume",
    "Integrated Rectified Gaussian Curvature",
    "Integrated Rectified Mean Curvature",
    "Intrinsic Curvature Index",
    "Surface Area",
    "Thickness StdDev",
]


class ReconAllResultsTestCase(TestCase):
    def setUp(self):
        self.default = ReconAllResults()

    def test_default_results_columns(self):
        value = len(self.default.raw_results.columns)
        expected = 2456
        self.assertEqual(value, expected)
