import unittest

from src import image_utils
import numpy as np

class TestImageUtils(unittest.TestCase):

    def test_basicTest(self):
        S1 = np.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.2, 0.3, 0.4]])
        B1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        c = 10
        calculated_mean1 = 1.5
        calculated_stddev1 = 0.91287092917528

        mean, std_deviation = image_utils.varonRatio(S1, B1, c)
        self.assertTrue(mean == calculated_mean1)
        np.testing.assert_allclose(std_deviation, calculated_stddev1, rtol=1e-5)

    def test_differentDenom(self):
        S2 = np.array([[0.1, 0.09], [0.15, 0.25], [0.2, 0.1]])
        B2 = np.array([[2, 3], [0.5, 0.25], [5, 10]])
        c2 = 100
        calculated_mean2 = 22.833333333333
        calculated_stddev2 = 35.456154457145

        mean, std_deviation = image_utils.varonRatio(S2, B2, c2)
        np.testing.assert_allclose(mean, calculated_mean2, rtol=1e-5)
        np.testing.assert_allclose(std_deviation, calculated_stddev2, rtol=1e-5)
    
    def test_varon_iteration_easy(self):
        mean, std_deviation = image_utils.varon_iteration("data/raw_data/STARCOP_train_easy", "tests/varon.npy", 2, 1)
        print(mean)
        print(std_deviation)
        data = np.load("tests/varon.npy")
        print(data) 
