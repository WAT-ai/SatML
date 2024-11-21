import unittest
import numpy as np
from pathlib import Path

from src import image_utils

class TestImageUtils(unittest.TestCase):
    def setUp(self):
        self.files_to_remove = []
        return super().setUp()

    def tearDown(self):
        for file_to_remove in self.files_to_remove:
            Path(file_to_remove).unlink()

        return super().tearDown()

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
    
    
    """
    Here's the link to the google sheet used to create the correct_matrix:
    https://docs.google.com/spreadsheets/d/1ibQIVitjaxNGXof7cjM9m5XKFG8H756YdaG6zqICUpI/edit?usp=sharing
    """
    def test_varon_iteration_easy(self):
        compute_matrix = image_utils.varon_iteration("data/raw_data/STARCOP_train_easy", "tests/varon.npy", 2, 3, 3, 5)
        self.files_to_remove.append('tests/varon.npy') 
        
        image_utils.createTestMatrix()
        correct_matrix = np.load("tests/varon_correct.npy")
        self.files_to_remove.append('tests/varon_correct.npy') 

        np.testing.assert_almost_equal(correct_matrix, compute_matrix, decimal=6) 
    

