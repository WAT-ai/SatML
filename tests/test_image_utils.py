import unittest
import numpy as np
from pathlib import Path

from src import image_utils

def get_expected_varon_matrix():
    data = np.array([
        [[0, -0.2104005415, -0.4138471552],
        [-0.2104005415, 0, -0.2586695576],
        [-0.4138471552, -0.2586695576, 0]],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
        [[0, -0.2246186874, -0.3063050874],
        [-0.2246186874, 0, -0.1064981483],
        [-0.3063050874, -0.1064981483, 0]]
    ])

    return data

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
        images = ["ang20190927t184620_r7541_c401_w151_h151",
                  "ang20190927t153023_r7101_c126_w151_h151",
                  "ang20191019t175004_r8192_c256_w512_h512"]
        compute_matrix = image_utils.varon_iteration("data/raw_data/STARCOP_train_easy", "tests/varon.npy", 2, 3, images,5)
        self.files_to_remove.append('tests/varon.npy') 
        
        image_utils.createTestMatrix()
        correct_matrix = np.load("tests/varon_correct.npy")
        self.files_to_remove.append('tests/varon_correct.npy') 
        np.testing.assert_almost_equal(correct_matrix, compute_matrix, decimal=6) 
    
    
    def test_binary_bbox(self):
        binary_image = np.array([
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0]
        ]) 
        expected_bboxes = np.array(
            [
                (1, 3, 0, 1),
                (4, 4, 2, 3),
                (0, 1, 3, 4)
            ]
        )
        np.testing.assert_array_equal(image_utils.binary_bbox(binary_image), expected_bboxes)
        # print(np.array2string(binary_bbox(binary_image), separator=', '))
        # useful output formatter if we want to write more tests.
        

class TestIOUFunctions(unittest.TestCase):
    
    def test_iou(self):
        # Test perfect overlap (loss = 0)
        result = image_utils.compare_bbox([[1,1,3,4]], [[1,1,3,4]], "iou")
        self.assertAlmostEqual(result, 0, places=4)

        # Test partial overlap 
        result = image_utils.compare_bbox([[1,1,3,4]], [[2,2,4,5]], "iou") 
        self.assertAlmostEqual(result, 0.8, places=4)

        # Test non-overlapping boxes (loss = 1)
        result = image_utils.compare_bbox([[1,1,3,4]], [[4,5,6,7]], "iou")
        self.assertEqual(result, 1)
        
    def test_giou(self):
        # Test perfect overlap (GIoU = 0)
        result = image_utils.compare_bbox([[1,1,3,4]], [[1,1,3,4]], "giou")
        self.assertAlmostEqual(result, 0, places=4)
        
        # Test GIoU with partial overlap
        result = image_utils.compare_bbox([[1,1,3,4]], [[2,2,4,5]], "giou")
        self.assertTrue(-1 <= result <= 1)
        
        # Test non-overlapping boxes (GIoU <= 0)
        result = image_utils.compare_bbox([[1,1,3,4]], [[4,5,6,7]], "giou")
        self.assertTrue(result >= 1 and result <= 2)
        
    def test_ciou(self):
        # Test perfect overlap (CIoU = 1)
        result = image_utils.compare_bbox([[1,1,3,4]], [[1,1,3,4]], "ciou")
        self.assertAlmostEqual(result, 0, places=4)
        
        # Test CIoU with partial overlap
        result = image_utils.compare_bbox([[1,1,3,4]], [[2,2,4,5]], "ciou")
        self.assertTrue(-1 <= result <= 1)

        # Test non-overlapping boxes (CIoU <= 0)
        result = image_utils.compare_bbox([[1,1,3,4]], [[4,5,6,7]], "ciou")
        self.assertTrue(result >= 1 and result <= 2)

class TestCompareBBox(unittest.TestCase):
    def test_compare_bbox_invalid_len(self):
        with self.assertRaises(ValueError):
            image_utils.compare_bbox([[1,1,3,4,4],[1, 1, 3, 4, 4]], [[4,5,6,7,2],[4,5,6,7,5]], "giou")

    def test_invalid_bbox_coord(self):
        with self.assertRaises(ValueError):
            image_utils.compare_bbox([[1, 1, 1, 4]], [[2,4,2,5]], "ciou")

    def test_invalid_metric(self):
        with self.assertRaises(ValueError):
            image_utils.compare_bbox([[1,1,3,4]], [[4,5,6,7]], "invalid_metric")
            
    def test_invalid_format(self):
        with self.assertRaises(ValueError):
            image_utils.compare_bbox(("in", 3, 1, 4), (4, "in", 5, 7), "iou")
