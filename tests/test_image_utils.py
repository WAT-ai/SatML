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
        # Test perfect overlap (IOU = 1)
        result = image_utils.iou_metrics((1, 3, 1, 4), (1, 3, 1, 4), "iou")
        self.assertAlmostEqual(result, 1.0, places=4)

        # Test partial overlap
        result = image_utils.iou_metrics((1, 3, 1, 4), (2, 4, 2, 5), "iou") 
        self.assertAlmostEqual(result, 0.2, places=4)

        # Test non-overlapping boxes (IOU = 0)
        result = image_utils.iou_metrics((1, 3, 1, 4), (4, 6, 5, 7), "iou")
        self.assertEqual(result, 0)

    def test_diou(self):
        # Test perfect overlap (DIoU = 1)
        result = image_utils.iou_metrics((1, 3, 1, 4), (1, 3, 1, 4), "diou")
        self.assertAlmostEqual(result, 1.0, places=4)
        
        # Test DIoU with partial overlap
        result = image_utils.iou_metrics((1, 3, 1, 4), (2, 4, 2, 5), "diou")
        self.assertTrue(-1 <= result <= 1)

        # Test non-overlapping boxes (DIoU <= 0)
        result = image_utils.iou_metrics((1, 3, 1, 4), (4, 6, 5, 7), "diou")
        self.assertTrue(result <= 0)
        
    def test_giou(self):
        # Test perfect overlap (GIoU = 1)
        result = image_utils.iou_metrics((1, 3, 1, 4), (1, 3, 1, 4), "giou")
        self.assertAlmostEqual(result, 1.0, places=4)
        
        # Test GIoU with partial overlap
        result = image_utils.iou_metrics((1, 3, 4, 1), (2, 4, 5, 2), "giou")
        self.assertTrue(-1 <= result <= 1)
        
        # Test non-overlapping boxes (GIoU <= 0)
        result = image_utils.iou_metrics((1, 3, 1, 4), (4, 6, 5, 7), "giou")
        self.assertTrue(result <= 0)
        
    def test_ciou(self):
        # Test perfect overlap (CIoU = 1)
        result = image_utils.iou_metrics((1, 3, 1, 4), (1, 3, 1, 4), "ciou")
        self.assertAlmostEqual(result, 1.0, places=4)
        
        # Test CIoU with partial overlap
        result = image_utils.iou_metrics((1, 3, 1, 4), (2, 4, 2, 5), "ciou")
        self.assertTrue(-1 <= result <= 1)

        # Test non-overlapping boxes (CIoU <= 0)
        result = image_utils.iou_metrics((1, 3, 1, 4), (4, 6, 5, 7), "ciou")
        self.assertTrue(result <= 0)

class TestCompareBBox(unittest.TestCase):
    def test_compare_bbox_invalid_len(self):
        with self.assertRaises(AssertionError):
            image_utils.compare_bbox((1, 3, 1, 4, 4), (4, 6, 5, 7), "diou")

    def test_invalid_bbox_coord(self):
        with self.assertRaises(ValueError):
            image_utils.compare_bbox((1, 1, 1, 4), (2, 4, 2, 5), "ciou")

    def test_invalid_metric(self):
        with self.assertRaises(ValueError):
            image_utils.compare_bbox((1, 3, 1, 4), (4, 6, 5, 7), "invalid_metric")
            
    def test_invalid_format(self):
        with self.assertRaises(ValueError):
            image_utils.compare_bbox(("in", 3, 1, 4), (4, "in", 5, 7), "iou")
