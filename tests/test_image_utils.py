from src import image_utils
import numpy as np

""" Varon Ratio Tests """
S1 = np.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.2, 0.3, 0.4]])
B1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
c = 10
calculated_mean1 = 1.5
calculated_stddev1 = 0.91287092917528

mean, std_deviation = image_utils.varonRatio(S1, B1, c)
assert mean == calculated_mean1
assert np.isclose(std_deviation, calculated_stddev1, rtol=1e-5)


S2 = np.array([[0.1, 0.09], [0.15, 0.25], [0.2, 0.1]])
B2 = np.array([[2, 3], [0.5, 0.25], [5, 10]])
c2 = 100
calculated_mean2 = 22.833333333333
calculated_stddev2 = 35.456154457145

mean, std_deviation = image_utils.varonRatio(S2, B2, c2)
assert np.isclose(mean, calculated_mean2, rtol=1e-5)
assert np.isclose(std_deviation, calculated_stddev2, rtol=1e-5)