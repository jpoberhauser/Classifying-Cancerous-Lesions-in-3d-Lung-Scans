# run w/ pytest
import pytest
import unittest
import numpy as np
from preprocessing import crop

class PreprocessingTestCase(unittest.TestCase):
    def setup_method(self, method):
        pass

    def test_case_one(self):
        # create three arrays each
        # three dimensions...
        arrays = []

        arrays.append(np.random.rand(2, 2, 2))
        arrays.append(np.random.rand(3, 3, 3))
        arrays.append(np.random.rand(4, 4, 4))

        arrays = crop(arrays)

        for array in arrays:
            assert array.shape == (3, 3, 3)
