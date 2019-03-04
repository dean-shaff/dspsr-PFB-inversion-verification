import os
import unittest

# import matplotlib.pyplot as plt
import numpy as np
from pfb.util import load_matlab_filter_coef

from src.design_pfb_fir import design_pfb_fir

current_dir = os.path.dirname(os.path.abspath(__file__))
matlab_fir_paths = {
    file_name.split(".")[1]: os.path.join(current_dir, file_name) for
    file_name in ["Prototype_FIR.40.mat", "Prototype_FIR.480.mat"]
}


class TestDesignPFBFIR(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.matlab_fir_filter_coef = {
            t: load_matlab_filter_coef(matlab_fir_paths[t])[-1]
            for t in matlab_fir_paths
        }

    def test_design_pfb_fir(self):
        nchan = 8
        os = "8/7"
        # fig, axes = plt.subplots(2, 1)
        for i, ntaps in enumerate(self.matlab_fir_filter_coef):
            coeff = design_pfb_fir(int(ntaps) + 1, nchan, os)
            matlab_coeff = self.matlab_fir_filter_coef[ntaps]
            allclose = np.allclose(coeff, matlab_coeff, atol=1e-6)
            self.assertTrue(allclose)
        #     print(ntaps, allclose)
        #     axes[i].grid(True)
        #     axes[i].plot(coeff)
        #     axes[i].plot(matlab_coeff)
        # plt.show()


if __name__ == '__main__':
    unittest.main()
