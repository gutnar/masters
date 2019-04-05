import numpy as np


class PDFSampler:
    def __init__(self, grid, pdf, add_uniform=False):
        self.cdf = np.cumsum(pdf)
        self.cdf = self.cdf / self.cdf[-1]
        self.grid = grid
    
    def __call__(self, N=1):
        choices = np.random.rand(N)
        indices = np.searchsorted(self.cdf, choices)

        return self.grid[indices]
