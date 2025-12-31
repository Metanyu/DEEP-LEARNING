

import numpy as np
from skimage import color


class ABGamut:
    
    def __init__(self, grid_size=10, sigma=5.0):
        self.grid_size = grid_size
        self.sigma = sigma
        
        a_values = np.arange(-110, 120, grid_size)
        b_values = np.arange(-110, 120, grid_size)
        self.ab_grid = []
        for a in a_values:
            for b in b_values:
                self.ab_grid.append([a, b])
        self.ab_grid = np.array(self.ab_grid)
        
        self.in_gamut_mask = self._compute_in_gamut_mask()
        self.ab_gamut = self.ab_grid[self.in_gamut_mask]
        self.Q = len(self.ab_gamut)
    
    def _compute_in_gamut_mask(self):
        in_gamut = []
        L = 50.0
        
        for ab in self.ab_grid:
            lab = np.array([[[L, ab[0], ab[1]]]])
            rgb = color.lab2rgb(lab)
            valid = np.all(rgb >= -0.01) and np.all(rgb <= 1.01)
            in_gamut.append(valid)
        
        return np.array(in_gamut)
    
    def encode_ab_to_bins(self, ab):
        original_shape = ab.shape
        
        if len(original_shape) == 3:
            ab = ab.reshape(-1, 2)
        else:
            ab = ab.reshape(-1, 2)
        
        diff = ab[:, np.newaxis, :] - self.ab_gamut[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=2)
        weights = np.exp(-distances / (2 * self.sigma ** 2))
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
        
        if len(original_shape) == 3:
            H, W = original_shape[:2]
            return weights.reshape(H, W, self.Q)
        else:
            N, H, W = original_shape[:3]
            return weights.reshape(N, H, W, self.Q)
    
    def encode_ab_to_hard_bins(self, ab):
        original_shape = ab.shape[:-1]
        ab = ab.reshape(-1, 2)
        diff = ab[:, np.newaxis, :] - self.ab_gamut[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=2)
        indices = np.argmin(distances, axis=1)
        
        return indices.reshape(original_shape)
    
    def decode_bins_to_ab(self, probs, T=0.38):
        original_shape = probs.shape[:-1]
        probs = probs.reshape(-1, self.Q)
        log_probs = np.log(probs + 1e-8)
        annealed = np.exp(log_probs / T)
        annealed = annealed / (annealed.sum(axis=1, keepdims=True) + 1e-8)
        ab = np.dot(annealed, self.ab_gamut)
        
        return ab.reshape(*original_shape, 2)
