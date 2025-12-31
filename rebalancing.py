import numpy as np
from tqdm import tqdm


class ColorRebalancer:
    
    def __init__(self, ab_gamut, lambda_mix=0.5, sigma=5.0):
        self.ab_gamut = ab_gamut
        self.lambda_mix = lambda_mix
        self.sigma = sigma
        self.Q = ab_gamut.Q
        self.weights = None
        self.empirical_dist = None
    
    def compute_empirical_distribution(self, dataloader, max_samples=10000):
        bin_counts = np.zeros(self.Q, dtype=np.float64)
        total_pixels = 0
        sample_count = 0
        
        for L, ab in tqdm(dataloader, desc="Sampling colors"):
            ab_np = ab.permute(0, 2, 3, 1).numpy()
            ab_np = ab_np * 128.0
            for i in range(ab_np.shape[0]):
                bin_indices = self.ab_gamut.encode_ab_to_hard_bins(ab_np[i])
                unique, counts = np.unique(bin_indices, return_counts=True)
                bin_counts[unique] += counts
                total_pixels += bin_indices.size
            
            sample_count += ab_np.shape[0]
            if sample_count >= max_samples:
                break
        
        self.empirical_dist = bin_counts / total_pixels
        self._compute_weights()
        
        return self.empirical_dist
    
    def _compute_weights(self):
        uniform = np.ones(self.Q) / self.Q
        mixed_dist = (1 - self.lambda_mix) * self.empirical_dist + self.lambda_mix * uniform
        self.weights = 1.0 / (mixed_dist + 1e-8)
        expected_weight = np.sum(self.empirical_dist * self.weights)
        self.weights = self.weights / expected_weight
    
    def use_prior_weights(self):
        ab_centers = self.ab_gamut.ab_gamut
        distances = np.sqrt(np.sum(ab_centers ** 2, axis=1))
        self.empirical_dist = np.exp(-distances / 30.0)
        self.empirical_dist = self.empirical_dist / self.empirical_dist.sum()
        self._compute_weights()
    
    def save_weights(self, path):
        np.savez(path, weights=self.weights, empirical_dist=self.empirical_dist)
    
    def load_weights(self, path):
        data = np.load(path)
        self.weights = data['weights']
        self.empirical_dist = data['empirical_dist']
