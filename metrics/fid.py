import numpy as np
import torch.nn.functional as F
from scipy import linalg
import torch.nn as nn
import torch
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm


class INCEPTION_V3_FID(nn.Module):
    def __init__(self):
        super().__init__()
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

        for param in inception.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(
            *[c for c in inception.children() if c.__class__.__name__ not in ['InceptionAux', 'Linear']]
        )

    def forward(self, x):
        """x: (B, 3, H, W) [0, 1]"""
        x = F.interpolate(x, size=(299, 299), mode='bilinear')

        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225

        x = self.features(x)
        x = torch.flatten(x, start_dim=1)

        return x


def get_activations(images, model, batch_size=16, device='cpu'):
    """
    Compute pool_3 activations for images in batches.
    
    Parameters
    ----------
    images : torch.Tensor
        Shape (N, 3, H, W), values in [0,1]
    model : INCEPTION_V3_FID instance
    batch_size : int
        Number of images per batch
    device : str
        'cpu' or 'cuda'
    
    Returns
    -------
    act : np.ndarray
        Shape (N, 2048) with pool_3 features
    """
    model.eval()
    model.to(device)
    n_images = images.size(0)
    activations = []

    loop = tqdm(range(0, n_images, batch_size), leave=True)
    with torch.no_grad():
        for start in loop:
            end = start + batch_size
            batch = images[start:end].to(device)
            feat = model(batch)  # shape (B, 2048)
            activations.append(feat.cpu())

    activations = torch.cat(activations, dim=0)
    return activations.numpy()


def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- act      : Numpy array of dimension (n_images, dim (e.g. 2048)).
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance (FID) between two Gaussians.
    The FID between two multivariate Gaussians X_1 ~ N(mu1, sigma1) and
    X_2 ~ N(mu2, sigma2) is
        d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    This version is stable for nearly singular covariance matrices.

    Parameters
    ----------
    mu1 : np.ndarray
        Mean of activations for generated samples (shape: [dim])
    sigma1 : np.ndarray
        Covariance of activations for generated samples (shape: [dim, dim])
    mu2 : np.ndarray
        Mean of activations for real samples (shape: [dim])
    sigma2 : np.ndarray
        Covariance of activations for real samples (shape: [dim, dim])
    eps : float
        Small value added to diagonal for numerical stability

    Returns
    -------
    float
        The Frechet Distance between the two distributions
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors must have the same length"
    assert sigma1.shape == sigma2.shape, "Covariance matrices must have the same dimensions"

    diff = mu1 - mu2

    # Add small value to diagonal to improve stability
    covmean = linalg.sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])).dot(
                            sigma2 + eps * np.eye(sigma2.shape[0])))

    # If sqrtm produces complex numbers due to numerical errors, take real part
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            max_imag = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component in sqrtm: {max_imag}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)


def calculate_fid(images_real, images_fake, batch_size=16, device='cpu'):
    model = INCEPTION_V3_FID()

    act_real = get_activations(images_real, model, batch_size, device)
    act_fake = get_activations(images_fake, model, batch_size, device)

    mu_real, sigma_real = calculate_activation_statistics(act_real)
    mu_fake, sigma_fake = calculate_activation_statistics(act_fake)

    fid_score = calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
    return fid_score


if __name__ == '__main__':
    # example usage
    images_real = torch.rand(100, 3, 256, 256)  # real images
    images_fake = torch.rand(100, 3, 256, 256)  # generated images

    fid_score = calculate_fid(images_real, images_fake, batch_size=12, device='cuda')
    print("FID:", fid_score)