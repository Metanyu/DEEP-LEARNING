import numpy as np
import torch
from skimage import color
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from PIL import Image


def lab_to_rgb_numpy(L, ab):
    if isinstance(L, torch.Tensor):
        L = L.cpu().numpy()
        ab = ab.cpu().numpy()
    
    if L.ndim == 3:
        L = L.squeeze(0)
    
    L = (L + 1.0) * 50.0
    ab = ab * 128.0
    Lab = np.stack([L, ab[0], ab[1]], axis=2)
    rgb = color.lab2rgb(Lab)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def calculate_colorfulness(rgb):
    R, G, B = rgb[:,:,0].astype(float), rgb[:,:,1].astype(float), rgb[:,:,2].astype(float)
    
    rg = R - G
    yb = 0.5 * (R + G) - B
    
    sigma_rg = np.std(rg)
    sigma_yb = np.std(yb)
    mu_rg = np.mean(rg)
    mu_yb = np.mean(yb)
    
    return np.sqrt(sigma_rg**2 + sigma_yb**2) + 0.3 * np.sqrt(mu_rg**2 + mu_yb**2)


def load_and_preprocess_image(image_path, img_size=256):
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    img = img.resize((img_size, img_size), Image.LANCZOS)
    img_np = np.array(img)
    lab = color.rgb2lab(img_np).astype(np.float32)
    L = lab[:, :, 0:1] / 50.0 - 1.0
    L_tensor = torch.from_numpy(L.transpose(2, 0, 1)).unsqueeze(0)
    return L_tensor, original_size, lab


def lab_to_rgb_for_save(L, ab, original_size=None):
    if L.dim() == 4:
        L = L[0]
    if ab.dim() == 4:
        ab = ab[0]
    
    rgb_np = lab_to_rgb_numpy(L, ab)
    rgb_img = Image.fromarray(rgb_np)
    
    if original_size is not None:
        rgb_img = rgb_img.resize(original_size, Image.LANCZOS)
    return rgb_img


def evaluate_single_image(rgb_pred, rgb_true):
    psnr = compute_psnr(rgb_true, rgb_pred, data_range=255)
    ssim = compute_ssim(rgb_true, rgb_pred, channel_axis=2, data_range=255)
    
    pred_color = calculate_colorfulness(rgb_pred)
    true_color = calculate_colorfulness(rgb_true)
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'colorfulness_pred': pred_color,
        'colorfulness_true': true_color,
        'colorfulness_ratio': pred_color / (true_color + 1e-8)
    }
