import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
from tqdm import tqdm

def calculate_psnr_batch(img1, img2, max_val=1.0):
    """Compute PSNR for a batch of images."""
    mse = F.mse_loss(img1, img2, reduction='none')  # shape: (B, C, H, W)
    mse = mse.view(mse.size(0), -1).mean(dim=1)     # mean per image
    psnr_batch = 20 * torch.log10(torch.tensor(max_val, device=img1.device)) - 10 * torch.log10(mse)
    return psnr_batch  # returns tensor of shape (B,)

def calculate_ssim_batch(img1, img2):
    """Compute SSIM per batch"""
    return ssim(img1, img2, data_range=1.0, size_average=False)  # returns tensor of shape (B,)

def calculate_ms_ssim_batch(img1, img2):
    """Compute MS-SSIM per batch"""
    return ms_ssim(img1, img2, data_range=1.0, size_average=False)  # returns tensor of shape (B,)

def calculate_psnr_ssim(real_images, fake_images, batch_size=16, device='cuda'):
    psnr_list, ssim_list, ms_ssim_list = [], [], []
    
    n = real_images.shape[0]
    loop = tqdm(range(0, n, batch_size), leave=True)
    for i in loop:
        real_batch = real_images[i:i+batch_size].to(device)
        fake_batch = fake_images[i:i+batch_size].to(device)
        
        psnr_batch = calculate_psnr_batch(real_batch, fake_batch)
        ssim_batch = calculate_ssim_batch(real_batch, fake_batch)
        ms_ssim_batch = calculate_ms_ssim_batch(real_batch, fake_batch)
        
        psnr_list.append(psnr_batch)
        ssim_list.append(ssim_batch)
        ms_ssim_list.append(ms_ssim_batch)
    
    # Concatenate all batches and take mean
    psnr_all = torch.cat(psnr_list).mean().item()
    ssim_all = torch.cat(ssim_list).mean().item()
    ms_ssim_all = torch.cat(ms_ssim_list).mean().item()
    
    return psnr_all, ssim_all, ms_ssim_all

# Example usage:
if __name__ == '__main__':
    real_images = torch.rand(10000, 3, 256, 256)  # values in [0,1]
    fake_images = torch.rand(10000, 3, 256, 256)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    psnr, ssim_val, ms_ssim_val = calculate_psnr_ssim(real_images, fake_images, batch_size=16, device=device)
    
    print("PSNR:", psnr)
    print("SSIM:", ssim_val)
    print("MS-SSIM:", ms_ssim_val)