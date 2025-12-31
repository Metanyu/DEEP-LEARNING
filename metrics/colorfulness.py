import torch

def calculate_colorfulness(images, device='cpu'):
    # images: (B, 3, H, W)
    images = images.float().to(device)     # (B, 3, H, W)
    if images.max() <= 1.0:
        images = images * 255.0
    R, G, B = images[:,0], images[:,1], images[:,2]
    rg = torch.abs(R - G)
    yb = torch.abs(0.5 * (R+G) - B)
    
    rbMean = torch.mean(rg, dim=(1,2))
    rbStd = torch.std(rg, dim=(1,2))
    ybMean = torch.mean(yb, dim=(1,2))
    ybStd = torch.std(yb, dim=(1,2))
    
    stdRoot = torch.sqrt(rbStd**2 + ybStd**2)
    meanRoot = torch.sqrt(rbMean**2 + ybMean**2)
    
    colorfulness = stdRoot + 0.3 * meanRoot
    return colorfulness.mean().item()