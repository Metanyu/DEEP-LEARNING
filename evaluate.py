#!/usr/bin/env python3

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import color

from ab_gamut import ABGamut
from models import ColorizationModel, ClassificationToAB
from metrics.colorfulness import calculate_colorfulness as calc_colorfulness_tensor
from metrics.fid import calculate_fid
from metrics.psnr_ssim import calculate_psnr_ssim


class ColorizationEvaluator:
    
    def __init__(self, model_path, device='cuda', temperature=0.38, model_type='classification'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.ab_gamut = ABGamut(grid_size=10, sigma=5.0)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = ColorizationModel(
            output_type=model_type,
            num_classes=self.ab_gamut.Q,
            dropout=0.0
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        if model_type == 'classification':
            self.cls_to_ab = ClassificationToAB(self.ab_gamut, temperature=temperature).to(self.device)
        else:
            self.cls_to_ab = None
    
    @torch.no_grad()
    def colorize_and_collect(self, image_paths, img_size=256, batch_size=8):
        real_images = []
        fake_images = []
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_size, img_size), Image.LANCZOS)
            img_np = np.array(img) / 255.0
            
            lab = color.rgb2lab(img_np).astype(np.float32)
            L = lab[:, :, 0:1] / 50.0 - 1.0
            L_tensor = torch.from_numpy(L.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            
            output = self.model(L_tensor)
            if self.model_type == 'classification':
                ab = self.cls_to_ab(output)
            else:
                ab = output
            
            ab = F.interpolate(ab, size=(img_size, img_size), mode='bilinear', align_corners=False)
            
            L_denorm = (L_tensor[0, 0].cpu().numpy() + 1.0) * 50.0
            ab_denorm = ab[0].cpu().numpy() * 128.0
            
            Lab = np.stack([L_denorm, ab_denorm[0], ab_denorm[1]], axis=2)
            rgb_fake = color.lab2rgb(Lab)
            rgb_fake = np.clip(rgb_fake, 0, 1)
            
            real_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
            fake_tensor = torch.from_numpy(rgb_fake.transpose(2, 0, 1)).float()
            
            real_images.append(real_tensor)
            fake_images.append(fake_tensor)
        
        real_images = torch.stack(real_images)
        fake_images = torch.stack(fake_images)
        
        return real_images, fake_images
    
    def evaluate(self, image_dir, img_size=256, batch_size=16, max_images=None):
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = []
        for f in sorted(os.listdir(image_dir)):
            if f.lower().endswith(extensions):
                image_paths.append(os.path.join(image_dir, f))
        
        if max_images is not None:
            image_paths = image_paths[:max_images]
        
        print(f"\nFound {len(image_paths)} images in {image_dir}")
        
        real_images, fake_images = self.colorize_and_collect(image_paths, img_size, batch_size)
        
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        print("\n[1/4] Calculating Colorfulness...")
        colorfulness_real = calc_colorfulness_tensor(real_images, device=self.device)
        colorfulness_fake = calc_colorfulness_tensor(fake_images, device=self.device)
        print(f"  Real images colorfulness: {colorfulness_real:.2f}")
        print(f"  Generated images colorfulness: {colorfulness_fake:.2f}")
        
        print("\n[2/4] Calculating FID...")
        fid_score = calculate_fid(real_images, fake_images, batch_size=batch_size, device=self.device)
        print(f"  FID Score: {fid_score:.2f}")
        
        print("\n[3/4] Calculating PSNR...")
        psnr, ssim_val, ms_ssim_val = calculate_psnr_ssim(
            real_images, fake_images, batch_size=batch_size, device=self.device
        )
        print(f"  PSNR: {psnr:.2f} dB")
        
        print("\n[4/4] Calculating SSIM & MS-SSIM...")
        print(f"  SSIM: {ssim_val:.4f}")
        print(f"  MS-SSIM: {ms_ssim_val:.4f}")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Model: {self.model_type}")
        print(f"Images evaluated: {len(image_paths)}")
        print(f"\nColorfulness (Real): {colorfulness_real:.2f}")
        print(f"Colorfulness (Generated): {colorfulness_fake:.2f}")
        print(f"\nFID: {fid_score:.2f}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim_val:.4f}")
        print(f"MS-SSIM: {ms_ssim_val:.4f}")
        
        return {
            'colorfulness_real': colorfulness_real,
            'colorfulness_fake': colorfulness_fake,
            'fid': fid_score,
            'psnr': psnr,
            'ssim': ssim_val,
            'ms_ssim': ms_ssim_val
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate colorization model using multiple metrics')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Directory containing images to evaluate')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Model type: classification or regression')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--temperature', type=float, default=0.38,
                       help='Temperature for annealed-mean (classification only)')
    parser.add_argument('--size', type=int, default=256,
                       help='Processing image size')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for metric computation')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to evaluate (default: all)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        parser.error(f'Model file not found: {args.model}')
    
    if not os.path.isdir(args.image_dir):
        parser.error(f'Image directory not found: {args.image_dir}')
    
    evaluator = ColorizationEvaluator(
        model_path=args.model,
        device=args.device,
        temperature=args.temperature,
        model_type=args.model_type
    )
    
    results = evaluator.evaluate(
        image_dir=args.image_dir,
        img_size=args.size,
        batch_size=args.batch_size,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()
