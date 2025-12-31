#!/usr/bin/env python3

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from ab_gamut import ABGamut
from models import ColorizationModel, ClassificationToAB
from utils import load_and_preprocess_image, lab_to_rgb_for_save


class ColorizationInference:
    
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
    def colorize_image(self, image_path, img_size=256):
        L_tensor, original_size, lab = load_and_preprocess_image(image_path, img_size)
        L_tensor = L_tensor.to(self.device)
        output = self.model(L_tensor)
        if self.model_type == 'classification':
            ab = self.cls_to_ab(output)
        else:
            ab = output
        target_h, target_w = L_tensor.shape[2], L_tensor.shape[3]
        ab = F.interpolate(ab, size=(target_h, target_w), mode='bilinear', align_corners=False)
        ab_norm = ab / 128.0
        rgb_img = lab_to_rgb_for_save(L_tensor[0], ab_norm[0], original_size)
        return rgb_img
    
    def colorize_batch(self, image_paths, output_dir, img_size=256):
        os.makedirs(output_dir, exist_ok=True)
        for img_path in image_paths:
            try:
                colorized = self.colorize_image(img_path, img_size)
                output_path = os.path.join(output_dir, os.path.basename(img_path))
                colorized.save(output_path)
            except Exception as e:
                pass


def main():
    parser = argparse.ArgumentParser(description='Colorize grayscale images using trained model')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--images', type=str, nargs='+', help='Paths to multiple input images')
    parser.add_argument('--dir', type=str, help='Directory containing images to colorize')
    parser.add_argument('--model', type=str, default='model_full.pth', 
                       help='Path to model checkpoint (default: model_full.pth)')
    parser.add_argument('--model-type', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Model type: classification (default) or regression (l2_baseline)')
    parser.add_argument('--output', type=str, help='Output path for single image')
    parser.add_argument('--output-dir', type=str, default='colorized',
                       help='Output directory for batch processing (default: colorized/)')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--temperature', type=float, default=0.38,
                       help='Temperature for annealed-mean (default: 0.38)')
    parser.add_argument('--size', type=int, default=256,
                       help='Processing image size (default: 256)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.images and not args.dir:
        parser.error('Please provide --image, --images, or --dir')
    
    if not os.path.exists(args.model):
        parser.error(f'Model file not found: {args.model}')
    
    # Initialize inference
    inferencer = ColorizationInference(
        model_path=args.model,
        device=args.device,
        temperature=args.temperature,
        model_type=args.model_type
    )
    
    # Process single image
    if args.image:
        print(f"\nColorizing {args.image}...")
        colorized = inferencer.colorize_image(args.image, img_size=args.size)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            base, ext = os.path.splitext(args.image)
            output_path = f"{base}_colorized{ext}"
        
        colorized.save(output_path)
        print(f"✓ Saved to {output_path}")
    
    # Process multiple images
    elif args.images:
        print(f"\nColorizing {len(args.images)} images...")
        inferencer.colorize_batch(args.images, args.output_dir, img_size=args.size)
    
    # Process directory
    elif args.dir:
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = []
        for f in os.listdir(args.dir):
            if f.lower().endswith(extensions):
                image_paths.append(os.path.join(args.dir, f))
        
        print(f"\nFound {len(image_paths)} images in {args.dir}")
        inferencer.colorize_batch(image_paths, args.output_dir, img_size=args.size)


if __name__ == '__main__':
    main()
