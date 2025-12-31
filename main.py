#!/usr/bin/env python3

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from config import *
from ab_gamut import ABGamut
from rebalancing import ColorRebalancer
from models import ColorizationModel, ClassificationToAB
from losses import MultinomialCrossEntropyLoss, L2LossWithRebalancing, PureMSELoss
from dataset import get_data_loaders
from utils import lab_to_rgb_numpy


class Trainer:
    
    def __init__(self, model, criterion, optimizer, scheduler=None,
                 device='cuda', model_type='regression', ab_gamut=None,
                 temperature=0.38, use_amp=True, verbose=True):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_type = model_type
        self.verbose = verbose
        
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler(device=device.type, enabled=self.use_amp)
        
        if model_type == 'classification' and ab_gamut is not None:
            self.cls_to_ab = ClassificationToAB(ab_gamut, temperature).to(device)
        else:
            self.cls_to_ab = None
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc="Training", disable=not self.verbose)
        for L, ab in pbar:
            L, ab = L.to(self.device), ab.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(L)
                loss = self.criterion(output, ab)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item() * L.size(0)
            if self.verbose:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return running_loss / len(dataloader.dataset)
    
    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        
        for L, ab in tqdm(dataloader, desc="Validating", disable=not self.verbose):
            L, ab = L.to(self.device), ab.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(L)
                loss = self.criterion(output, ab)
            
            running_loss += loss.item() * L.size(0)
        
        return running_loss / len(dataloader.dataset)
    
    def train(self, train_loader, val_loader, num_epochs, save_path=None, start_epoch=0):
        if self.verbose:
            print(f"Training for {num_epochs} epochs...")
            print(f"Model type: {self.model_type}")
        
        for epoch in range(num_epochs):
            if self.verbose:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if self.verbose:
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if save_path is not None:
                    torch.save({
                        'epoch': epoch + start_epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, save_path)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }


def create_experiment(exp_type, ab_gamut, rebalancer=None, device='cuda'):
    
    if exp_type == 'full':
        model = ColorizationModel(
            output_type='classification',
            num_classes=ab_gamut.Q,
            dropout=MODEL_CONFIG['dropout']
        )
        criterion = MultinomialCrossEntropyLoss(
            ab_gamut, rebalancer=rebalancer, device=device
        ).to(device)
        model_type = 'classification'
        
    elif exp_type == 'multinomial_only':
        model = ColorizationModel(
            output_type='classification',
            num_classes=ab_gamut.Q,
            dropout=MODEL_CONFIG['dropout']
        )
        criterion = MultinomialCrossEntropyLoss(
            ab_gamut, rebalancer=None, device=device
        ).to(device)
        model_type = 'classification'
        
    elif exp_type == 'l2_baseline':
        model = ColorizationModel(
            output_type='regression',
            num_classes=ab_gamut.Q,
            dropout=MODEL_CONFIG['dropout']
        )
        criterion = PureMSELoss()
        model_type = 'regression'
        
    elif exp_type == 'l2_rebalanced':
        model = ColorizationModel(
            output_type='regression',
            num_classes=ab_gamut.Q,
            dropout=MODEL_CONFIG['dropout']
        )
        criterion = L2LossWithRebalancing(
            ab_gamut, rebalancer=rebalancer, device=device
        ).to(device)
        model_type = 'regression'
    
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=TRAIN_CONFIG['learning_rate'], 
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_type=model_type,
        ab_gamut=ab_gamut,
        temperature=MODEL_CONFIG['temperature'],
        use_amp=TRAIN_CONFIG['use_amp'],
        verbose=TRAIN_CONFIG['verbose']
    )
    
    return trainer


def main():
    print("Image Colorization Training")
    print("="*60)
    
    ab_gamut = ABGamut(grid_size=10, sigma=REBALANCE_CONFIG['sigma'])
    train_loader, val_loader, train_dataset, val_dataset = get_data_loaders(
        DATA_CONFIG['image_dir'],
        batch_size=DATA_CONFIG['batch_size'],
        img_size=DATA_CONFIG['img_size'],
        num_workers=DATA_CONFIG['num_workers'],
        max_images=DATA_CONFIG['max_images']
    )
    
    # Setup rebalancing weights
    print("\nSetting up class rebalancing...")
    rebalancer = ColorRebalancer(ab_gamut, lambda_mix=REBALANCE_CONFIG['lambda_mix'])
    
    if os.path.exists(REBALANCE_CONFIG['weights_path']):
        rebalancer.load_weights(REBALANCE_CONFIG['weights_path'])
    else:
        if REBALANCE_CONFIG['compute_from_data']:
            rebalancer.compute_empirical_distribution(
                train_loader, 
                max_samples=REBALANCE_CONFIG['max_samples']
            )
            rebalancer.save_weights(REBALANCE_CONFIG['weights_path'])
        else:
            rebalancer.use_prior_weights()
            rebalancer.save_weights(REBALANCE_CONFIG['weights_path'])
    
    exp_type = 'full'
    
    trainer = create_experiment(
        exp_type=exp_type,
        ab_gamut=ab_gamut,
        rebalancer=rebalancer,
        device=DEVICE
    )
    
    # Check for checkpoint
    checkpoint_path = CHECKPOINT_PATHS.get(exp_type)
    start_epoch = 0
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        trainer.best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=TRAIN_CONFIG['num_epochs'],
        save_path=checkpoint_path,
        start_epoch=start_epoch
    )
    
    print(f"\nBest val loss: {results['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
