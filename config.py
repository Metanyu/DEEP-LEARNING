import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_CONFIG = {
    'image_dir': 'val2017',
    'img_size': 256,
    'batch_size': 32,
    'num_workers': 4,
    'train_split': 0.8,
    'max_images': None,
}
MODEL_CONFIG = {
    'num_classes': 313,  # Number of ab bins (will be set by ABGamut)
    'dropout': 0.2,      # Dropout for regularization
    'temperature': 0.38,  # Annealed-mean temperature
}

# Training configuration
TRAIN_CONFIG = {
    'learning_rate': 3e-5,
    'num_epochs': 5,
    'weight_decay': 1e-4,  # L2 regularization
    'use_amp': True,       # Mixed precision training
    'verbose': False,      # Show progress bars
}

# Rebalancing configuration
REBALANCE_CONFIG = {
    'lambda_mix': 0.5,      # Mixing with uniform distribution
    'sigma': 5.0,           # Gaussian smoothing
    'weights_path': 'color_rebalancing_weights.npz',
    'compute_from_data': False,  # If False, use prior weights
    'max_samples': 20000,   # For computing empirical distribution
}

# Experiment types
EXPERIMENT_TYPES = {
    'full': 'Multinomial CE + Rebalancing (Paper)',
    'multinomial_only': 'Multinomial CE (No Rebalancing)',
    'l2_baseline': 'L2 Loss (Baseline)',
    'l2_rebalanced': 'L2 Loss + Rebalancing'
}

# Checkpoint paths for resuming training
CHECKPOINT_PATHS = {
    'full': 'model_full.pth',
    'multinomial_only': 'model_multinomial_only.pth',
    'l2_baseline': 'model_l2_baseline.pth',
    'l2_rebalanced': 'model_l2_rebalanced.pth',
}
