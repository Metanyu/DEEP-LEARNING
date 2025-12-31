import torch
import torch.nn as nn
import torch.nn.functional as F


class MultinomialCrossEntropyLoss(nn.Module):
    
    def __init__(self, ab_gamut, rebalancer=None, device='cuda', soft_encode_k=5):
        super().__init__()
        self.ab_gamut = ab_gamut
        self.rebalancer = rebalancer
        self.device = device
        self.Q = ab_gamut.Q
        self.soft_encode_k = soft_encode_k
        
        self.register_buffer('ab_centers', 
                            torch.from_numpy(ab_gamut.ab_gamut).float())
        
        if rebalancer is not None and rebalancer.weights is not None:
            self.register_buffer('class_weights',
                               torch.from_numpy(rebalancer.weights).float())
        else:
            self.class_weights = None
    
    def soft_encode_ab_fast(self, ab, sigma=5.0):
        B, _, H, W = ab.shape
        k = self.soft_encode_k
        
        ab_flat = ab.permute(0, 2, 3, 1).reshape(-1, 2)
        M = ab_flat.shape[0]
        
        distances = torch.cdist(ab_flat, self.ab_centers, p=2).pow(2)
        topk_dists, topk_indices = torch.topk(distances, k, dim=1, largest=False)
        topk_weights = torch.exp(-topk_dists / (2 * sigma ** 2))  # (M, k)
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Scatter into full distribution
        output = torch.zeros(M, self.Q, device=ab.device, dtype=ab.dtype)
        output.scatter_(1, topk_indices, topk_weights)
        
        # Reshape back: (M, Q) -> (B, H, W, Q) -> (B, Q, H, W)
        return output.reshape(B, H, W, self.Q).permute(0, 3, 1, 2)
    
    def get_pixel_weights_fast(self, ab):
        if self.class_weights is None:
            return None
        
        B, _, H, W = ab.shape
        ab_flat = ab.permute(0, 2, 3, 1).reshape(-1, 2)
        distances = torch.cdist(ab_flat, self.ab_centers, p=2)
        nearest_bins = torch.argmin(distances, dim=1)
        pixel_weights = self.class_weights[nearest_bins]
        return pixel_weights.reshape(B, H, W)
    
    def forward(self, pred_logits, target_ab):
        pred_h, pred_w = pred_logits.shape[2], pred_logits.shape[3]
        target_h, target_w = target_ab.shape[2], target_ab.shape[3]
        
        if pred_h != target_h or pred_w != target_w:
            target_ab = F.interpolate(target_ab, size=(pred_h, pred_w), mode='bilinear', align_corners=False)
        
        target_ab_denorm = target_ab * 128.0
        target_probs = self.soft_encode_ab_fast(target_ab_denorm)
        log_pred = F.log_softmax(pred_logits, dim=1)
        ce_loss = -torch.sum(target_probs * log_pred, dim=1)
        if self.class_weights is not None:
            pixel_weights = self.get_pixel_weights_fast(target_ab_denorm)
            ce_loss = ce_loss * pixel_weights
        return ce_loss.mean()


class L2LossWithRebalancing(nn.Module):
    
    def __init__(self, ab_gamut, rebalancer=None, device='cuda'):
        super().__init__()
        self.ab_gamut = ab_gamut
        self.rebalancer = rebalancer
        self.device = device
        
        self.register_buffer('ab_centers',
                            torch.from_numpy(ab_gamut.ab_gamut).float())
        
        if rebalancer is not None and rebalancer.weights is not None:
            self.register_buffer('class_weights',
                               torch.from_numpy(rebalancer.weights).float())
        else:
            self.class_weights = None
    
    def get_pixel_weights(self, ab):
        if self.class_weights is None:
            return None
        
        B, _, H, W = ab.shape
        ab_flat = ab.permute(0, 2, 3, 1).reshape(-1, 2)
        diff = ab_flat.unsqueeze(1) - self.ab_centers.unsqueeze(0)
        distances = torch.sum(diff ** 2, dim=2)
        nearest_bins = torch.argmin(distances, dim=1)
        pixel_weights = self.class_weights[nearest_bins]
        return pixel_weights.reshape(B, H, W)
    
    def forward(self, pred_ab, target_ab):
        pred_h, pred_w = pred_ab.shape[2], pred_ab.shape[3]
        target_h, target_w = target_ab.shape[2], target_ab.shape[3]
        
        if pred_h != target_h or pred_w != target_w:
            target_ab = F.interpolate(target_ab, size=(pred_h, pred_w), mode='bilinear', align_corners=False)
        
        l2_loss = torch.sum((pred_ab - target_ab) ** 2, dim=1)
        if self.class_weights is not None:
            target_ab_denorm = target_ab * 128.0
            pixel_weights = self.get_pixel_weights(target_ab_denorm)
            l2_loss = l2_loss * pixel_weights
        
        return l2_loss.mean()


class PureMSELoss(nn.Module):
    def forward(self, pred_ab, target_ab):
        return F.mse_loss(pred_ab, target_ab)
