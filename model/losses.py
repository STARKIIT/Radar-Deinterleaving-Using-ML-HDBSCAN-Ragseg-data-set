import torch
import torch.nn as nn
import torch.nn.functional as F

def batch_all_triplet(z, labels, margin=1.9):
    """Compute batch-all triplet loss (labels: (B,T) ints)."""
    B, T, D = z.shape
    z = z.reshape(B*T, D)
    lab = labels.reshape(-1)
    dist = torch.cdist(z, z, p=2)              # (N,N)
    anchor_equal = lab.unsqueeze(0) == lab.unsqueeze(1)
    pos_mask  = anchor_equal.float() - torch.eye(B*T, device=z.device)
    neg_mask  = 1.0 - anchor_equal.float()

    ap_dist = dist.unsqueeze(2)
    an_dist = dist.unsqueeze(1)

    loss = F.relu(ap_dist - an_dist + margin)
    valid = pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)
    loss  = (loss * valid).sum() / (valid.sum() + 1e-12)
    return loss

class ContrastiveLoss(nn.Module):
    """Modern contrastive loss for radar deinterleaving."""
    
    def __init__(self, temperature=0.1, margin=1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B*T, D) tensor of embeddings
            labels: (B*T,) tensor of emitter labels
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive and negative masks
        labels = labels.unsqueeze(0)
        pos_mask = (labels == labels.T).float()
        neg_mask = (labels != labels.T).float()
        
        # Remove self-similarity
        eye = torch.eye(embeddings.size(0), device=embeddings.device)
        pos_mask = pos_mask - eye
        
        # Compute InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        all_sum = (exp_sim * (1 - eye)).sum(dim=1)
        
        loss = -torch.log(pos_sum / (all_sum + 1e-8))
        return loss.mean()

class TripletLossWithHardMining(nn.Module):
    """Advanced triplet loss with hard negative mining."""
    
    def __init__(self, margin=1.0, hard_mining=True):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining
    
    def forward(self, embeddings, labels):
        """Enhanced triplet loss with hard negative mining."""
        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # Create masks for positives and negatives
        labels_expanded = labels.unsqueeze(0)
        pos_mask = (labels_expanded == labels_expanded.T).float()
        neg_mask = (labels_expanded != labels_expanded.T).float()
        
        # Remove diagonal
        eye = torch.eye(embeddings.size(0), device=embeddings.device)
        pos_mask = pos_mask - eye
        
        if self.hard_mining:
            # Hard positive mining (furthest positive)
            pos_dist = dist_matrix * pos_mask + (1 - pos_mask) * (-1e8)
            hard_pos_dist = pos_dist.max(dim=1)[0]
            
            # Hard negative mining (closest negative)
            neg_dist = dist_matrix * neg_mask + (1 - neg_mask) * 1e8
            hard_neg_dist = neg_dist.min(dim=1)[0]
            
            # Compute triplet loss
            loss = F.relu(hard_pos_dist - hard_neg_dist + self.margin)
        else:
            # All valid triplets
            pos_dist = dist_matrix.unsqueeze(2)  # (N, N, 1)
            neg_dist = dist_matrix.unsqueeze(1)  # (N, 1, N)
            
            triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
            
            # Apply masks
            valid_triplets = pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)
            loss = (triplet_loss * valid_triplets).sum() / (valid_triplets.sum() + 1e-8)
        
        return loss.mean()

class TemporalConsistencyLoss(nn.Module):
    """Encourage temporal consistency in embeddings."""
    
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, embeddings, timestamps):
        """
        Args:
            embeddings: (B, T, D) sequence embeddings
            timestamps: (B, T) timestamps for each pulse
        """
        B, T, D = embeddings.shape
        
        # Compute temporal differences
        time_diffs = torch.diff(timestamps, dim=1)  # (B, T-1)
        embed_diffs = torch.diff(embeddings, dim=1)  # (B, T-1, D)
        
        # Normalize by time differences
        embed_diff_norms = torch.norm(embed_diffs, p=2, dim=2)  # (B, T-1)
        
        # Penalize large embedding changes for small time differences
        consistency_loss = (embed_diff_norms / (time_diffs + 1e-8)).mean()
        
        return self.alpha * consistency_loss

def combined_loss(embeddings, labels, timestamps=None, 
                 loss_weights={'contrastive': 1.0, 'triplet': 0.5, 'temporal': 0.1}):
    """Combine multiple loss functions for robust training."""
    total_loss = 0
    
    if loss_weights['contrastive'] > 0:
        contrastive_fn = ContrastiveLoss()
        total_loss += loss_weights['contrastive'] * contrastive_fn(embeddings, labels)
    
    if loss_weights['triplet'] > 0:
        triplet_fn = TripletLossWithHardMining()
        total_loss += loss_weights['triplet'] * triplet_fn(embeddings, labels)
    
    if timestamps is not None and loss_weights['temporal'] > 0:
        temporal_fn = TemporalConsistencyLoss()
        # Reshape for temporal loss
        B = int(embeddings.size(0) / timestamps.size(1))
        emb_reshaped = embeddings.view(B, -1, embeddings.size(1))
        total_loss += loss_weights['temporal'] * temporal_fn(emb_reshaped, timestamps)
    
    return total_loss
