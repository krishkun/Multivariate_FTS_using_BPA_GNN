"""
Fuzzy Basic Probability Assignment (BPA) Module

This module implements the core contribution of the thesis:
- Fuzzy membership functions for uncertainty quantification
- Evidence Machine Kernel for BPA computation
- Integration with Dempster-Shafer theory

Based on TEFN (Time Evidence Fusion Network) and extended for GNN integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Literal, Dict, Any


class FuzzyMembershipFunction(nn.Module):
    """
    Fuzzy membership functions for computing Basic Probability Assignments.
    
    Supports multiple membership function types:
    - Gaussian: μ(x) = exp(-(x-c)²/2σ²)
    - Triangular: μ(x) = max(0, 1 - |x-c|/w)
    - Trapezoidal: μ(x) = max(0, min((x-a)/(b-a), 1, (d-x)/(d-c)))
    - Sigmoid: μ(x) = 1/(1 + exp(-k(x-c)))
    """
    
    def __init__(
        self,
        num_classes: int,
        membership_type: Literal['gaussian', 'triangular', 'trapezoidal', 'sigmoid'] = 'gaussian',
        learnable: bool = True,
    ):
        super(FuzzyMembershipFunction, self).__init__()
        
        self.num_classes = num_classes
        self.membership_type = membership_type
        self.learnable = learnable
        
        # Initialize membership parameters
        if learnable:
            # Centers for each class
            self.centers = nn.Parameter(
                torch.linspace(-1, 1, num_classes), requires_grad=True
            )
            # Widths for each class
            self.widths = nn.Parameter(
                torch.ones(num_classes) * 0.5, requires_grad=True
            )
        else:
            self.register_buffer(
                'centers', torch.linspace(-1, 1, num_classes)
            )
            self.register_buffer(
                'widths', torch.ones(num_classes) * 0.5
            )
    
    def gaussian(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian membership function."""
        x = x.unsqueeze(-1)  # (batch, seq, 1)
        centers = self.centers.view(1, 1, -1)  # (1, 1, num_classes)
        widths = self.widths.view(1, 1, -1).abs() + 1e-6
        return torch.exp(-((x - centers) ** 2) / (2 * widths ** 2))
    
    def triangular(self, x: torch.Tensor) -> torch.Tensor:
        """Triangular membership function."""
        x = x.unsqueeze(-1)
        centers = self.centers.view(1, 1, -1)
        widths = self.widths.view(1, 1, -1).abs() + 1e-6
        return torch.relu(1 - torch.abs(x - centers) / widths)
    
    def trapezoidal(self, x: torch.Tensor) -> torch.Tensor:
        """Trapezoidal membership function."""
        x = x.unsqueeze(-1)
        centers = self.centers.view(1, 1, -1)
        widths = self.widths.view(1, 1, -1).abs() + 1e-6
        
        # Define trapezoid corners
        a = centers - widths
        b = centers - widths * 0.5
        c = centers + widths * 0.5
        d = centers + widths
        
        left = (x - a) / (b - a + 1e-6)
        right = (d - x) / (d - c + 1e-6)
        
        return torch.relu(torch.min(
            torch.clamp(left, 0, 1),
            torch.clamp(right, 0, 1)
        ))
    
    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid membership function."""
        x = x.unsqueeze(-1)
        centers = self.centers.view(1, 1, -1)
        widths = self.widths.view(1, 1, -1).abs() + 1e-6
        return torch.sigmoid((x - centers) / widths)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute membership values for input.
        
        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, seq_len, features)
            
        Returns:
            Membership values of shape (batch, seq_len, num_classes)
        """
        if x.dim() == 3:
            x = x.mean(dim=-1)  # Average across features
        
        if self.membership_type == 'gaussian':
            return self.gaussian(x)
        elif self.membership_type == 'triangular':
            return self.triangular(x)
        elif self.membership_type == 'trapezoidal':
            return self.trapezoidal(x)
        elif self.membership_type == 'sigmoid':
            return self.sigmoid(x)
        else:
            raise ValueError(f"Unknown membership type: {self.membership_type}")


class EvidenceMachineKernel(nn.Module):
    """
    Evidence Machine Kernel for computing Basic Probability Assignments.
    
    This module maps input features to evidence (BPA) space using:
    1. Learnable class-specific weights and biases
    2. Non-linear activation functions
    3. Residual connections
    
    The output represents the degree of belief for each hypothesis class.
    
    Args:
        num_classes: Number of hypothesis classes (F)
        num_features: Number of input features (C)
        activation: Activation function type
        use_residual: Whether to use residual connections
    """
    
    def __init__(
        self,
        num_classes: int,
        num_features: int,
        activation: Optional[str] = None,
        use_residual: bool = True,
    ):
        super(EvidenceMachineKernel, self).__init__()
        
        self.num_classes = num_classes
        self.num_features = num_features
        self.F = 2 ** num_features  # Feature dimension
        self.use_residual = use_residual
        
        # Class-specific weights and biases
        self.C_weight = nn.Parameter(torch.randn(num_classes, self.F))
        self.C_bias = nn.Parameter(torch.randn(num_classes, self.F))
        
        # Activation function
        self.activation = self._get_activation(activation, self.num_classes * self.F)
    
    def _get_activation(self, activation: Optional[str], dim: int) -> Optional[nn.Module]:
        """Get activation function by name."""
        if activation is None or activation == 'linear':
            return None
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'swish':
            return nn.SiLU()
        elif activation == 'mish':
            return nn.Mish()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'mlp':
            return nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim)
            )
        elif activation == 'attn':
            return nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute evidence (BPA) from input features.
        
        Args:
            x: Input tensor of shape (batch, time, classes)
            
        Returns:
            Evidence tensor of shape (batch, time, classes, F)
        """
        # Compute evidence using einsum
        evidence = torch.einsum('btc,cf->btcf', x, self.C_weight) + self.C_bias
        
        B, T, C, F = evidence.shape
        merged = evidence.reshape(B, T, -1)
        
        # Apply activation
        if self.activation is not None:
            if isinstance(self.activation, nn.MultiheadAttention):
                act_out, _ = self.activation(merged, merged, merged)
            else:
                act_out = self.activation(merged)
            
            if self.use_residual:
                merged = act_out + merged
            else:
                merged = act_out
        
        return merged.reshape(B, T, C, F)


class FuzzyBPAModule(nn.Module):
    """
    Fuzzy Basic Probability Assignment Module.
    
    This is the core contribution module that:
    1. Computes fuzzy membership values for input data
    2. Generates Basic Probability Assignments (BPAs)
    3. Combines evidence from multiple sources using Dempster's rule
    
    The module integrates fuzzy set theory with Dempster-Shafer evidence theory
    to provide interpretable uncertainty quantification.
    
    Args:
        input_dim: Dimension of input features
        num_hypotheses: Number of hypothesis classes
        membership_type: Type of fuzzy membership function
        fusion_method: Method for combining evidence ('dempster', 'murphy', 'average')
    """
    
    def __init__(
        self,
        input_dim: int,
        num_hypotheses: int = 8,
        membership_type: str = 'gaussian',
        fusion_method: Literal['dempster', 'murphy', 'average', 'yager'] = 'dempster',
    ):
        super(FuzzyBPAModule, self).__init__()
        
        self.input_dim = input_dim
        self.num_hypotheses = num_hypotheses
        self.fusion_method = fusion_method
        
        # Fuzzy membership functions
        self.membership_fn = FuzzyMembershipFunction(
            num_classes=num_hypotheses,
            membership_type=membership_type,
            learnable=True,
        )
        
        # Evidence machine kernels for time and channel dimensions
        self.time_kernel = EvidenceMachineKernel(
            num_classes=input_dim,
            num_features=3,  # 2^3 = 8 features
            activation='gelu',
            use_residual=True,
        )
        
        self.channel_kernel = EvidenceMachineKernel(
            num_classes=input_dim,
            num_features=3,
            activation='gelu',
            use_residual=True,
        )
        
        # Fusion layer
        self.fusion_layer = nn.Linear(2 * 8, 8)  # 2 * F -> F
    
    def dempster_combination(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """
        Dempster's rule of combination for combining evidence.
        
        m1 ⊕ m2(A) = (1/(1-K)) * Σ m1(B) * m2(C) where B∩C=A
        
        where K = Σ m1(B) * m2(C) where B∩C=∅
        
        Args:
            m1: First BPA of shape (..., num_hypotheses)
            m2: Second BPA of shape (..., num_hypotheses)
            
        Returns:
            Combined BPA of shape (..., num_hypotheses)
        """
        # Compute conflict
        conflict = 1 - torch.sum(m1 * m2, dim=-1, keepdim=True)
        
        # Apply Dempster's rule
        combined = m1 * m2 / (conflict + 1e-8)
        
        return combined
    
    def murphy_combination(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """
        Murphy's average combination rule.
        
        Simple averaging of BPAs, more robust to conflict.
        
        Args:
            m1: First BPA
            m2: Second BPA
            
        Returns:
            Combined BPA
        """
        return (m1 + m2) / 2
    
    def yager_combination(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """
        Yager's combination rule.
        
        Assigns conflict to the universal set.
        
        Args:
            m1: First BPA
            m2: Second BPA
            
        Returns:
            Combined BPA
        """
        # Compute conflict
        conflict = 1 - torch.sum(m1 * m2, dim=-1, keepdim=True)
        
        # Normal combination
        combined = m1 * m2
        
        # Add conflict to universal set (all hypotheses)
        combined = combined + conflict / self.num_hypotheses
        
        return combined
    
    def combine_evidence(
        self,
        m1: torch.Tensor,
        m2: torch.Tensor,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Combine two BPAs using specified method.
        
        Args:
            m1: First BPA
            m2: Second BPA
            method: Combination method (overrides default)
            
        Returns:
            Combined BPA
        """
        method = method or self.fusion_method
        
        if method == 'dempster':
            return self.dempster_combination(m1, m2)
        elif method == 'murphy' or method == 'average':
            return self.murphy_combination(m1, m2)
        elif method == 'yager':
            return self.yager_combination(m1, m2)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
    
    def compute_bpa(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Basic Probability Assignment from input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Dictionary containing:
                - 'time_evidence': Evidence from time dimension
                - 'channel_evidence': Evidence from channel dimension
                - 'fused_evidence': Combined evidence
                - 'membership': Fuzzy membership values
        """
        B, T, C = x.shape
        
        # Compute fuzzy membership
        membership = self.membership_fn(x)  # (B, T, num_hypotheses)
        
        # Compute evidence from time dimension
        time_evidence = self.time_kernel(x)  # (B, T, C, F)
        time_evidence = time_evidence.mean(dim=-1)  # Average over F
        time_evidence = F.softmax(time_evidence, dim=-1)
        
        # Compute evidence from channel dimension
        channel_evidence = self.channel_kernel(x.transpose(1, 2).transpose(-1, -2))
        channel_evidence = channel_evidence.mean(dim=-1)
        channel_evidence = F.softmax(channel_evidence, dim=-1)
        
        # Combine evidence
        fused_evidence = self.combine_evidence(time_evidence, channel_evidence)
        
        return {
            'time_evidence': time_evidence,
            'channel_evidence': channel_evidence,
            'fused_evidence': fused_evidence,
            'membership': membership,
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for BPA computation.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Fused evidence tensor of shape (batch, seq_len, num_hypotheses)
        """
        bpa_dict = self.compute_bpa(x)
        return bpa_dict['fused_evidence']


class BeliefFunction(nn.Module):
    """
    Belief and Plausibility computation from BPA.
    
    In Dempster-Shafer theory:
    - Bel(A) = Σ m(B) where B ⊆ A
    - Pl(A) = Σ m(B) where B ∩ A ≠ ∅
    - Uncertainty = Pl(A) - Bel(A)
    """
    
    def __init__(self, num_hypotheses: int):
        super(BeliefFunction, self).__init__()
        self.num_hypotheses = num_hypotheses
    
    def compute_belief(self, bpa: torch.Tensor) -> torch.Tensor:
        """
        Compute belief function from BPA.
        
        Args:
            bpa: BPA tensor of shape (..., num_hypotheses)
            
        Returns:
            Belief values
        """
        # For singleton sets, belief equals the mass
        return bpa
    
    def compute_plausibility(self, bpa: torch.Tensor) -> torch.Tensor:
        """
        Compute plausibility function from BPA.
        
        Args:
            bpa: BPA tensor of shape (..., num_hypotheses)
            
        Returns:
            Plausibility values
        """
        # For singleton sets, plausibility equals the mass
        return bpa
    
    def compute_uncertainty(self, bpa: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty from BPA.
        
        Args:
            bpa: BPA tensor of shape (..., num_hypotheses)
            
        Returns:
            Uncertainty values
        """
        belief = self.compute_belief(bpa)
        plausibility = self.compute_plausibility(bpa)
        return plausibility - belief
    
    def forward(self, bpa: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute belief functions from BPA.
        
        Args:
            bpa: BPA tensor
            
        Returns:
            Dictionary with belief, plausibility, and uncertainty
        """
        return {
            'belief': self.compute_belief(bpa),
            'plausibility': self.compute_plausibility(bpa),
            'uncertainty': self.compute_uncertainty(bpa),
        }
