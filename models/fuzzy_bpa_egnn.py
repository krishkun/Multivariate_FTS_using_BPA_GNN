"""
Fuzzy BPA EGNN: Main Model

This module integrates:
1. Graph Construction for time series
2. Fuzzy BPA Module for evidence computation
3. EGNN for graph neural network processing
4. Prediction head for time series forecasting

The model provides interpretable predictions with uncertainty quantification
using Dempster-Shafer evidence theory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Literal

from .graph_constructor import TimeSeriesGraphConstructor, DynamicGraphConstructor
from .fuzzy_bpa import FuzzyBPAModule, EvidenceMachineKernel, BeliefFunction
from .egnn_layer import MultiLayerEGNN, GraphAttentionLayer


class NormLayer(nn.Module):
    """Normalization layer for time series."""
    
    def __init__(self):
        super(NormLayer, self).__init__()
        self.means = None
        self.stds = None
    
    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input."""
        self.means = x.mean(dim=1, keepdim=True).detach()
        x = x - self.means
        self.stds = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / self.stds
        return x
    
    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output."""
        x = x * self.stds + self.means
        return x


class FuzzyBPAEGNN(nn.Module):
    """
    Fuzzy BPA EGNN: Fuzzy Basic Probability Assignment Edge Graph Neural Network
    
    This is the main model that combines:
    1. Time series to graph conversion
    2. Fuzzy membership-based BPA computation
    3. EGNN-based graph processing
    4. Evidence-based prediction with uncertainty
    
    Args:
        config: Configuration object with model parameters
    """
    
    def __init__(self, config):
        super(FuzzyBPAEGNN, self).__init__()
        
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_variables = config.enc_in
        
        # Configuration flags
        self.use_norm = getattr(config, 'use_norm', True)
        self.use_graph = getattr(config, 'use_graph', True)
        self.use_bpa = getattr(config, 'use_bpa', True)
        self.fusion_method = getattr(config, 'fusion_method', 'dempster')
        self.membership_type = getattr(config, 'membership_type', 'gaussian')
        
        # Normalization layer
        if self.use_norm:
            self.norm_layer = NormLayer()
        
        # Input projection
        self.input_proj = nn.Linear(self.num_variables, config.d_model)
        
        # Graph constructor
        if self.use_graph:
            self.graph_constructor = TimeSeriesGraphConstructor(
                num_nodes=self.seq_len * self.num_variables,
                seq_len=self.seq_len,
                num_variables=self.num_variables,
                k_neighbors=getattr(config, 'k_neighbors', 10),
                graph_type=getattr(config, 'graph_type', 'adaptive'),
                threshold=getattr(config, 'graph_threshold', 0.1),
                use_learnable_adj=getattr(config, 'use_learnable_adj', True),
            )
        
        # Fuzzy BPA module
        if self.use_bpa:
            self.bpa_module = FuzzyBPAModule(
                input_dim=config.d_model,
                num_hypotheses=getattr(config, 'num_hypotheses', 8),
                membership_type=self.membership_type,
                fusion_method=self.fusion_method,
            )
            
            # Evidence machine kernels for time and channel
            self.time_kernel = EvidenceMachineKernel(
                num_classes=self.seq_len,
                num_features=getattr(config, 'e_layers', 3),
                activation=getattr(config, 'kernel_activation', 'gelu'),
                use_residual=getattr(config, 'use_residual', True),
            )
            
            self.channel_kernel = EvidenceMachineKernel(
                num_classes=self.num_variables,
                num_features=getattr(config, 'e_layers', 3),
                activation=getattr(config, 'kernel_activation', 'gelu'),
                use_residual=getattr(config, 'use_residual', True),
            )
        
        # EGNN layers
        if self.use_graph:
            self.egnn = MultiLayerEGNN(
                in_features=config.d_model,
                node_features=getattr(config, 'node_features', 64),
                edge_features=getattr(config, 'edge_features', 64),
                num_layers=getattr(config, 'num_layers', 3),
                dropout=getattr(config, 'dropout', 0.1),
            )
        
        # Temporal processing
        self.temporal_proj = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        
        # Fusion layer for combining evidence
        if self.fusion_method == 'concat':
            self.fusion_linear = nn.Linear(
                2 * (2 ** getattr(config, 'e_layers', 3)),
                2 ** getattr(config, 'e_layers', 3)
            )
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, self.num_variables)
        
        # Belief function for uncertainty
        self.belief_fn = BeliefFunction(
            num_hypotheses=getattr(config, 'num_hypotheses', 8)
        )
        
        # Optional probabilistic layer
        if getattr(config, 'use_probabilistic_layer', False):
            self.probabilistic_layer = nn.Dropout(p=config.dropout)
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input time series."""
        if self.use_norm:
            x = self.norm_layer.norm(x)
        return x
    
    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Postprocess output predictions."""
        if self.use_norm:
            x = self.norm_layer.denorm(x)
        return x
    
    def construct_graph(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct graph from time series.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_variables)
            
        Returns:
            Tuple of node features and adjacency matrix
        """
        return self.graph_constructor(x)
    
    def compute_evidence(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute evidence (BPA) from input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Dictionary with evidence tensors
        """
        B, T, C = x.shape
        
        # Time dimension evidence
        t_out = self.time_kernel(x.permute(0, 2, 1)).permute(0, 2, 1, 3)
        
        # Channel dimension evidence
        c_out = self.channel_kernel(x)
        
        # Fuse evidence
        if self.fusion_method == 'add':
            fused = t_out + c_out
        elif self.fusion_method == 'concat':
            fused = torch.cat([t_out, c_out], dim=-1)
            fused = self.fusion_linear(fused)
        else:
            fused = t_out + c_out
        
        return {
            'time_evidence': t_out,
            'channel_evidence': c_out,
            'fused_evidence': fused,
        }
    
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Fuzzy BPA EGNN.
        
        Args:
            x_enc: Input encoder sequence of shape (batch, seq_len, num_variables)
            x_mark_enc: Optional temporal markers for encoder
            x_dec: Optional decoder input
            x_mark_dec: Optional temporal markers for decoder
            mask: Optional mask
            
        Returns:
            Predictions of shape (batch, pred_len, num_variables)
        """
        # Preprocess
        x = self.preprocess(x_enc)
        
        # Input projection
        x = self.input_proj(x)  # (B, T, d_model)
        
        # Compute evidence
        if self.use_bpa:
            evidence_dict = self.compute_evidence(x)
            evidence = evidence_dict['fused_evidence']
            
            # Reduce evidence to feature dimension
            B, T, C, F = evidence.shape
            evidence_flat = evidence.reshape(B, T, -1)
            
            # Combine with input
            x = x + 0.1 * evidence_flat[:, :, :x.size(-1)]
        
        # Graph construction and EGNN processing
        if self.use_graph:
            # Construct graph
            node_feat, adj = self.construct_graph(x_enc)
            
            # Initialize edge features from adjacency
            edge_feat = torch.stack([
                adj.mean(dim=1),
                1 - adj.mean(dim=1)
            ], dim=1)  # (B, 2, N, N)
            
            # Project node features
            node_feat = self.input_proj(x_enc)
            B, T, D = node_feat.shape
            node_feat = node_feat.reshape(B, -1, D)  # Flatten to nodes
            
            # Apply EGNN
            node_feat, edge_feat_list = self.egnn(node_feat, edge_feat)
            
            # Reshape back to time series
            x = node_feat.reshape(B, T, -1)
        
        # Temporal projection
        x = self.temporal_proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Output projection
        output = self.output_proj(x)
        
        # Get prediction window
        output = output[:, -self.pred_len:, :]
        
        # Postprocess
        output = self.postprocess(output)
        
        return output
    
    def predict_with_uncertainty(
        self,
        x_enc: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            x_enc: Input encoder sequence
            
        Returns:
            Dictionary with predictions and uncertainty metrics
        """
        # Preprocess
        x = self.preprocess(x_enc)
        
        # Input projection
        x = self.input_proj(x)
        
        # Compute evidence
        if self.use_bpa:
            evidence_dict = self.compute_evidence(x)
            evidence = evidence_dict['fused_evidence']
            
            # Compute belief functions
            evidence_flat = evidence.mean(dim=-1)  # Average over F dimension
            belief_dict = self.belief_fn(evidence_flat)
        
        # Forward pass
        output = self.forward(x_enc, **kwargs)
        
        return {
            'prediction': output,
            'evidence': evidence_dict if self.use_bpa else None,
            'belief': belief_dict if self.use_bpa else None,
            'uncertainty': belief_dict['uncertainty'] if self.use_bpa else None,
        }


class FuzzyBPAEGNNConfig:
    """Configuration class for FuzzyBPAEGNN."""
    
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        enc_in: int = 7,
        d_model: int = 64,
        e_layers: int = 3,
        num_layers: int = 3,
        node_features: int = 64,
        edge_features: int = 64,
        num_hypotheses: int = 8,
        k_neighbors: int = 10,
        dropout: float = 0.1,
        use_norm: bool = True,
        use_graph: bool = True,
        use_bpa: bool = True,
        fusion_method: str = 'dempster',
        membership_type: str = 'gaussian',
        kernel_activation: str = 'gelu',
        use_residual: bool = True,
        graph_type: str = 'adaptive',
        graph_threshold: float = 0.1,
        use_learnable_adj: bool = True,
        use_probabilistic_layer: bool = False,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.e_layers = e_layers
        self.num_layers = num_layers
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_hypotheses = num_hypotheses
        self.k_neighbors = k_neighbors
        self.dropout = dropout
        self.use_norm = use_norm
        self.use_graph = use_graph
        self.use_bpa = use_bpa
        self.fusion_method = fusion_method
        self.membership_type = membership_type
        self.kernel_activation = kernel_activation
        self.use_residual = use_residual
        self.graph_type = graph_type
        self.graph_threshold = graph_threshold
        self.use_learnable_adj = use_learnable_adj
        self.use_probabilistic_layer = use_probabilistic_layer


def create_model(config) -> FuzzyBPAEGNN:
    """Factory function to create FuzzyBPAEGNN model."""
    return FuzzyBPAEGNN(config)
