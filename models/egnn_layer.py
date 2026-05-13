"""
Edge Graph Neural Network (EGNN) Layer

This module implements the EGNN architecture adapted from fewshot-egnn
for time series forecasting with fuzzy BPA integration.

The EGNN operates on edge features and node features, updating them
iteratively through message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, Tuple, List


class NodeUpdateNetwork(nn.Module):
    """
    Node update network for EGNN.
    
    Updates node features by aggregating information from neighboring edges.
    Uses attention-like mechanism for weighted aggregation.
    
    Args:
        in_features: Input feature dimension
        num_features: Number of hidden features
        ratio: Ratio for hidden layer sizes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int,
        num_features: int,
        ratio: List[int] = [2, 1],
        dropout: float = 0.0,
    ):
        super(NodeUpdateNetwork, self).__init__()
        
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout
        
        # Build layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else in_features * 3,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False
            )
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(
                num_features=self.num_features_list[l]
            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()
            
            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)
        
        self.network = nn.Sequential(layer_list)
    
    def forward(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Update node features based on edge features.
        
        Args:
            node_feat: Node features of shape (batch, num_nodes, in_features)
            edge_feat: Edge features of shape (batch, 2, num_nodes, num_nodes)
            
        Returns:
            Updated node features of shape (batch, num_nodes, num_features)
        """
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        
        # Create diagonal mask
        diag_mask = 1.0 - torch.eye(num_data, device=node_feat.device).unsqueeze(0).unsqueeze(0)
        diag_mask = diag_mask.repeat(num_tasks, 2, 1, 1)
        
        # Normalize edge features
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)
        
        # Aggregate features from edges
        aggr_feat = torch.bmm(
            torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1),
            node_feat
        )
        
        # Concatenate with original features
        node_feat = torch.cat([
            node_feat,
            torch.cat(aggr_feat.split(num_data, 1), -1)
        ], -1).transpose(1, 2)
        
        # Apply network
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        
        return node_feat


class EdgeUpdateNetwork(nn.Module):
    """
    Edge update network for EGNN.
    
    Updates edge features based on node feature differences.
    Computes similarity/dissimilarity between nodes.
    
    Args:
        in_features: Input feature dimension
        num_features: Number of hidden features
        ratio: Ratio for hidden layer sizes
        separate_dissimilarity: Whether to use separate network for dissimilarity
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int,
        num_features: int,
        ratio: List[int] = [2, 2, 1, 1],
        separate_dissimilarity: bool = False,
        dropout: float = 0.0,
    ):
        super(EdgeUpdateNetwork, self).__init__()
        
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout
        
        # Build similarity network
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l-1] if l > 0 else in_features,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False
            )
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(
                num_features=self.num_features_list[l]
            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()
            
            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)
        
        layer_list['conv_out'] = nn.Conv2d(
            in_channels=self.num_features_list[-1],
            out_channels=1,
            kernel_size=1
        )
        self.sim_network = nn.Sequential(layer_list)
        
        # Optional separate dissimilarity network
        if self.separate_dissimilarity:
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                layer_list['conv{}'.format(l)] = nn.Conv2d(
                    in_channels=self.num_features_list[l-1] if l > 0 else in_features,
                    out_channels=self.num_features_list[l],
                    kernel_size=1,
                    bias=False
                )
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(
                    num_features=self.num_features_list[l]
                )
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                
                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)
            
            layer_list['conv_out'] = nn.Conv2d(
                in_channels=self.num_features_list[-1],
                out_channels=1,
                kernel_size=1
            )
            self.dsim_network = nn.Sequential(layer_list)
    
    def forward(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Update edge features based on node features.
        
        Args:
            node_feat: Node features of shape (batch, num_nodes, in_features)
            edge_feat: Edge features of shape (batch, 2, num_nodes, num_nodes)
            
        Returns:
            Updated edge features of shape (batch, 2, num_nodes, num_nodes)
        """
        # Compute pairwise differences
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)
        
        # Compute similarity
        sim_val = torch.sigmoid(self.sim_network(x_ij))
        
        if self.separate_dissimilarity:
            dsim_val = torch.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val
        
        # Apply diagonal mask
        diag_mask = 1.0 - torch.eye(node_feat.size(1), device=node_feat.device)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1)
        edge_feat = edge_feat * diag_mask
        
        merge_sum = torch.sum(edge_feat, -1, True)
        
        # Normalize and combine
        edge_feat = F.normalize(
            torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1
        ) * merge_sum
        
        # Add self-loops
        force_edge_feat = torch.cat(
            (torch.eye(node_feat.size(1), device=node_feat.device).unsqueeze(0),
             torch.zeros(node_feat.size(1), node_feat.size(1), device=node_feat.device).unsqueeze(0)),
            0
        ).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1)
        edge_feat = edge_feat + force_edge_feat
        
        # Normalize
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)
        
        return edge_feat


class EGNNLayer(nn.Module):
    """
    Complete EGNN layer combining node and edge updates.
    
    This layer performs one round of message passing:
    1. Update nodes based on edges
    2. Update edges based on nodes
    
    Args:
        in_features: Input feature dimension
        node_features: Number of node features
        edge_features: Number of edge features
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int,
        node_features: int,
        edge_features: int,
        dropout: float = 0.0,
    ):
        super(EGNNLayer, self).__init__()
        
        self.node_update = NodeUpdateNetwork(
            in_features=in_features,
            num_features=node_features,
            dropout=dropout
        )
        
        self.edge_update = EdgeUpdateNetwork(
            in_features=node_features,
            num_features=edge_features,
            separate_dissimilarity=False,
            dropout=dropout
        )
    
    def forward(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for EGNN layer.
        
        Args:
            node_feat: Node features
            edge_feat: Edge features
            
        Returns:
            Tuple of updated node and edge features
        """
        # Update nodes
        node_feat = self.node_update(node_feat, edge_feat)
        
        # Update edges
        edge_feat = self.edge_update(node_feat, edge_feat)
        
        return node_feat, edge_feat


class MultiLayerEGNN(nn.Module):
    """
    Multi-layer EGNN for deep graph neural network processing.
    
    Args:
        in_features: Input feature dimension
        node_features: Number of node features
        edge_features: Number of edge features
        num_layers: Number of EGNN layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int,
        node_features: int,
        edge_features: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super(MultiLayerEGNN, self).__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        for l in range(num_layers):
            self.layers.append(EGNNLayer(
                in_features=in_features if l == 0 else node_features,
                node_features=node_features,
                edge_features=edge_features,
                dropout=dropout if l < num_layers - 1 else 0.0
            ))
    
    def forward(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through all EGNN layers.
        
        Args:
            node_feat: Initial node features
            edge_feat: Initial edge features
            
        Returns:
            Tuple of final node features and list of edge features from each layer
        """
        edge_feat_list = []
        
        for layer in self.layers:
            node_feat, edge_feat = layer(node_feat, edge_feat)
            edge_feat_list.append(edge_feat)
        
        return node_feat, edge_feat_list


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for alternative message passing.
    
    Uses multi-head attention for node feature updates.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super(GraphAttentionLayer, self).__init__()
        
        self.num_heads = num_heads
        self.out_features = out_features
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.out_proj = nn.Linear(in_features, out_features)
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_features)
    
    def forward(
        self,
        node_feat: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for graph attention layer.
        
        Args:
            node_feat: Node features of shape (batch, num_nodes, in_features)
            adj: Optional adjacency matrix for masking
            
        Returns:
            Updated node features of shape (batch, num_nodes, out_features)
        """
        # Self-attention
        attn_out, _ = self.attention(node_feat, node_feat, node_feat)
        
        # Project and normalize
        out = self.out_proj(attn_out)
        out = self.norm(out + self.out_proj(node_feat))
        
        return out
