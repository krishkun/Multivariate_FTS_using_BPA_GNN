"""
Graph Construction Module for Multivariate Fuzzy Time Series GNN

This module constructs graphs from multivariate time series data by:
1. Creating nodes for each time step and variable
2. Computing edge weights based on temporal and variable correlations
3. Supporting multiple graph construction strategies
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Literal


class TimeSeriesGraphConstructor(nn.Module):
    """
    Constructs graphs from multivariate time series data for GNN processing.
    
    The graph construction follows these principles:
    - Nodes represent time-variable pairs
    - Edges capture temporal dependencies and variable correlations
    - Edge weights are computed using fuzzy membership functions
    
    Args:
        num_nodes: Number of nodes (typically seq_len * num_variables)
        seq_len: Length of input sequence
        num_variables: Number of variables/channels in time series
        k_neighbors: Number of nearest neighbors for graph sparsification
        graph_type: Type of graph construction ('adaptive', 'static', 'dynamic')
        threshold: Threshold for edge creation (default: 0.1)
        use_learnable_adj: Whether to use learnable adjacency matrix
    """
    
    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        num_variables: int,
        k_neighbors: int = 10,
        graph_type: Literal['adaptive', 'static', 'dynamic'] = 'adaptive',
        threshold: float = 0.1,
        use_learnable_adj: bool = True,
    ):
        super(TimeSeriesGraphConstructor, self).__init__()
        
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.num_variables = num_variables
        self.k_neighbors = k_neighbors
        self.graph_type = graph_type
        self.threshold = threshold
        self.use_learnable_adj = use_learnable_adj
        
        # Node embedding for learnable adjacency
        if use_learnable_adj:
            self.node_embedding = nn.Parameter(
                torch.randn(num_nodes, 64), requires_grad=True
            )
            self.adj_linear = nn.Linear(64, 64, bias=False)
        
        # Temporal distance encoding
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Variable correlation encoder
        self.variable_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Fuzzy membership parameters for edge weight computation
        self.membership_center = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.membership_width = nn.Parameter(torch.ones(1), requires_grad=True)
        
    def compute_temporal_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal adjacency matrix based on time step proximity.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_variables)
            
        Returns:
            Temporal adjacency matrix of shape (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Create temporal distance matrix
        time_idx = torch.arange(seq_len, device=x.device).float()
        time_dist = torch.abs(time_idx.unsqueeze(1) - time_idx.unsqueeze(0))
        
        # Normalize distances
        time_dist_norm = time_dist / (seq_len - 1)
        
        # Apply fuzzy membership function (Gaussian-like)
        temporal_adj = torch.exp(-time_dist_norm ** 2 / (2 * self.membership_width ** 2))
        
        return temporal_adj.unsqueeze(0).expand(batch_size, -1, -1)
    
    def compute_variable_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute variable correlation adjacency matrix.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_variables)
            
        Returns:
            Variable adjacency matrix of shape (batch, num_variables, num_variables)
        """
        batch_size, seq_len, num_vars = x.shape
        
        # Compute correlation between variables across time
        x_centered = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / (seq_len - 1)
        
        # Normalize to get correlation
        std = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).unsqueeze(-1) + 1e-8)
        corr = cov / (std @ std.transpose(1, 2) + 1e-8)
        
        # Apply ReLU to keep only positive correlations
        var_adj = torch.relu(corr)
        
        return var_adj
    
    def compute_adaptive_adjacency(self) -> torch.Tensor:
        """
        Compute learnable adaptive adjacency matrix.
        
        Returns:
            Adaptive adjacency matrix of shape (num_nodes, num_nodes)
        """
        if not self.use_learnable_adj:
            return torch.eye(self.num_nodes, device=self.node_embedding.device)
        
        # Compute node similarities
        node_features = self.adj_linear(self.node_embedding)
        adj = torch.mm(node_features, node_features.t())
        
        # Apply ReLU and normalize
        adj = torch.relu(adj)
        adj = adj / (adj.max() + 1e-8)
        
        return adj
    
    def construct_full_graph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct the full graph from input time series.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_variables)
            
        Returns:
            Tuple of:
                - Node features of shape (batch, num_nodes, node_feat_dim)
                - Adjacency matrix of shape (batch, num_nodes, num_nodes)
        """
        batch_size, seq_len, num_vars = x.shape
        num_nodes = seq_len * num_vars
        
        # Reshape input to node features
        # Each node represents a (time_step, variable) pair
        node_features = x.reshape(batch_size, num_nodes, 1)
        
        # Compute temporal adjacency
        temporal_adj = self.compute_temporal_adjacency(x)
        
        # Compute variable adjacency
        var_adj = self.compute_variable_adjacency(x)
        
        # Construct full adjacency using Kronecker product structure
        # adj_full[i,j] = temporal_adj[t1,t2] * var_adj[v1,v2]
        # where i = t1 * num_vars + v1, j = t2 * num_vars + v2
        
        # Efficient computation using einsum
        adj_full = torch.einsum('bts,bvw->btvsw', temporal_adj, var_adj)
        adj_full = adj_full.reshape(batch_size, num_nodes, num_nodes)
        
        # Add adaptive adjacency if using learnable version
        if self.use_learnable_adj and self.graph_type == 'adaptive':
            adaptive_adj = self.compute_adaptive_adjacency()
            adj_full = 0.5 * adj_full + 0.5 * adaptive_adj.unsqueeze(0)
        
        # Sparsify using threshold
        adj_full = adj_full * (adj_full > self.threshold).float()
        
        # Add self-loops
        eye = torch.eye(num_nodes, device=x.device).unsqueeze(0)
        adj_full = adj_full + eye
        
        # Normalize adjacency (symmetric normalization)
        degree = adj_full.sum(dim=-1, keepdim=True)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        adj_normalized = degree_inv_sqrt * adj_full * degree_inv_sqrt.transpose(-1, -2)
        
        return node_features, adj_normalized
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for graph construction.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_variables)
            
        Returns:
            Tuple of node features and adjacency matrix
        """
        return self.construct_full_graph(x)


class DynamicGraphConstructor(nn.Module):
    """
    Dynamic graph constructor that updates graph structure during training.
    
    This module learns to construct graphs adaptively based on input data,
    combining static structural information with dynamic patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_nodes: int,
        num_heads: int = 4,
    ):
        super(DynamicGraphConstructor, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        
        # Query, Key, Value projections for attention-based graph construction
        self.query_proj = nn.Linear(input_dim, hidden_dim * num_heads)
        self.key_proj = nn.Linear(input_dim, hidden_dim * num_heads)
        self.value_proj = nn.Linear(input_dim, hidden_dim * num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim * num_heads, input_dim)
        
        # Temperature parameter for attention
        self.temperature = nn.Parameter(torch.ones(1) * np.sqrt(hidden_dim))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct dynamic graph using multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, num_nodes, input_dim)
            
        Returns:
            Tuple of updated node features and attention-based adjacency
        """
        batch_size, num_nodes, _ = x.shape
        
        # Compute queries, keys, values
        q = self.query_proj(x).view(batch_size, num_nodes, self.num_heads, -1)
        k = self.key_proj(x).view(batch_size, num_nodes, self.num_heads, -1)
        v = self.value_proj(x).view(batch_size, num_nodes, self.num_heads, -1)
        
        # Transpose for attention: (batch, heads, nodes, hidden)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, num_nodes, -1)
        out = self.out_proj(out)
        
        # Average attention across heads for adjacency
        adj = attn.mean(dim=1)
        
        return out, adj
