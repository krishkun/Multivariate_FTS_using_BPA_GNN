"""
Explainability Module for Fuzzy BPA EGNN

This module provides tools for model interpretability:
1. SHAP-based feature importance
2. Belief visualization
3. Uncertainty metrics
4. Evidence decomposition
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any
import warnings


class BeliefVisualizer:
    """
    Visualizer for belief functions and evidence.
    
    Provides visualization methods for:
    - Basic Probability Assignments
    - Belief and Plausibility intervals
    - Evidence combination
    - Uncertainty quantification
    """
    
    def __init__(self, num_hypotheses: int):
        self.num_hypotheses = num_hypotheses
    
    def plot_bpa(
        self,
        bpa: np.ndarray,
        title: str = "Basic Probability Assignment",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot BPA as a bar chart.
        
        Args:
            bpa: BPA array of shape (num_hypotheses,) or (batch, num_hypotheses)
            title: Plot title
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib figure
        """
        if bpa.ndim == 2:
            bpa = bpa.mean(axis=0)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()
        
        x = np.arange(self.num_hypotheses)
        ax.bar(x, bpa, color='steelblue', alpha=0.7)
        ax.set_xlabel('Hypothesis')
        ax.set_ylabel('Mass')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f'H{i}' for i in range(self.num_hypotheses)])
        
        return fig
    
    def plot_belief_plausibility(
        self,
        belief: np.ndarray,
        plausibility: np.ndarray,
        title: str = "Belief and Plausibility",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot belief and plausibility intervals.
        
        Args:
            belief: Belief values
            plausibility: Plausibility values
            title: Plot title
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()
        
        x = np.arange(self.num_hypotheses)
        width = 0.35
        
        ax.bar(x - width/2, belief, width, label='Belief', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, plausibility, width, label='Plausibility', color='coral', alpha=0.7)
        
        ax.set_xlabel('Hypothesis')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.legend()
        
        return fig
    
    def plot_uncertainty(
        self,
        uncertainty: np.ndarray,
        title: str = "Uncertainty Distribution",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot uncertainty distribution.
        
        Args:
            uncertainty: Uncertainty values
            title: Plot title
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()
        
        ax.hist(uncertainty.flatten(), bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        
        return fig
    
    def plot_evidence_combination(
        self,
        evidence_list: List[np.ndarray],
        labels: List[str],
        title: str = "Evidence Combination",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot multiple evidence sources and their combination.
        
        Args:
            evidence_list: List of BPA arrays
            labels: Labels for each evidence source
            title: Plot title
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.get_figure()
        
        x = np.arange(self.num_hypotheses)
        width = 0.8 / len(evidence_list)
        
        for i, (evidence, label) in enumerate(zip(evidence_list, labels)):
            offset = (i - len(evidence_list)/2 + 0.5) * width
            ax.bar(x + offset, evidence, width, label=label, alpha=0.7)
        
        ax.set_xlabel('Hypothesis')
        ax.set_ylabel('Mass')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.legend()
        
        return fig


class UncertaintyMetrics:
    """
    Compute uncertainty metrics from BPA.
    
    Metrics include:
    - Total uncertainty (non-specificity)
    - Conflict measure
    - Entropy-based measures
    - Ambiguity measure
    """
    
    def __init__(self, num_hypotheses: int):
        self.num_hypotheses = num_hypotheses
    
    def total_uncertainty(self, bpa: torch.Tensor) -> torch.Tensor:
        """
        Compute total uncertainty from BPA.
        
        Total uncertainty = Plausibility - Belief
        
        Args:
            bpa: BPA tensor
            
        Returns:
            Total uncertainty
        """
        # For singleton sets, this is 0
        # For general case, compute from focal sets
        return torch.zeros(bpa.shape[0], device=bpa.device)
    
    def non_specificity(self, bpa: torch.Tensor) -> torch.Tensor:
        """
        Compute non-specificity (Yager's measure).
        
        N(m) = Σ m(A) * log2(|A|)
        
        For singleton hypotheses, this is 0.
        
        Args:
            bpa: BPA tensor
            
        Returns:
            Non-specificity measure
        """
        # For singleton sets, non-specificity is 0
        return torch.zeros(bpa.shape[0], device=bpa.device)
    
    def conflict_measure(self, bpa: torch.Tensor) -> torch.Tensor:
        """
        Compute conflict measure.
        
        Conflict is related to how much mass is on disjoint sets.
        
        Args:
            bpa: BPA tensor
            
        Returns:
            Conflict measure
        """
        # Compute entropy-like measure
        entropy = -torch.sum(bpa * torch.log(bpa + 1e-8), dim=-1)
        max_entropy = np.log(self.num_hypotheses)
        
        return entropy / max_entropy
    
    def ambiguity_measure(self, bpa: torch.Tensor) -> torch.Tensor:
        """
        Compute ambiguity measure.
        
        Ambiguity = 1 - max(belief)
        
        Args:
            bpa: BPA tensor
            
        Returns:
            Ambiguity measure
        """
        return 1 - torch.max(bpa, dim=-1)[0]
    
    def compute_all_metrics(self, bpa: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all uncertainty metrics.
        
        Args:
            bpa: BPA tensor
            
        Returns:
            Dictionary of metrics
        """
        return {
            'total_uncertainty': self.total_uncertainty(bpa),
            'non_specificity': self.non_specificity(bpa),
            'conflict': self.conflict_measure(bpa),
            'ambiguity': self.ambiguity_measure(bpa),
        }


class SHAPExplainer:
    """
    SHAP-based explainer for Fuzzy BPA EGNN.
    
    Provides feature importance and model explanations using
    SHAP (SHapley Additive exPlanations) values.
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: The model to explain
            background_data: Background data for SHAP
        """
        self.model = model
        self.background_data = background_data
        
        # Try to import SHAP
        try:
            import shap
            self.shap = shap
            self.explainer = None
        except ImportError:
            warnings.warn("SHAP not installed. Install with: pip install shap")
            self.shap = None
    
    def explain(
        self,
        x: torch.Tensor,
        nsamples: int = 100,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for input.
        
        Args:
            x: Input tensor
            nsamples: Number of samples for SHAP
            
        Returns:
            Dictionary with SHAP values and explanations
        """
        if self.shap is None:
            return {'error': 'SHAP not installed'}
        
        # Create a wrapper function for the model
        def model_fn(x_np):
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x_np).to(next(self.model.parameters()).device)
                output = self.model(x_tensor)
            return output.cpu().numpy()
        
        # Create explainer
        self.explainer = self.shap.KernelExplainer(
            model_fn,
            self.background_data.cpu().numpy()
        )
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(
            x.cpu().numpy(),
            nsamples=nsamples
        )
        
        return {
            'shap_values': shap_values,
            'base_values': self.explainer.expected_value,
        }
    
    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Plot feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values
            feature_names: Optional feature names
            
        Returns:
            Matplotlib figure
        """
        if self.shap is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'SHAP not installed', ha='center', va='center')
            return fig
        
        # Compute mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(mean_shap))]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, mean_shap, color='steelblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('Feature Importance')
        
        return fig


class EvidenceDecomposer:
    """
    Decompose predictions into evidence contributions.
    
    Analyzes how different evidence sources contribute to the final prediction.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def decompose(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose prediction into evidence contributions.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with evidence contributions
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get full prediction
            full_output = self.model(x)
            
            # Get evidence if available
            if hasattr(self.model, 'predict_with_uncertainty'):
                result = self.model.predict_with_uncertainty(x)
                evidence = result.get('evidence', None)
            else:
                evidence = None
        
        return {
            'prediction': full_output,
            'evidence': evidence,
        }
    
    def analyze_temporal_contribution(
        self,
        x: torch.Tensor,
        time_steps: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze contribution of different time steps.
        
        Args:
            x: Input tensor
            time_steps: Time steps to analyze
            
        Returns:
            Dictionary with temporal contributions
        """
        if time_steps is None:
            time_steps = list(range(x.shape[1]))
        
        contributions = {}
        
        for t in time_steps:
            # Mask out time step t
            x_masked = x.clone()
            x_masked[:, t, :] = 0
            
            # Get prediction without time step t
            with torch.no_grad():
                output_masked = self.model(x_masked)
            
            # Contribution is difference
            with torch.no_grad():
                full_output = self.model(x)
            
            contributions[f't_{t}'] = full_output - output_masked
        
        return contributions
    
    def analyze_variable_contribution(
        self,
        x: torch.Tensor,
        variable_indices: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze contribution of different variables.
        
        Args:
            x: Input tensor
            variable_indices: Variable indices to analyze
            
        Returns:
            Dictionary with variable contributions
        """
        if variable_indices is None:
            variable_indices = list(range(x.shape[2]))
        
        contributions = {}
        
        for v in variable_indices:
            # Mask out variable v
            x_masked = x.clone()
            x_masked[:, :, v] = 0
            
            # Get prediction without variable v
            with torch.no_grad():
                output_masked = self.model(x_masked)
            
            # Contribution is difference
            with torch.no_grad():
                full_output = self.model(x)
            
            contributions[f'var_{v}'] = full_output - output_masked
        
        return contributions


class ModelExplainer:
    """
    Main explainer class combining all explainability tools.
    
    Provides a unified interface for:
    - Belief visualization
    - Uncertainty metrics
    - SHAP explanations
    - Evidence decomposition
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_hypotheses: int = 8,
        background_data: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.num_hypotheses = num_hypotheses
        
        # Initialize components
        self.visualizer = BeliefVisualizer(num_hypotheses)
        self.uncertainty_metrics = UncertaintyMetrics(num_hypotheses)
        self.decomposer = EvidenceDecomposer(model)
        
        if background_data is not None:
            self.shap_explainer = SHAPExplainer(model, background_data)
        else:
            self.shap_explainer = None
    
    def explain_prediction(
        self,
        x: torch.Tensor,
        compute_shap: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction.
        
        Args:
            x: Input tensor
            compute_shap: Whether to compute SHAP values
            
        Returns:
            Dictionary with all explanations
        """
        explanation = {}
        
        # Get prediction with uncertainty
        if hasattr(self.model, 'predict_with_uncertainty'):
            result = self.model.predict_with_uncertainty(x)
            explanation['prediction'] = result['prediction']
            explanation['evidence'] = result['evidence']
            explanation['uncertainty'] = result['uncertainty']
        else:
            with torch.no_grad():
                explanation['prediction'] = self.model(x)
        
        # Compute uncertainty metrics
        if explanation.get('evidence') is not None:
            evidence = explanation['evidence']
            if isinstance(evidence, dict):
                bpa = evidence.get('fused_evidence', None)
                if bpa is not None:
                    bpa_flat = bpa.mean(dim=-1)  # Average over F dimension
                    explanation['uncertainty_metrics'] = self.uncertainty_metrics.compute_all_metrics(bpa_flat)
        
        # Decompose evidence
        explanation['decomposition'] = self.decomposer.decompose(x)
        
        # Compute SHAP if requested
        if compute_shap and self.shap_explainer is not None:
            explanation['shap'] = self.shap_explainer.explain(x)
        
        return explanation
    
    def visualize(
        self,
        explanation: Dict[str, Any],
        figsize: Tuple[int, int] = (15, 10),
    ) -> plt.Figure:
        """
        Create visualization of explanation.
        
        Args:
            explanation: Explanation dictionary
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot BPA if available
        if explanation.get('evidence') is not None:
            evidence = explanation['evidence']
            if isinstance(evidence, dict):
                bpa = evidence.get('fused_evidence', None)
                if bpa is not None:
                    bpa_np = bpa.mean(dim=-1).cpu().numpy()
                    if bpa_np.ndim == 3:
                        bpa_np = bpa_np.mean(axis=0)
                    self.visualizer.plot_bpa(bpa_np, ax=axes[0, 0], title="Fused Evidence")
        
        # Plot uncertainty if available
        if explanation.get('uncertainty') is not None:
            uncertainty = explanation['uncertainty'].cpu().numpy()
            self.visualizer.plot_uncertainty(uncertainty, ax=axes[0, 1])
        
        # Plot prediction
        if explanation.get('prediction') is not None:
            pred = explanation['prediction'].cpu().numpy()
            if pred.ndim == 3:
                pred = pred[0]  # Take first batch
            axes[1, 0].plot(pred)
            axes[1, 0].set_title('Prediction')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Value')
        
        # Plot uncertainty metrics
        if explanation.get('uncertainty_metrics') is not None:
            metrics = explanation['uncertainty_metrics']
            metric_names = list(metrics.keys())
            metric_values = [metrics[m].mean().item() for m in metric_names]
            axes[1, 1].bar(metric_names, metric_values, color='steelblue', alpha=0.7)
            axes[1, 1].set_title('Uncertainty Metrics')
            axes[1, 1].set_ylabel('Value')
        
        plt.tight_layout()
        return fig
