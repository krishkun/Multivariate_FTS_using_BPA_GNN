"""
Alternative Evidence Theory Modules (Contribution 2)

This module implements alternative evidence combination methods:
1. Transferable Belief Model (TBM)
2. Pignistic Transformation Module
3. Evidential k-NN Module
4. Credal Classification Module

These modules provide different approaches to evidence combination and
uncertainty quantification, enabling comparative analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Literal


class TransferableBeliefModel(nn.Module):
    """
    Transferable Belief Model (TBM) Module
    
    TBM is a two-level model:
    1. Credal level: beliefs are entertained and combined
    2. Pignistic level: beliefs are transformed to probabilities for decision making
    
    Key features:
    - Open world assumption (allows for unknown hypotheses)
    - No normalization (conflict is kept as mass on empty set)
    - Decision making via pignistic transformation
    
    Args:
        num_hypotheses: Number of hypothesis classes
        open_world: Whether to use open world assumption
    """
    
    def __init__(
        self,
        num_hypotheses: int,
        open_world: bool = True,
    ):
        super(TransferableBeliefModel, self).__init__()
        
        self.num_hypotheses = num_hypotheses
        self.open_world = open_world
        
        # Learnable parameters for belief assignment
        self.belief_weights = nn.Parameter(
            torch.ones(num_hypotheses) / num_hypotheses
        )
        
        # Empty set mass (for open world)
        if open_world:
            self.empty_mass = nn.Parameter(torch.tensor(0.0))
    
    def conjunctive_combination(
        self,
        m1: torch.Tensor,
        m2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conjunctive rule of combination (TBM).
        
        m₁₂(A) = Σ m₁(B) * m₂(C) where B ∩ C = A
        
        Unlike Dempster's rule, no normalization is applied.
        
        Args:
            m1: First BPA of shape (..., num_hypotheses)
            m2: Second BPA of shape (..., num_hypotheses)
            
        Returns:
            Combined BPA
        """
        # Compute conjunctive combination
        combined = m1 * m2
        
        return combined
    
    def disjunctive_combination(
        self,
        m1: torch.Tensor,
        m2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Disjunctive rule of combination.
        
        m₁₂(A) = Σ m₁(B) * m₂(C) where B ∪ C = A
        
        Args:
            m1: First BPA
            m2: Second BPA
            
        Returns:
            Combined BPA
        """
        # For singleton hypotheses, this is equivalent to product
        combined = m1 * m2
        return combined
    
    def pignistic_transformation(
        self,
        bpa: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pignistic transformation for decision making.
        
        BetP(ω) = Σ (|A ∩ {ω}| / |A|) * m(A)
        
        For singleton hypotheses:
        BetP(ωᵢ) = m({ωᵢ}) + m(∅) / n
        
        Args:
            bpa: BPA tensor of shape (..., num_hypotheses)
            
        Returns:
            Pignistic probability distribution
        """
        if self.open_world:
            # Distribute empty set mass equally
            empty_mass = torch.sigmoid(self.empty_mass)
            pignistic = bpa + empty_mass / self.num_hypotheses
        else:
            pignistic = bpa
        
        # Normalize
        pignistic = pignistic / (pignistic.sum(dim=-1, keepdim=True) + 1e-8)
        
        return pignistic
    
    def compute_conflict(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """
        Compute conflict between two BPAs.
        
        Conflict k = Σ m₁(B) * m₂(C) where B ∩ C = ∅
        
        For singleton hypotheses, this is 1 - Σ m₁(ωᵢ) * m₂(ωᵢ)
        
        Args:
            m1: First BPA
            m2: Second BPA
            
        Returns:
            Conflict value
        """
        agreement = torch.sum(m1 * m2, dim=-1)
        return 1 - agreement
    
    def forward(
        self,
        m1: torch.Tensor,
        m2: torch.Tensor,
        method: str = 'conjunctive',
    ) -> Dict[str, torch.Tensor]:
        """
        Combine evidence using TBM.
        
        Args:
            m1: First BPA
            m2: Second BPA
            method: Combination method ('conjunctive' or 'disjunctive')
            
        Returns:
            Dictionary with combined BPA and pignistic probability
        """
        if method == 'conjunctive':
            combined = self.conjunctive_combination(m1, m2)
        else:
            combined = self.disjunctive_combination(m1, m2)
        
        pignistic = self.pignistic_transformation(combined)
        conflict = self.compute_conflict(m1, m2)
        
        return {
            'combined': combined,
            'pignistic': pignistic,
            'conflict': conflict,
        }


class PignisticTransformationModule(nn.Module):
    """
    Pignistic Transformation Module
    
    Implements the pignistic probability transformation and its inverse.
    Used for decision making under uncertainty.
    
    The module also implements:
    - Pignistic distance for comparing BPAs
    - Decision making based on pignistic probabilities
    
    Args:
        num_hypotheses: Number of hypothesis classes
    """
    
    def __init__(self, num_hypotheses: int):
        super(PignisticTransformationModule, self).__init__()
        
        self.num_hypotheses = num_hypotheses
    
    def transform(self, bpa: torch.Tensor) -> torch.Tensor:
        """
        Transform BPA to pignistic probability.
        
        BetP(ωᵢ) = Σ (|A ∩ {ωᵢ}| / |A|) * m(A)
        
        Args:
            bpa: BPA tensor
            
        Returns:
            Pignistic probability
        """
        # For singleton hypotheses
        pignistic = bpa / (bpa.sum(dim=-1, keepdim=True) + 1e-8)
        return pignistic
    
    def inverse_transform(
        self,
        pignistic: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inverse pignistic transformation.
        
        This is an ill-posed problem, but we can use the least commitment principle.
        
        Args:
            pignistic: Pignistic probability
            prior: Optional prior distribution
            
        Returns:
            BPA
        """
        # Simple approximation: assume singleton masses
        bpa = pignistic
        return bpa
    
    def pignistic_distance(
        self,
        betp1: torch.Tensor,
        betp2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pignistic distance between two probability distributions.
        
        d(BetP₁, BetP₂) = ||BetP₁ - BetP₂||₂
        
        Args:
            betp1: First pignistic probability
            betp2: Second pignistic probability
            
        Returns:
            Distance value
        """
        return torch.norm(betp1 - betp2, p=2, dim=-1)
    
    def make_decision(
        self,
        bpa: torch.Tensor,
        criterion: str = 'max_belief',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make decision based on BPA.
        
        Args:
            bpa: BPA tensor
            criterion: Decision criterion ('max_belief', 'max_plausibility', 'pignistic')
            
        Returns:
            Tuple of (decision, confidence)
        """
        if criterion == 'pignistic':
            prob = self.transform(bpa)
        else:
            prob = bpa
        
        decision = torch.argmax(prob, dim=-1)
        confidence = torch.max(prob, dim=-1)[0]
        
        return decision, confidence
    
    def forward(self, bpa: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply pignistic transformation.
        
        Args:
            bpa: BPA tensor
            
        Returns:
            Dictionary with pignistic probability and decision
        """
        pignistic = self.transform(bpa)
        decision, confidence = self.make_decision(bpa, criterion='pignistic')
        
        return {
            'pignistic': pignistic,
            'decision': decision,
            'confidence': confidence,
        }


class EvidentialKNNModule(nn.Module):
    """
    Evidential k-NN Module
    
    Implements k-nearest neighbors classifier with evidence theory.
    Each neighbor provides evidence for class membership.
    Evidence is combined using Dempster's rule.
    
    Args:
        k: Number of neighbors
        num_classes: Number of classes
        alpha: Parameter for mass assignment
    """
    
    def __init__(
        self,
        k: int = 5,
        num_classes: int = 10,
        alpha: float = 0.95,
    ):
        super(EvidentialKNNModule, self).__init__()
        
        self.k = k
        self.num_classes = num_classes
        self.alpha = alpha
        
        # Distance metric (learnable)
        self.distance_weight = nn.Parameter(torch.ones(1))
    
    def compute_distances(
        self,
        x: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distances between query and support samples.
        
        Args:
            x: Query samples of shape (batch, features)
            support: Support samples of shape (num_support, features)
            
        Returns:
            Distance matrix of shape (batch, num_support)
        """
        # Euclidean distance
        diff = x.unsqueeze(1) - support.unsqueeze(0)
        distances = torch.norm(diff, p=2, dim=-1)
        
        return distances * torch.abs(self.distance_weight)
    
    def distance_to_mass(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert distances to mass functions.
        
        m(ωᵢ) = α * exp(-γ * d²)
        m(Ω) = 1 - m(ωᵢ)
        
        where Ω is the universal set.
        
        Args:
            distances: Distances to k nearest neighbors
            labels: Labels of k nearest neighbors
            
        Returns:
            Mass functions of shape (batch, k, num_classes + 1)
        """
        batch_size = distances.shape[0]
        
        # Initialize masses
        masses = torch.zeros(batch_size, self.k, self.num_classes + 1, device=distances.device)
        
        # Compute class-specific masses
        gamma = 1.0  # Scale parameter
        class_mass = self.alpha * torch.exp(-gamma * distances ** 2)
        
        # Assign masses based on neighbor labels
        for i in range(self.k):
            for b in range(batch_size):
                label = labels[b, i].long()
                masses[b, i, label] = class_mass[b, i]
                masses[b, i, -1] = 1 - class_mass[b, i]  # Universal set
        
        return masses
    
    def combine_masses(self, masses: torch.Tensor) -> torch.Tensor:
        """
        Combine masses from all neighbors using Dempster's rule.
        
        Args:
            masses: Mass functions of shape (batch, k, num_classes + 1)
            
        Returns:
            Combined mass of shape (batch, num_classes + 1)
        """
        batch_size = masses.shape[0]
        
        # Initialize with first neighbor
        combined = masses[:, 0, :]
        
        # Combine with remaining neighbors
        for i in range(1, self.k):
            m1 = combined
            m2 = masses[:, i, :]
            
            # Dempster's combination
            # For focal sets: singleton classes and universal set
            new_combined = torch.zeros_like(combined)
            
            # Singleton masses
            for c in range(self.num_classes):
                new_combined[:, c] = m1[:, c] * m2[:, c] + \
                                     m1[:, c] * m2[:, -1] + \
                                     m1[:, -1] * m2[:, c]
            
            # Universal set mass
            new_combined[:, -1] = m1[:, -1] * m2[:, -1]
            
            # Normalize
            total = new_combined.sum(dim=-1, keepdim=True)
            combined = new_combined / (total + 1e-8)
        
        return combined
    
    def forward(
        self,
        x: torch.Tensor,
        support: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Classify using evidential k-NN.
        
        Args:
            x: Query samples
            support: Support samples
            support_labels: Labels of support samples
            
        Returns:
            Dictionary with combined mass and prediction
        """
        # Compute distances
        distances = self.compute_distances(x, support)
        
        # Get k nearest neighbors
        knn_distances, knn_indices = torch.topk(distances, self.k, largest=False)
        knn_labels = support_labels[knn_indices]
        
        # Convert to masses
        masses = self.distance_to_mass(knn_distances, knn_labels)
        
        # Combine masses
        combined = self.combine_masses(masses)
        
        # Make prediction
        class_masses = combined[:, :self.num_classes]
        prediction = torch.argmax(class_masses, dim=-1)
        
        return {
            'masses': combined,
            'prediction': prediction,
            'knn_distances': knn_distances,
            'knn_labels': knn_labels,
        }


class CredalClassificationModule(nn.Module):
    """
    Credal Classification Module
    
    Implements credal classification where:
    - A class is assigned if its belief is above a threshold
    - Otherwise, the sample is assigned to a set of classes (imprecise classification)
    
    This provides a more robust classification under uncertainty.
    
    Args:
        num_classes: Number of classes
        belief_threshold: Threshold for precise classification
        uncertainty_threshold: Threshold for rejecting classification
    """
    
    def __init__(
        self,
        num_classes: int,
        belief_threshold: float = 0.5,
        uncertainty_threshold: float = 0.3,
    ):
        super(CredalClassificationModule, self).__init__()
        
        self.num_classes = num_classes
        self.belief_threshold = belief_threshold
        self.uncertainty_threshold = uncertainty_threshold
    
    def compute_credal_set(
        self,
        bpa: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute credal set from BPA.
        
        A credal set is a set of probability distributions.
        Here we use the interval [Bel, Pl] for each class.
        
        Args:
            bpa: BPA tensor of shape (..., num_classes)
            
        Returns:
            Tuple of (lower probability, upper probability)
        """
        # For singleton sets, Bel = Pl = mass
        lower = bpa  # Belief
        upper = bpa  # Plausibility
        
        return lower, upper
    
    def classify(
        self,
        bpa: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform credal classification.
        
        Decision rules:
        1. If max(Bel) > threshold: precise classification
        2. If multiple classes have similar belief: imprecise classification
        3. If total uncertainty > threshold: reject
        
        Args:
            bpa: BPA tensor
            
        Returns:
            Tuple of (decision, decision_type, confidence)
            decision_type: 0=precise, 1=imprecise, 2=reject
        """
        batch_size = bpa.shape[0]
        
        # Compute belief and plausibility
        belief, plausibility = self.compute_credal_set(bpa)
        
        # Compute uncertainty
        uncertainty = plausibility - belief
        
        # Decision
        decision = torch.zeros(batch_size, dtype=torch.long, device=bpa.device)
        decision_type = torch.zeros(batch_size, dtype=torch.long, device=bpa.device)
        confidence = torch.zeros(batch_size, device=bpa.device)
        
        for i in range(batch_size):
            bel = belief[i]
            unc = uncertainty[i]
            
            max_bel, max_class = torch.max(bel, dim=0)
            
            if max_bel > self.belief_threshold:
                # Precise classification
                decision[i] = max_class
                decision_type[i] = 0
                confidence[i] = max_bel
            elif torch.sum(bel > self.belief_threshold * 0.5) > 1:
                # Imprecise classification (multiple classes)
                candidate_classes = (bel > self.belief_threshold * 0.5).nonzero()
                decision[i] = max_class  # Primary decision
                decision_type[i] = 1
                confidence[i] = max_bel
            else:
                # Reject
                decision[i] = max_class
                decision_type[i] = 2
                confidence[i] = max_bel
        
        return decision, decision_type, confidence
    
    def compute_credal_distance(
        self,
        bpa1: torch.Tensor,
        bpa2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distance between two credal sets.
        
        Uses the Hausdorff distance between the probability intervals.
        
        Args:
            bpa1: First BPA
            bpa2: Second BPA
            
        Returns:
            Distance value
        """
        lower1, upper1 = self.compute_credal_set(bpa1)
        lower2, upper2 = self.compute_credal_set(bpa2)
        
        # Hausdorff distance
        d1 = torch.max(torch.abs(lower1 - lower2), torch.abs(upper1 - upper2))
        d2 = torch.max(torch.abs(lower1 - upper2), torch.abs(upper1 - lower2))
        
        distance = torch.max(d1, d2)
        return distance
    
    def forward(self, bpa: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform credal classification.
        
        Args:
            bpa: BPA tensor
            
        Returns:
            Dictionary with classification results
        """
        lower, upper = self.compute_credal_set(bpa)
        decision, decision_type, confidence = self.classify(bpa)
        
        return {
            'decision': decision,
            'decision_type': decision_type,
            'confidence': confidence,
            'lower_probability': lower,
            'upper_probability': upper,
            'uncertainty': upper - lower,
        }


class EvidenceModuleFactory:
    """Factory for creating evidence modules."""
    
    @staticmethod
    def create(
        module_type: str,
        num_classes: int,
        **kwargs,
    ) -> nn.Module:
        """
        Create an evidence module.
        
        Args:
            module_type: Type of module ('tbm', 'pignistic', 'eknn', 'credal')
            num_classes: Number of classes
            **kwargs: Additional arguments
            
        Returns:
            Evidence module
        """
        if module_type == 'tbm':
            return TransferableBeliefModel(
                num_hypotheses=num_classes,
                open_world=kwargs.get('open_world', True),
            )
        elif module_type == 'pignistic':
            return PignisticTransformationModule(num_hypotheses=num_classes)
        elif module_type == 'eknn':
            return EvidentialKNNModule(
                k=kwargs.get('k', 5),
                num_classes=num_classes,
                alpha=kwargs.get('alpha', 0.95),
            )
        elif module_type == 'credal':
            return CredalClassificationModule(
                num_classes=num_classes,
                belief_threshold=kwargs.get('belief_threshold', 0.5),
                uncertainty_threshold=kwargs.get('uncertainty_threshold', 0.3),
            )
        else:
            raise ValueError(f"Unknown module type: {module_type}")
