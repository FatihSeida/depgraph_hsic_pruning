"""
DepGraph-HSIC Pruning Method v2 - Implementasi yang sesuai dengan BasePruningMethod.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import torch_pruning as tp
from pathlib import Path

from ultralytics import YOLO
from .base import BasePruningMethod
from .hsic_lasso import compute_channel_wise_hsic, solve_hsic_lasso
from .utils import collect_backbone_convs


class DepGraphHSICMethod2(BasePruningMethod):
    """
    Pruning berbasis penggabungan DepGraph (dependency) dan HSIC Lasso (informasi).
    Implementasi yang sesuai dengan BasePruningMethod abstract.
    """
    
    requires_reconfiguration: bool = True
    
    def __init__(
        self, 
        model: YOLO, 
        workdir: str | Path = "runs/pruning",
        alpha_range: Tuple[float, float] = (1e-3, 1.0),
        n_alphas: int = 10,
        sigma: Optional[float] = None,
        max_samples: int = 1000,
        seed: int = 42,
        ignored_layers: Optional[List[str]] = None,
        example_inputs: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__(model, workdir)
        
        # HSIC-Lasso parameters
        self.alpha_range = alpha_range
        self.n_alphas = n_alphas
        self.sigma = sigma
        self.max_samples = max_samples
        self.seed = seed
        self.ignored_layers = ignored_layers or []
        self.example_inputs = example_inputs or torch.randn(1, 3, 640, 640)
        
        # Internal state
        self.DG: Optional[tp.DependencyGraph] = None
        self.activations: Dict[int, List[torch.Tensor]] = {}
        self.labels: List[torch.Tensor] = []
        self.layer_shapes: Dict[int, Tuple[int, ...]] = {}
        self.num_activations: Dict[int, int] = {}
        self.layers: List[nn.Module] = []
        self.pruning_groups: List = []
        self.group_scores: Dict[int, torch.Tensor] = {}
        
        # Cache
        self._reset_cache()
    
    def _reset_cache(self) -> None:
        """Reset internal cache."""
        self.activations.clear()
        self.labels.clear()
        self.layer_shapes.clear()
        self.num_activations.clear()
        self.layers.clear()
        self.pruning_groups.clear()
        self.group_scores.clear()
        self.masks.clear()
    
    def analyze_model(self) -> None:
        """Inspect model structure and gather information for pruning."""
        self.logger.info("Analyzing model structure for DepGraph-HSIC pruning...")
        
        # Get backbone layers
        self.layers = []
        backbone_convs = collect_backbone_convs(self.model, num_modules=10)
        
        for parent, attr_name, bn in backbone_convs:
            conv_layer = getattr(parent, attr_name)
            if isinstance(conv_layer, nn.Conv2d) and conv_layer.out_channels > 1:
                self.layers.append(conv_layer)
        
        if not self.layers:
            raise RuntimeError("No convolutional layers found for pruning")
        
        self.logger.info(f"Found {len(self.layers)} convolutional layers for pruning")
        
        # Build dependency graph
        try:
            self.DG = tp.DependencyGraph()
            # Use the model instance directly like in depgraph_hsic.py
            example_inputs = self._inputs_tuple()
            self.DG.build_dependency(self.model, example_inputs)
            self.logger.info("Dependency graph built successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to build dependency graph: {e}")
        
        # Get pruning groups
        try:
            self.pruning_groups = list(self.DG.get_all_groups(root_module_types=[nn.Conv2d]))
            self.logger.info(f"Found {len(self.pruning_groups)} pruning groups")
        except Exception as e:
            raise RuntimeError(f"Failed to get pruning groups: {e}")
        
        if not self.pruning_groups:
            raise RuntimeError("No pruning groups found in dependency graph")
        
        # Register activation hooks
        self._register_activation_hooks()
    
    def _register_activation_hooks(self) -> None:
        """Register hooks to collect activations."""
        def make_hook(layer_idx: int):
            def hook(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
                target_shape = self.layer_shapes.get(layer_idx)
                if target_shape is None:
                    self.layer_shapes[layer_idx] = output.shape[2:]
                    processed = output.detach().cpu()
                else:
                    if output.shape[2:] != target_shape:
                        processed = torch.nn.functional.adaptive_avg_pool2d(output, target_shape).detach().cpu()
                    else:
                        processed = output.detach().cpu()
                
                self.activations.setdefault(layer_idx, []).append(processed)
                self.num_activations[layer_idx] = self.num_activations.get(layer_idx, 0) + 1
                
            return hook
        
        # Register hooks for each layer
        for idx, layer in enumerate(self.layers):
            layer.register_forward_hook(make_hook(idx))
    
    def generate_pruning_mask(self, ratio: float, dataloader=None) -> None:
        """Create a pruning mask with the given sparsity ratio."""
        if ratio <= 0 or ratio >= 1:
            raise ValueError(f"Pruning ratio must be between 0 and 1, got {ratio}")
        
        if dataloader is None:
            raise ValueError("Dataloader is required for HSIC computation. No fallback allowed.")
        
        self.logger.info(f"Generating pruning mask with ratio {ratio:.3f}")
        
        # Analyze model if not done yet
        if not self.layers:
            self.analyze_model()
        
        # Collect activations and labels
        self._collect_activations(dataloader)
        
        if len(self.labels) < 4:
            raise ValueError(f"Insufficient labels ({len(self.labels)} < 4) for HSIC computation. Need more data samples.")
        
        # Compute HSIC scores for each group
        self._compute_group_hsic_scores()
        
        if not self.group_scores:
            raise RuntimeError("Failed to compute HSIC scores for any group. Check model structure and data.")
        
        # Apply HSIC-Lasso pruning
        self._apply_hsic_lasso_pruning(ratio)
        
        self.logger.info("Pruning mask generated successfully")
    
    def _collect_activations(self, dataloader) -> None:
        """Collect activations and labels from dataloader."""
        self.logger.info("Collecting activations and labels...")
        
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        
        train_state = self.model.training
        self.model.eval()
        
        sample_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= self.max_samples:
                    break
                
                # Extract images and labels
                images = None
                labels = None
                
                if isinstance(batch, dict):
                    images = batch.get("img") or batch.get("images") or batch.get("inputs")
                    labels = batch.get("cls") or batch.get("label") or batch.get("labels")
                elif isinstance(batch, (list, tuple)):
                    if len(batch) > 0:
                        images = batch[0]
                    if len(batch) > 1:
                        labels = batch[1]
                else:
                    images = batch
                
                if images is None:
                    self.logger.warning("Skipping batch with no images")
                    continue
                
                # Ensure images are float32 and in correct range
                if images.dtype != torch.float32:
                    if images.dtype == torch.uint8:
                        images = images.float() / 255.0
                    else:
                        images = images.float()
                
                if images.max() > 1.0:
                    images = images / 255.0
                
                try:
                    # Forward pass
                    self.model(images.to(device))
                    
                    # Store labels
                    if labels is not None and isinstance(labels, torch.Tensor):
                        self.labels.append(labels.cpu())
                    else:
                        self.logger.warning("No valid labels found in batch")
                    
                    sample_count += 1
                    
                    if sample_count % 10 == 0:
                        self.logger.debug(f"Collected {sample_count} samples")
                        
                except Exception as e:
                    self.logger.error(f"Error during forward pass: {e}")
                    raise RuntimeError(f"Failed to process batch during activation collection: {e}")
        
        self.model.train(train_state)
        self.logger.info(f"Collected activations from {sample_count} samples")
        
        if sample_count == 0:
            raise RuntimeError("No samples were collected. Check dataloader format and data.")
        
        if len(self.labels) == 0:
            raise RuntimeError("No labels were collected. Check dataloader format and label extraction.")
    
    def _compute_group_hsic_scores(self) -> None:
        """Compute HSIC scores for each pruning group."""
        self.logger.info("Computing HSIC scores for pruning groups...")
        
        if not self.pruning_groups:
            # No fallback - raise error if no pruning groups found
            raise RuntimeError("No pruning groups found in dependency graph. Cannot proceed with HSIC computation.")
        
        # Process each pruning group
        for group_idx, group in enumerate(self.pruning_groups):
            group_activations = []
            group_channels = []
            
            for dep in group:
                target_module = self._extract_target_module(dep)
                if target_module is None:
                    continue
                
                if isinstance(target_module, nn.Conv2d):
                    layer_idx = None
                    for idx, layer in enumerate(self.layers):
                        if layer is target_module:
                            layer_idx = idx
                            break
                    
                    if layer_idx is not None and layer_idx in self.activations:
                        acts = torch.cat(self.activations[layer_idx], dim=0)
                        group_activations.append(acts)
                        group_channels.append(acts.size(1))
            
            if not group_activations:
                self.logger.warning(f"No activations found for group {group_idx}")
                continue
            
            # Concatenate activations from all layers in group
            combined_activations = torch.cat(group_activations, dim=1)
            
            # Prepare labels
            if len(self.labels) == 0:
                raise RuntimeError("No labels available for HSIC computation")
            
            labels_tensor = torch.cat(self.labels, dim=0)
            
            # Ensure same number of samples
            min_samples = min(combined_activations.size(0), labels_tensor.size(0))
            combined_activations = combined_activations[:min_samples]
            labels_tensor = labels_tensor[:min_samples]
            
            # Compute HSIC scores
            sigma = self.sigma or (1.0 / combined_activations.size(1) if combined_activations.size(1) > 0 else 1.0)
            scores = compute_channel_wise_hsic(combined_activations, labels_tensor, sigma)
            self.group_scores[group_idx] = scores
    
    def _extract_target_module(self, dep) -> Optional[nn.Module]:
        """Extract target module from dependency item."""
        # Handle different types of dependency items
        if hasattr(dep, 'module'):
            return dep.module
        elif hasattr(dep, 'target'):
            return dep.target
        elif hasattr(dep, 'layer'):
            return dep.layer
        elif hasattr(dep, 'dep'):
            # Recursively resolve nested dependencies
            return self._extract_target_module(dep.dep)
        else:
            return None
    
    def _apply_hsic_lasso_pruning(self, ratio: float) -> None:
        """Apply HSIC-Lasso pruning to determine which channels to keep."""
        # Combine all scores for Lasso regression
        all_scores = []
        group_indices = []
        channel_indices = []
        
        for group_idx, scores in self.group_scores.items():
            all_scores.extend(scores.tolist())
            group_indices.extend([group_idx] * len(scores))
            channel_indices.extend(list(range(len(scores))))
        
        if not all_scores:
            raise RuntimeError("No HSIC scores available for pruning")
        
        # Apply HSIC-Lasso
        kept_indices = solve_hsic_lasso(
            np.array(all_scores),
            ratio,
            alpha_range=self.alpha_range,
            n_alphas=self.n_alphas,
            seed=self.seed
        )
        
        # Create masks for each group
        self.masks.clear()
        
        # Group-based pruning only (no fallback to layer-based)
        for group_idx, group in enumerate(self.pruning_groups):
            group_kept_indices = [i for i, g_idx in enumerate(group_indices) if g_idx == group_idx]
            group_kept_indices = [i for i in group_kept_indices if i in kept_indices]
            
            # Create mask for this group
            total_channels = len(self.group_scores.get(group_idx, []))
            if total_channels > 0:
                mask = torch.zeros(total_channels, dtype=torch.bool)
                mask[group_kept_indices] = True
                self.masks.append(mask)
        
        if not self.masks:
            raise RuntimeError("No pruning masks were created. Check HSIC-Lasso computation.")
    
    def apply_pruning(self, rebuild: bool = False) -> None:
        """Apply the previously generated pruning mask to the model."""
        if not self.masks:
            raise RuntimeError("No pruning mask available. Call generate_pruning_mask first.")
        
        self.logger.info("Applying pruning masks to model...")
        
        # Note: This is a simplified implementation that only creates masks
        # For full model pruning, you would need to use torch-pruning's DG.prune_group() method
        # or implement custom module replacement logic
        
        self.logger.info(f"Created {len(self.masks)} pruning masks")
        self.logger.warning("Note: This implementation only creates pruning masks. "
                           "For actual model pruning, use torch-pruning's DG.prune_group() method "
                           "or implement custom module replacement logic.")
        
        # Log mask statistics
        for i, mask in enumerate(self.masks):
            kept_channels = mask.sum().item()
            total_channels = len(mask)
            self.logger.info(f"Mask {i}: {kept_channels}/{total_channels} channels kept "
                           f"({100 * kept_channels / total_channels:.1f}%)")
    
    def visualize_hsic_scores(self) -> None:
        """Visualize HSIC scores for each group."""
        if not self.group_scores:
            self.logger.warning("No HSIC scores available for visualization")
            return
        
        try:
            fig, axes = plt.subplots(len(self.group_scores), 1, figsize=(12, 4 * len(self.group_scores)))
            if len(self.group_scores) == 1:
                axes = [axes]
            
            for i, (group_idx, scores) in enumerate(self.group_scores.items()):
                axes[i].bar(range(len(scores)), scores.cpu().numpy())
                axes[i].set_title(f'HSIC Scores for Group {group_idx}')
                axes[i].set_xlabel('Channel Index')
                axes[i].set_ylabel('HSIC Score')
            
            plt.tight_layout()
            plt.savefig(self.workdir / "hsic_scores.png")
            plt.close(fig)
            self.logger.info("HSIC scores visualization saved")
            
        except Exception as e:
            self.logger.warning(f"Failed to create HSIC scores visualization: {e}") 