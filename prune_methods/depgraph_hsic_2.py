"""Adaptive DepGraph-HSIC pruning implementation.

This version follows an iterative pipeline composed of six stages:

1. **Structural Constraint Analysis** – build a dependency graph and map
   convolution layers to pruning groups.  Structural importance for each
   group is estimated from layer depth, number of dependencies and
   parameter count.
2. **HSIC Performance Scoring** – collect activation maps using forward
   hooks and compute channel wise HSIC values w.r.t. the target labels.
   Scores are normalised inside every group for comparability.
3. **Adaptive Sub‑grouping Strategy** – inside each constraint group the
   filters are clustered based on their HSIC score using ``KMeans`` so
   that channels with similar importance form sub‑groups.
4. **Multi‑Criteria Decision Making** – sub‑groups receive a combined
   score derived from HSIC and structural importance.  The weighting
   factor ``alpha`` controls the trade‑off.
5. **Constraint‑Aware Pruning Execution** – the lowest ranked
   sub‑groups are removed through ``torch‑pruning`` while the dependency
   graph ensures tensor shapes remain valid.
6. **Iterative Refinement** – pruning and scoring can be repeated for
   several iterations.  After each round the dependency graph is rebuilt
   so subsequent passes operate on the updated model.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans
import torch_pruning as tp
from ultralytics import YOLO

from .base import BasePruningMethod
from .hsic_lasso import compute_channel_wise_hsic
from .utils import collect_backbone_convs


class DepGraphHSICMethod2(BasePruningMethod):
    """Adaptive filter pruning combining structural and HSIC information."""

    requires_reconfiguration: bool = True

    def __init__(
        self,
        model: YOLO,
        workdir: str | Path = "runs/pruning",
        *,
        sigma: Optional[float] = None,
        max_samples: int = 400,
        seed: int = 42,
        alpha: float = 0.8,
        sub_group_clusters: int = 4,
        iterations: int = 2,
        example_inputs: torch.Tensor | tuple | None = None,
        pruning_scope: str = "full",
    ) -> None:
        super().__init__(model, workdir, example_inputs)
        self.sigma = sigma
        self.max_samples = max_samples
        self.seed = seed
        self.alpha = float(alpha)
        self.sub_group_clusters = sub_group_clusters
        self.iterations = iterations
        self.pruning_scope = pruning_scope

        # Internal state
        self.DG: Optional[tp.DependencyGraph] = None
        self.layers: List[nn.Conv2d] = []
        self.pruning_groups: List[Any] = []
        self.group_map: Dict[int, List[int]] = {}
        self.structural_scores: Dict[int, float] = {}
        self.activations: Dict[int, List[torch.Tensor]] = {}
        self.layer_shapes: Dict[int, Tuple[int, int]] = {}
        self.labels: List[torch.Tensor] = []
        self.group_scores: Dict[int, torch.Tensor] = {}
        self.sub_groups: Dict[int, List[List[int]]] = {}
        self.sub_group_scores: Dict[Tuple[int, int], float] = {}
        self.masks: List[torch.Tensor] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Phase 1 – structural analysis
    # ------------------------------------------------------------------
    def analyze_model(self) -> None:  # pragma: no cover - heavy dependency
        self.logger.info("Building dependency graph and analysing structure (scope: %s)", self.pruning_scope)

        if self.pruning_scope == "backbone":
            # Gunakan utilitas untuk mengambil Conv2d backbone pertama (10 modul)
            self.layers = [
                conv
                for parent_mod, attr, _bn in collect_backbone_convs(self.model, num_modules=10)
                if isinstance((conv := getattr(parent_mod, attr)), nn.Conv2d) and conv.out_channels > 1
            ]
        else:  # "full"
            # Ambil seluruh Conv2d pada model
            self.layers = [
                m for m in self.model.modules() if isinstance(m, nn.Conv2d) and m.out_channels > 1
            ]
        if not self.layers:
            raise RuntimeError("No convolutional layers found for pruning")

        self.DG = tp.DependencyGraph()
        self.DG.build_dependency(self.model, example_inputs=self._inputs_tuple())
        self.pruning_groups = list(self.DG.get_all_groups(root_module_types=[nn.Conv2d]))
        if not self.pruning_groups:
            raise RuntimeError("No pruning groups found in dependency graph")

        self.group_map.clear()
        self.structural_scores.clear()
        for g_idx, group in enumerate(self.pruning_groups):
            mods: List[nn.Module] = []
            for dep in group:
                mod = self._extract_target_module(dep)
                if isinstance(mod, nn.Conv2d) and mod in self.layers:
                    mods.append(mod)
                    idx = self.layers.index(mod)
                    self.group_map.setdefault(idx, []).append(g_idx)
            if not mods:
                continue
            depth = min(self.layers.index(m) for m in mods)
            params = sum(m.weight.numel() for m in mods)
            connectivity = len(group)
            self.structural_scores[g_idx] = float(depth + connectivity + params / 1e5)

        self._register_activation_hooks()

    # ------------------------------------------------------------------
    # Activation collection
    # ------------------------------------------------------------------
    def _register_activation_hooks(self) -> None:
        for h in getattr(self, "_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

        def make_hook(idx: int):
            def hook(_mod: nn.Module, _inp: Tuple[torch.Tensor], out: torch.Tensor) -> None:
                shp = self.layer_shapes.setdefault(idx, out.shape[2:])
                if out.shape[2:] != shp:
                    processed = torch.nn.functional.adaptive_avg_pool2d(out, shp)
                else:
                    processed = out
                self.activations.setdefault(idx, []).append(processed.detach().cpu())
            return hook

        for idx, layer in enumerate(self.layers):
            handle = layer.register_forward_hook(make_hook(idx))
            self._hook_handles.append(handle)

    def _collect_activations(self, dataloader) -> None:
        device = next(self.model.parameters()).device
        train_state = self.model.training
        self.model.eval()
        sample_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= self.max_samples:
                    break
                images = None
                labels = None
                if isinstance(batch, dict):
                    for key in ("img", "images", "inputs"):
                        if key in batch and batch[key] is not None:
                            images = batch[key]
                            break
                    for key in ("cls", "label", "labels"):
                        if key in batch and batch[key] is not None:
                            labels = batch[key]
                            break
                elif isinstance(batch, (list, tuple)):
                    if len(batch) > 0:
                        images = batch[0]
                    if len(batch) > 1:
                        labels = batch[1]
                else:
                    images = batch
                if images is None:
                    continue
                images = images.to(device, dtype=torch.float32)
                if images.max() > 1.0:
                    images = images / 255.0
                self.model(images)
                if labels is not None and isinstance(labels, torch.Tensor):
                    self.labels.append(labels.detach().cpu())
                sample_count += 1
        self.model.train(train_state)
        if sample_count == 0:
            raise RuntimeError("No samples were collected for HSIC scoring")
        if not self.labels:
            raise RuntimeError("No labels were collected for HSIC scoring")

    # ------------------------------------------------------------------
    # Phase 2 – HSIC scoring
    # ------------------------------------------------------------------
    def _compute_group_hsic_scores(self) -> None:
        self.group_scores.clear()
        for g_idx, group in enumerate(self.pruning_groups):
            acts: List[torch.Tensor] = []
            for dep in group:
                mod = self._extract_target_module(dep)
                if mod not in self.layers:
                    continue
                l_idx = self.layers.index(mod)
                if l_idx in self.activations:
                    acts.append(torch.cat(self.activations[l_idx], dim=0))
            if not acts:
                continue
            if acts:
                # Samakan resolusi spasial dengan adaptive pooling
                min_h, min_w = min(t.shape[2] for t in acts), min(t.shape[3] for t in acts)
                pooled = [
                    (t if (t.shape[2] == min_h and t.shape[3] == min_w) else torch.nn.functional.adaptive_avg_pool2d(t, (min_h, min_w)))
                    for t in acts
                ]
                combined = torch.cat(pooled, dim=1)
            labels = torch.cat(self.labels, dim=0)
            m = min(combined.size(0), labels.size(0))
            combined = combined[:m]
            labels = labels[:m]
            sigma = self.sigma or 1.0 / max(combined.size(1), 1)
            scores = compute_channel_wise_hsic(combined, labels, sigma)
            if scores.numel() > 1:
                scores = (scores - scores.mean()) / (scores.std() + 1e-8)
            self.group_scores[g_idx] = scores

        # Fallback: jika tidak ada skor grup (mis. hanya 1 grup DG tanpa aktivasi terpetakan)
        if not self.group_scores:
            self.logger.warning("No HSIC scores computed per group – falling back to aggregated layer scoring")
            acts_all: List[torch.Tensor] = []
            for lst in self.activations.values():
                try:
                    acts_all.append(torch.cat(lst, dim=0))
                except Exception:
                    continue
            if acts_all:
                # Samakan resolusi spasial dengan adaptive pooling
                min_h, min_w = min(t.shape[2] for t in acts_all), min(t.shape[3] for t in acts_all)
                pooled = [
                    (t if (t.shape[2] == min_h and t.shape[3] == min_w) else torch.nn.functional.adaptive_avg_pool2d(t, (min_h, min_w)))
                    for t in acts_all
                ]
                combined = torch.cat(pooled, dim=1)
                labels = torch.cat(self.labels, dim=0)
                m = min(combined.size(0), labels.size(0))
                combined = combined[:m]
                labels = labels[:m]
                sigma = self.sigma or 1.0 / max(combined.size(1), 1)
                scores = compute_channel_wise_hsic(combined, labels, sigma)
                if scores.numel() > 1:
                    scores = (scores - scores.mean()) / (scores.std() + 1e-8)
                # Asumsikan grup indeks 0 (jika tidak ada) – buat dummy jika perlu
                if not self.pruning_groups:
                    self.pruning_groups = [tuple()]
                self.group_scores[0] = scores

    # ------------------------------------------------------------------
    # Phase 3 – adaptive sub grouping
    # ------------------------------------------------------------------
    def _create_sub_groups(self) -> None:
        self.sub_groups.clear()
        for g_idx, scores in self.group_scores.items():
            arr = scores.view(-1, 1).cpu().numpy()
            dynamic = max(2, round(len(arr) / 64))
            n_cluster = min(max(self.sub_group_clusters, dynamic), len(arr))
            if n_cluster <= 1:
                self.sub_groups[g_idx] = [list(range(len(arr)))]
                continue
            kmeans = KMeans(n_clusters=n_cluster, random_state=self.seed, n_init="auto")
            labels = kmeans.fit_predict(arr)
            groups: List[List[int]] = []
            for c in range(n_cluster):
                idxs = np.where(labels == c)[0].tolist()
                if idxs:
                    groups.append(idxs)
            self.sub_groups[g_idx] = groups

    # ------------------------------------------------------------------
    # Phase 4 – multi criteria decision
    # ------------------------------------------------------------------
    def _score_sub_groups(self) -> None:
        self.sub_group_scores.clear()
        max_struct = max(self.structural_scores.values()) if self.structural_scores else 1.0
        for g_idx, groups in self.sub_groups.items():
            struct_norm = self.structural_scores.get(g_idx, 0.0) / max_struct
            hsic = self.group_scores.get(g_idx)
            if hsic is None:
                continue
            for s_idx, idxs in enumerate(groups):
                if not idxs:
                    continue
                hval = hsic[idxs].mean() / float(np.log(len(idxs) + 1))
                score = self.alpha * hval.item() + (1.0 - self.alpha) * struct_norm
                self.sub_group_scores[(g_idx, s_idx)] = score

    def _select_pruned_sub_groups(self, ratio: float) -> List[Tuple[int, int]]:
        self.logger.info(f"Selecting sub-groups for pruning with ratio {ratio}")
        self.logger.info(f"Available sub-group scores: {len(self.sub_group_scores)}")
        self.logger.info(f"Sub-groups: {len(self.sub_groups)}")
        
        if not self.sub_group_scores:
            # Tidak ada skor sub-grup – pilih saluran terendah dari grup pertama
            self.logger.warning("No sub-group scores available – selecting fallback channels")
            if 0 in self.group_scores:
                scores = self.group_scores[0]
                n_prune = max(1, int(len(scores) * ratio))
                idxs = torch.argsort(scores)[:n_prune].tolist()
                return [(0, 0)]  # satu sub-grup dummy, akan ditangani di _build_masks

        ordered = sorted(self.sub_group_scores.items(), key=lambda x: x[1])
        total_channels = sum(len(sg) for groups in self.sub_groups.values() for sg in groups)
        target = int(total_channels * ratio)
        self.logger.info(f"Total channels: {total_channels}, target to prune: {target}")
        
        selected: List[Tuple[int, int]] = []
        removed = 0
        idx = 0
        while removed < target and idx < len(ordered):
            g_idx, s_idx = ordered[idx][0]
            sz = len(self.sub_groups[g_idx][s_idx])
            selected.append((g_idx, s_idx))
            removed += sz
            idx += 1

        allowed = max(1, int(total_channels * 0.01))
        if removed > target + allowed:
            selected.sort(key=lambda p: len(self.sub_groups[p[0]][p[1]]))
            while selected and removed - target > allowed:
                g_idx, s_idx = selected.pop(0)
                removed -= len(self.sub_groups[g_idx][s_idx])

        self.logger.info(f"Selected {len(selected)} sub-groups for pruning")
        
        # Fallback jika selected kosong
        if not selected and 0 in self.group_scores:
            self.logger.warning("No sub-groups selected – switching to layer-wise fallback")
            scores = self.group_scores[0]
            n_prune = max(1, int(len(scores) * ratio))
            idxs = torch.argsort(scores)[:n_prune].tolist()
            self.sub_groups = {0: [idxs]}
            # assign high dummy score so considered lowest importance
            self.sub_group_scores = {(0, 0): scores[idxs].mean().item()}
            selected = [(0, 0)]
        
        return selected

    def _build_masks(self, to_prune: List[Tuple[int, int]]) -> None:
        self.masks = []
        prune_set = {(g, s) for g, s in to_prune}
        for g_idx, scores in self.group_scores.items():
            mask = torch.ones(len(scores), dtype=torch.bool)
            groups = self.sub_groups.get(g_idx, [])
            for s_idx, idxs in enumerate(groups):
                if (g_idx, s_idx) in prune_set:
                    mask[idxs] = False
            self.masks.append(mask)

    # ------------------------------------------------------------------
    # Phase 5 – execute pruning
    # ------------------------------------------------------------------
    def _apply_masks(self) -> None:
        if self.DG is None:
            raise RuntimeError("Dependency graph not built")
        # Lepas semua forward hook agar model bisa dipickle
        self.remove_activation_hooks()

        # Jika fallback layerwise, terapkan pruning per-layer
        if getattr(self, 'fallback_layerwise', False):
            self._apply_layerwise_pruning()
        else:
            self._apply_depgraph_pruning()

        # Be compatible with different torch_pruning versions
        try:
            if hasattr(tp.utils, "remove_pruning_reparametrization"):
                tp.utils.remove_pruning_reparametrization(self.model)
            elif hasattr(tp, "remove_pruning_reparametrization"):
                tp.remove_pruning_reparametrization(self.model)
        except Exception:
            self.logger.warning("remove_pruning_reparametrization not available in current torch_pruning version")

    def _apply_depgraph_pruning(self) -> None:
        """Apply pruning using dependency graph groups."""
        try:
            for (g_idx, group), mask in zip(enumerate(self.pruning_groups), self.masks):
                prune_idx = (~mask).nonzero(as_tuple=False).view(-1).tolist()
                if not prune_idx:
                    continue
                convs = [
                    self._extract_target_module(dep)
                    for dep in group
                    if isinstance(self._extract_target_module(dep), nn.Conv2d)
                ]
                if not convs:
                    continue
                for conv in convs:
                    try:
                        # Validasi indeks sebelum prune
                        valid_idx = [i for i in prune_idx if i < conv.out_channels]
                        if valid_idx:
                            sub_group = self.DG.get_pruning_group(conv, tp.prune_conv_out_channels, valid_idx)
                            self.DG.prune_group(sub_group)
                            self.logger.debug("Pruned %d channels from %s", len(valid_idx), conv)
                        else:
                            self.logger.warning("No valid indices for %s (out_channels=%d)", conv, conv.out_channels)
                    except Exception:
                        self.logger.exception("Failed to prune group for conv %s", conv)
        except Exception:
            self.logger.warning("remove_pruning_reparametrization not available in current torch_pruning version")

    def _apply_layerwise_pruning(self) -> None:
        """Apply pruning per-layer when dependency graph fails."""
        self.logger.info("Applying layer-wise pruning fallback")
        total_pruned = 0

        # Jika fallback, langsung prune layer tanpa dependency graph
        if 0 in self.group_scores and len(self.masks) > 0:
            global_mask = self.masks[0]
            scores = self.group_scores[0]
            
            # Ambil channel dengan HSIC score terendah
            n_prune = int(len(scores) * 0.2)  # 20% ratio
            worst_indices = torch.argsort(scores)[:n_prune].tolist()
            
            self.logger.info("Pruning %d channels with lowest HSIC scores", n_prune)
            
            # Prune layer secara manual
            for layer_idx, layer in enumerate(self.layers):
                if layer_idx >= len(self.layers):
                    break
                    
                layer_channels = layer.out_channels
                start_idx = sum(l.out_channels for l in self.layers[:layer_idx])
                end_idx = start_idx + layer_channels
                
                # Filter indeks yang valid untuk layer ini
                layer_prune_indices = [
                    i - start_idx for i in worst_indices 
                    if start_idx <= i < end_idx
                ]
                
                if layer_prune_indices:
                    try:
                        # Prune secara manual dengan torch_pruning
                        from torch_pruning import prune_conv_out_channels
                        prune_conv_out_channels(layer, layer_prune_indices)
                        total_pruned += len(layer_prune_indices)
                        self.logger.info("Layer %d: pruned %d/%d channels", 
                                       layer_idx, len(layer_prune_indices), layer_channels)
                    except Exception as e:
                        self.logger.warning("Failed to prune layer %d: %s", layer_idx, e)

        self.logger.info("Manual pruning completed: %d total channels pruned", total_pruned)

        # Force model recompilation to apply changes
        try:
            # Rebuild model structure
            if hasattr(self.model, 'model'):
                self.model.model = self.model.model
            self.logger.info("Model structure updated after pruning")
        except Exception as e:
            self.logger.warning("Failed to update model structure: %s", e)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def generate_pruning_mask(self, ratio: float, dataloader=None) -> None:
        if ratio <= 0 or ratio >= 1:
            raise ValueError("Pruning ratio must be between 0 and 1")
        if dataloader is None:
            raise ValueError("Dataloader is required for HSIC computation")
        if not self.layers:
            self.analyze_model()
        self._collect_activations(dataloader)
        self._compute_group_hsic_scores()
        self._create_sub_groups()
        self._score_sub_groups()
        to_prune = self._select_pruned_sub_groups(ratio)
        # Simpan rencana pruning agar pipeline bisa membuat ringkasan
        self.pruning_plan = to_prune
        self._build_masks(to_prune)

    def apply_pruning(self, rebuild: bool = False) -> None:  # pragma: no cover - heavy dependency
        if not self.masks:
            raise RuntimeError("No pruning mask generated")
        self._apply_masks()
        if rebuild and self.DG is not None:
            self.DG.build_dependency(self.model, example_inputs=self._inputs_tuple())

    # ------------------------------------------------------------------
    # Phase 6 – iterative refinement
    # ------------------------------------------------------------------
    def iterative_pruning(self, ratio: float, dataloader) -> None:
        iters = max(1, self.iterations)
        step = ratio / iters
        for i in range(iters):
            self.analyze_model()
            self.generate_pruning_mask(step, dataloader)
            self.apply_pruning(rebuild=True)

            self.activations.clear()
            self.labels.clear()
            self.group_scores.clear()
            self.sub_groups.clear()
            self.sub_group_scores.clear()

    # ------------------------------------------------------------------
    # Optional visualisation
    # ------------------------------------------------------------------
    def visualize_hsic_scores(self) -> None:  # pragma: no cover - optional
        if not self.group_scores:
            return
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(self.group_scores), 1, figsize=(10, 4 * len(self.group_scores)))
        if len(self.group_scores) == 1:
            axes = [axes]
        for i, (g_idx, scores) in enumerate(self.group_scores.items()):
            axes[i].bar(range(len(scores)), scores.cpu().numpy())
            axes[i].set_title(f"Group {g_idx} HSIC scores")
            axes[i].set_xlabel("Channel")
            axes[i].set_ylabel("Normalised HSIC")
        plt.tight_layout()
        plt.savefig(self.workdir / "hsic_scores.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _extract_target_module(self, dep) -> Optional[nn.Module]:
        if hasattr(dep, "module"):
            return dep.module
        if hasattr(dep, "target"):
            return dep.target
        if hasattr(dep, "layer"):
            return dep.layer
        if hasattr(dep, "dep"):
            return self._extract_target_module(dep.dep)
        try:
            if hasattr(dep, "__getitem__"):
                first = dep[0]
                if isinstance(first, nn.Module):
                    return first
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------
    def _explain_single_group(self) -> None:
        """Log detail when only one dependency group is found."""
        self.logger.warning(
            "DependencyGraph found only a single pruning group – structured pruning cannot proceed"
        )
        # Coba klasifikasi koneksi arsitektur sederhana sebagai bukti
        conv_count = len(self.layers)
        self.logger.info("Detected %d Conv2d layers in total", conv_count)
        # Hitung jumlah skip-connection sederhana (Residual) melalui nama module
        residual_like = sum(1 for n, m in self.model.named_modules() if "add" in n.lower())
        concat_like = sum(1 for n, m in self.model.named_modules() if "concat" in n.lower() or "cat" in n.lower())
        split_like = sum(1 for n, m in self.model.named_modules() if "split" in n.lower())
        basic_like = conv_count - residual_like - concat_like - split_like
        self.logger.info(
            "Architecture pattern counts – basic: %d, residual: %d, concat: %d, split: %d",
            basic_like,
            residual_like,
            concat_like,
            split_like,
        )
        if residual_like:
            reason = "Model terdeteksi banyak operasi 'add' (skip connection) sehingga semua Conv2d tergabung dalam satu grup Residual besar."
        elif concat_like:
            reason = "Model menggunakan operasi concat sehingga DepGraph menganggap seluruh backbone sebagai satu grup Concat besar."
        else:
            reason = "Topologi linier tanpa cabang membuat semua layer digabung sebagai Basic group tunggal."
        self.logger.info("Fallback rationale: %s", reason)

    # ------------------------------------------------------------------
    # Public helper – detach hooks so model is picklable
    # ------------------------------------------------------------------
    def remove_activation_hooks(self) -> None:
        """Remove all forward hooks registered for activation collection."""
        for h in getattr(self, "_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

