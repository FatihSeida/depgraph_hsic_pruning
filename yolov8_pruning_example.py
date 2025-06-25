import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO


def l1_importance(weights: torch.Tensor, amount: int) -> list:
    """L1-norm based importance function."""
    importance = weights.abs().sum(dim=(1, 2, 3))
    _, indices = torch.sort(importance)
    return indices[:amount].tolist()


def random_importance(weights: torch.Tensor, amount: int) -> list:
    """Random importance function."""
    indices = torch.randperm(weights.shape[0])[:amount]
    return indices.tolist()


STRATEGY_MAP = {
    "l1": l1_importance,
    "random": random_importance,
}


def build_dependency_graph(model: nn.Module, example_inputs: torch.Tensor) -> tp.DependencyGraph:
    """Return a Torch-Pruning dependency graph built for *model*."""
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs)
    return DG


def select_prunable_layers(model: nn.Module):
    """Yield prune-able Conv2d layers (exclude depthwise and final detect head)."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Skip depthwise conv or very small layers (e.g., channel==1)
            if m.groups == m.in_channels or m.out_channels <= 1:
                continue
            yield m


def prune_model(model: YOLO, ratio: float = 0.2, strategy_name: str = "l1"):
    """Prune *ratio* output channels from each Conv2d layer using *strategy_name*."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(dev).eval()

    example_inputs = torch.randn(1, 3, 640, 640, device=dev)
    DG = build_dependency_graph(model.model, example_inputs)

    strategy = STRATEGY_MAP.get(strategy_name.lower())
    if strategy is None:
        raise ValueError(f"Unknown strategy {strategy_name}; choose from {list(STRATEGY_MAP)}")

    # Get all pruning groups from dependency graph
    pruning_groups = list(DG.get_all_groups(root_module_types=[nn.Conv2d]))
    print(f"Found {len(pruning_groups)} pruning groups in dependency graph")

    total_pruned = 0
    for group_idx, group in enumerate(pruning_groups):
        # Find Conv2d layers in this group
        conv_layers = []
        for dep in group:
            if hasattr(dep, 'layer') and isinstance(dep.layer, nn.Conv2d):
                conv_layers.append(dep.layer)
            elif hasattr(dep, 'module') and isinstance(dep.module, nn.Conv2d):
                conv_layers.append(dep.module)
        
        if not conv_layers:
            continue
            
        # Use the first Conv2d layer in the group for pruning decision
        layer = conv_layers[0]
        
        # Skip depthwise conv or very small layers
        if layer.groups == layer.in_channels or layer.out_channels <= 1:
            continue
            
        # Decide how many channels to prune for this layer
        n_prune = max(1, int(layer.out_channels * ratio))
        if n_prune >= layer.out_channels:
            n_prune = layer.out_channels - 1  # Keep at least 1 channel
            
        prune_idx = strategy(layer.weight, n_prune)
        
        try:
            # Try to prune the entire group
            DG.prune_group(group)
            total_pruned += len(prune_idx)
            print(f"Group {group_idx}: pruned {len(prune_idx)} channels from {layer}")
        except Exception as e:
            print(f"Warning: Failed to prune group {group_idx}: {e}")
            continue

    # Remove pruning reparametrization with fallback
    try:
        if hasattr(tp.utils, "remove_pruning_reparametrization"):
            tp.utils.remove_pruning_reparametrization(model.model)
        elif hasattr(tp, "remove_pruning_reparametrization"):
            tp.remove_pruning_reparametrization(model.model)
        else:
            print("Warning: remove_pruning_reparametrization not available, skipping...")
    except Exception as e:
        print(f"Warning: Failed to remove pruning reparametrization: {e}")
        
    print(f"Pruned total {total_pruned} channels from the model")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 pruning example with Torch-Pruning")
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to YOLOv8 weights (pt)")
    parser.add_argument("--output", default="yolov8n_pruned.pt", help="Output path for pruned model")
    parser.add_argument("--ratio", type=float, default=0.2, help="Pruning ratio per layer (0-1)")
    parser.add_argument("--strategy", choices=list(STRATEGY_MAP), default="l1", help="Pruning importance strategy")
    args = parser.parse_args()

    print("Loading model…")
    yolo = YOLO(args.weights)
    prune_model(yolo, ratio=args.ratio, strategy_name=args.strategy)

    out_path = Path(args.output)
    print(f"Saving pruned model to {out_path}")
    yolo.save(str(out_path))
    print("Done.")


if __name__ == "__main__":
    main() 