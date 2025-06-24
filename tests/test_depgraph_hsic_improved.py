#!/usr/bin/env python3
"""
Test script untuk memverifikasi implementasi DepGraph-HSIC yang diperbaiki
"""

import pytest
pytest.skip("requires heavy dependencies", allow_module_level=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Import DepGraph-HSIC method
from prune_methods.depgraph_hsic import DepgraphHSICMethod


def create_test_dataloader(num_samples=50, batch_size=4):
    """Create a simple test dataloader with synthetic data."""
    # Create synthetic images and labels
    images = torch.randn(num_samples, 3, 640, 640)
    labels = torch.randint(0, 10, (num_samples, 1)).float()  # 10 classes
    
    # Create dataset and dataloader
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def test_depgraph_hsic_basic():
    """Test basic functionality of improved DepGraph-HSIC."""
    print("Testing DepGraph-HSIC basic functionality...")
    
    # Load a small YOLO model
    model = YOLO("yolov8n.pt")
    
    # Create test dataloader
    dataloader = create_test_dataloader(num_samples=20, batch_size=2)
    
    # Create temporary workdir
    with tempfile.TemporaryDirectory() as workdir:
        # Initialize DepGraph-HSIC method with backbone scope
        pruner = DepgraphHSICMethod(model, workdir=workdir, pruning_scope="backbone")
        
        try:
            # Test 1: Analyze model
            print("  - Testing model analysis...")
            pruner.analyze_model()
            print("    ✓ Model analysis successful")
            
            # Test 2: Generate pruning mask
            print("  - Testing pruning mask generation...")
            pruner.generate_pruning_mask(ratio=0.1, dataloader=dataloader)
            print("    ✓ Pruning mask generation successful")
            
            # Test 3: Apply pruning
            print("  - Testing pruning application...")
            pruner.apply_pruning()
            print("    ✓ Pruning application successful")
            
            # Test 4: Validate pruning
            print("  - Testing pruning validation...")
            validation = pruner.validate_pruning()
            print(f"    ✓ Validation successful: {validation}")
            
            # Test 5: Get summary
            print("  - Testing summary generation...")
            summary = pruner.get_pruning_summary()
            print(f"    ✓ Summary generated: {summary}")
            
            print("✓ All basic tests passed!")
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            return False


def test_depgraph_hsic_pruning_scopes():
    """Test different pruning scopes (backbone vs full)."""
    print("Testing DepGraph-HSIC pruning scopes...")
    
    # Load a small YOLO model
    model = YOLO("yolov8n.pt")
    
    # Create test dataloader
    dataloader = create_test_dataloader(num_samples=20, batch_size=2)
    
    # Create temporary workdir
    with tempfile.TemporaryDirectory() as workdir:
        # Test backbone scope
        print("  - Testing backbone scope...")
        pruner_backbone = DepgraphHSICMethod(model, workdir=workdir, pruning_scope="backbone")
        pruner_backbone.analyze_model()
        backbone_layers = len(pruner_backbone.layers)
        print(f"    ✓ Backbone scope: {backbone_layers} layers")
        
        # Test full scope
        print("  - Testing full scope...")
        pruner_full = DepgraphHSICMethod(model, workdir=workdir, pruning_scope="full")
        pruner_full.analyze_model()
        full_layers = len(pruner_full.layers)
        print(f"    ✓ Full scope: {full_layers} layers")
        
        # Verify that full scope has more layers than backbone
        if full_layers > backbone_layers:
            print("    ✓ Full scope correctly has more layers than backbone")
        else:
            print("    ✗ Full scope should have more layers than backbone")
            return False
        
        print("✓ All pruning scope tests passed!")
        return True


def test_depgraph_hsic_error_handling():
    """Test error handling of improved DepGraph-HSIC."""
    print("Testing DepGraph-HSIC error handling...")
    
    # Load a small YOLO model
    model = YOLO("yolov8n.pt")
    
    # Create temporary workdir
    with tempfile.TemporaryDirectory() as workdir:
        # Initialize DepGraph-HSIC method
        pruner = DepgraphHSICMethod(model, workdir=workdir)
        
        # Test 1: No dataloader provided
        print("  - Testing no dataloader error...")
        try:
            pruner.generate_pruning_mask(ratio=0.1, dataloader=None)
            print("    ✗ Should have raised ValueError")
            return False
        except ValueError as e:
            if "Dataloader is required" in str(e):
                print("    ✓ Correctly raised ValueError for missing dataloader")
            else:
                print(f"    ✗ Unexpected error: {e}")
                return False
        
        # Test 2: Invalid pruning ratio
        print("  - Testing invalid pruning ratio...")
        dataloader = create_test_dataloader(num_samples=10, batch_size=2)
        try:
            pruner.generate_pruning_mask(ratio=1.5, dataloader=dataloader)
            print("    ✗ Should have raised ValueError")
            return False
        except ValueError as e:
            if "Pruning ratio must be between 0 and 1" in str(e):
                print("    ✓ Correctly raised ValueError for invalid ratio")
            else:
                print(f"    ✗ Unexpected error: {e}")
                return False
        
        # Test 3: Invalid pruning scope
        print("  - Testing invalid pruning scope...")
        try:
            pruner_invalid = DepgraphHSICMethod(model, workdir=workdir, pruning_scope="invalid")
            print("    ✗ Should have raised ValueError for invalid scope")
            return False
        except Exception as e:
            print(f"    ✓ Correctly handled invalid scope: {e}")
        
        print("✓ All error handling tests passed!")
        return True


def test_depgraph_hsic_validation():
    """Test validation features of improved DepGraph-HSIC."""
    print("Testing DepGraph-HSIC validation features...")
    
    # Load a small YOLO model
    model = YOLO("yolov8n.pt")
    
    # Create test dataloader
    dataloader = create_test_dataloader(num_samples=20, batch_size=2)
    
    # Create temporary workdir
    with tempfile.TemporaryDirectory() as workdir:
        # Initialize DepGraph-HSIC method
        pruner = DepgraphHSICMethod(model, workdir=workdir)
        
        try:
            # Generate and apply pruning
            pruner.analyze_model()
            pruner.generate_pruning_mask(ratio=0.05, dataloader=dataloader)  # Small ratio
            pruner.apply_pruning()
            
            # Test validation
            validation = pruner.validate_pruning()
            
            # Check validation results
            assert "model_functional" in validation, "Missing model_functional in validation"
            assert "pruning_ratio_achieved" in validation, "Missing pruning_ratio_achieved in validation"
            assert "total_channels_remaining" in validation, "Missing total_channels_remaining in validation"
            
            print(f"  - Model functional: {validation['model_functional']}")
            print(f"  - Pruning ratio achieved: {validation['pruning_ratio_achieved']:.3f}")
            print(f"  - Channels remaining: {validation['total_channels_remaining']}")
            
            # Test summary
            summary = pruner.get_pruning_summary()
            assert "total_pruning_groups" in summary, "Missing total_pruning_groups in summary"
            assert "layers_analyzed" in summary, "Missing layers_analyzed in summary"
            
            print(f"  - Total pruning groups: {summary['total_pruning_groups']}")
            print(f"  - Layers analyzed: {summary['layers_analyzed']}")
            
            print("✓ All validation tests passed!")
            return True
            
        except Exception as e:
            print(f"✗ Validation test failed: {e}")
            return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Improved DepGraph-HSIC Implementation")
    print("=" * 60)
    
    tests = [
        test_depgraph_hsic_basic,
        test_depgraph_hsic_pruning_scopes,
        test_depgraph_hsic_error_handling,
        test_depgraph_hsic_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! DepGraph-HSIC implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    main() 