#!/usr/bin/env python3
"""
Test script untuk memverifikasi bahwa default methods menggunakan seluruh method yang tersedia
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import METHODS_MAP, parse_args


def test_default_methods():
    """Test bahwa default methods menggunakan seluruh method yang tersedia."""
    print("Testing default methods configuration...")
    
    # Test 1: Verifikasi METHODS_MAP berisi semua method
    expected_methods = [
        "l1", "random", "depgraph", "tp_random", 
        "isomorphic", "hsic_lasso", "whc", "depgraph_hsic"
    ]
    
    print(f"  - METHODS_MAP keys: {list(METHODS_MAP.keys())}")
    print(f"  - Expected methods: {expected_methods}")
    
    for method in expected_methods:
        if method not in METHODS_MAP:
            print(f"    ✗ Missing method: {method}")
            return False
        else:
            print(f"    ✓ Found method: {method}")
    
    # Test 2: Verifikasi default arguments menggunakan semua method
    # Simulasi args tanpa --methods
    sys.argv = [
        "main.py",
        "--model", "yolov8n.pt",
        "--data", "biotech_model_train.yaml"
    ]
    
    args = parse_args()
    print(f"  - Default methods from args: {args.methods}")
    
    if set(args.methods) == set(METHODS_MAP.keys()):
        print("    ✓ Default methods correctly use all available methods")
    else:
        print("    ✗ Default methods should use all available methods")
        print(f"    Expected: {set(METHODS_MAP.keys())}")
        print(f"    Got: {set(args.methods)}")
        return False
    
    # Test 3: Verifikasi bahwa specific methods work
    sys.argv = [
        "main.py",
        "--model", "yolov8n.pt",
        "--data", "biotech_model_train.yaml",
        "--methods", "l1", "depgraph_hsic"
    ]
    
    args = parse_args()
    print(f"  - Specific methods from args: {args.methods}")
    
    if args.methods == ["l1", "depgraph_hsic"]:
        print("    ✓ Specific methods correctly parsed")
    else:
        print("    ✗ Specific methods not correctly parsed")
        return False
    
    print("✓ All default methods tests passed!")
    return True


def test_method_choices():
    """Test bahwa choices untuk methods sesuai dengan METHODS_MAP."""
    print("Testing method choices...")
    
    # Simulasi args dengan invalid method
    sys.argv = [
        "main.py",
        "--model", "yolov8n.pt",
        "--data", "biotech_model_train.yaml",
        "--methods", "invalid_method"
    ]
    
    try:
        args = parse_args()
        print("    ✗ Should have raised error for invalid method")
        return False
    except SystemExit:
        print("    ✓ Correctly rejected invalid method")
    
    print("✓ Method choices test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Default Methods Configuration")
    print("=" * 60)
    
    tests = [
        test_default_methods,
        test_method_choices
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
        print("🎉 All tests passed! Default methods configuration is working correctly.")
    else:
        print("❌ Some tests failed. Please check the configuration.")
    
    return passed == total


if __name__ == "__main__":
    main() 