#!/usr/bin/env python3
"""
Execution script for running the deep learning training pipeline
Handles errors and provides detailed logging
"""

import subprocess
import sys
import os

def run_script():
    """Run the training script with error handling"""
    print("="*80)
    print("NVIDIA GARAGE DATASET - DEEP LEARNING MODEL TRAINING PIPELINE")
    print("="*80)
    print()
    
    # Check if train_models.py exists
    if not os.path.exists('train_models.py'):
        print("ERROR: train_models.py not found!")
        print("Please ensure train_models.py is in the current directory")
        return False
    
    # Install required packages
    print("Installing required packages...")
    required_packages = [
        'tensorflow>=2.12.0',
        'scikit-learn',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'psutil',
        'gputil'
    ]
    
    for package in required_packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-q', package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Could not install {package}: {e}")
    
    print("\nStarting training pipeline...")
    print("-"*80)
    
    # Run the training script
    try:
        result = subprocess.run(
            [sys.executable, 'train_models.py'],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n" + "="*80)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("\nOutput files generated:")
            print("  ✓ model_results.csv - Model performance metrics")
            print("  ✓ graphs/accuracy_comparison.png - Accuracy comparison chart")
            print("  ✓ graphs/auc_curves.png - AUC curves for all models")
            print("  ✓ graphs/confusion_matrices.png - Confusion matrices for each model")
            return True
        else:
            print(f"\nERROR: Script exited with code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\nERROR: Failed to run script: {e}")
        return False

if __name__ == "__main__":
    success = run_script()
    sys.exit(0 if success else 1)
