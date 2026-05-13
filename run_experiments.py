"""
Run Experiments Script for Fuzzy BPA EGNN

This script runs experiments for:
- Multiple datasets (Electricity, Traffic, Weather, ETT)
- Multiple prediction horizons (96, 192, 336)
- Multiple evidence combination methods
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fuzzy_bpa_egnn import FuzzyBPAEGNN, FuzzyBPAEGNNConfig
from train import Trainer, TimeSeriesDataset
from data_provider import DataProvider
from torch.utils.data import DataLoader


def get_config(
    dataset: str,
    pred_len: int,
    evidence_method: str = 'dempster',
) -> FuzzyBPAEGNNConfig:
    """Get model configuration for a specific dataset and prediction horizon."""
    
    # Dataset-specific configurations
    dataset_configs = {
        'Electricity': {
            'enc_in': 321,
            'd_model': 128,
            'seq_len': 336,
        },
        'Traffic': {
            'enc_in': 862,
            'd_model': 128,
            'seq_len': 336,
        },
        'Weather': {
            'enc_in': 21,
            'd_model': 64,
            'seq_len': 336,
        },
        'ETTh1': {
            'enc_in': 7,
            'd_model': 64,
            'seq_len': 336,
        },
        'ETTh2': {
            'enc_in': 7,
            'd_model': 64,
            'seq_len': 336,
        },
        'ETTm1': {
            'enc_in': 7,
            'd_model': 64,
            'seq_len': 336,
        },
        'ETTm2': {
            'enc_in': 7,
            'd_model': 64,
            'seq_len': 336,
        },
        'Exchange': {
            'enc_in': 8,
            'd_model': 64,
            'seq_len': 336,
        },
    }
    
    if dataset not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    config = FuzzyBPAEGNNConfig(
        pred_len=pred_len,
        fusion_method=evidence_method,
        **dataset_configs[dataset],
    )
    
    return config


def run_single_experiment(
    dataset: str,
    pred_len: int,
    evidence_method: str,
    data_path: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Run a single experiment."""
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}, Pred Len: {pred_len}, Method: {evidence_method}")
    print(f"{'='*60}\n")
    
    # Get configuration
    config = get_config(dataset, pred_len, evidence_method)
    
    # Add training config
    config.epochs = epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.checkpoint_dir = os.path.join(output_dir, 'checkpoints', f"{dataset}_{pred_len}_{evidence_method}")
    
    # Get data
    provider = DataProvider(
        data_path=data_path,
        dataset_name=dataset,
        seq_len=config.seq_len,
        pred_len=pred_len,
    )
    train_loader, val_loader, test_loader = provider.get_data(batch_size=batch_size)
    
    # Create model
    model = FuzzyBPAEGNN(config)
    model = model.to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=torch.device(device),
    )
    
    # Train
    history = trainer.train(epochs)
    
    # Test
    model.eval()
    test_losses = []
    test_maes = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            mae = torch.mean(torch.abs(output - y))
            test_losses.append(loss.item())
            test_maes.append(mae.item())
    
    test_mse = np.mean(test_losses)
    test_mae = np.mean(test_maes)
    
    print(f"\nTest MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Save results
    results = {
        'dataset': dataset,
        'pred_len': pred_len,
        'evidence_method': evidence_method,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'num_params': num_params,
        'history': history,
    }
    
    result_path = os.path.join(output_dir, 'results', f"{dataset}_{pred_len}_{evidence_method}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'history'}, f, indent=2)
    
    return results


def run_all_experiments(
    datasets: List[str],
    pred_lens: List[int],
    evidence_methods: List[str],
    data_path: str,
    output_dir: str,
    **kwargs,
) -> Dict[str, Any]:
    """Run all experiments."""
    
    all_results = {}
    
    for dataset in datasets:
        for pred_len in pred_lens:
            for method in evidence_methods:
                key = f"{dataset}_{pred_len}_{method}"
                try:
                    results = run_single_experiment(
                        dataset=dataset,
                        pred_len=pred_len,
                        evidence_method=method,
                        data_path=data_path,
                        output_dir=output_dir,
                        **kwargs,
                    )
                    all_results[key] = results
                except Exception as e:
                    print(f"Error in {key}: {e}")
                    all_results[key] = {'error': str(e)}
    
    # Save all results
    results_path = os.path.join(output_dir, 'all_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


def generate_comparison_table(results: Dict[str, Any]) -> str:
    """Generate a comparison table of results."""
    
    # Group by dataset and pred_len
    grouped = {}
    for key, res in results.items():
        if 'error' in res:
            continue
        parts = key.rsplit('_', 2)
        dataset = parts[0]
        pred_len = parts[1]
        method = parts[2]
        
        group_key = f"{dataset}_{pred_len}"
        if group_key not in grouped:
            grouped[group_key] = {}
        grouped[group_key][method] = res
    
    # Generate table
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of Evidence Combination Methods}")
    lines.append("\\begin{tabular}{l|cccc}")
    lines.append("\\hline")
    lines.append("Dataset & Dempster & Murphy & Yager & Average \\\\")
    lines.append("\\hline")
    
    for group_key, methods in sorted(grouped.items()):
        row = [group_key]
        for method in ['dempster', 'murphy', 'yager', 'average']:
            if method in methods:
                res = methods[method]
                row.append(f"{res['test_mse']:.4f}/{res['test_mae']:.4f}")
            else:
                row.append("-")
        lines.append(" & ".join(row) + " \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Run Fuzzy BPA EGNN experiments')
    parser.add_argument('--dataset', type=str, default='ETTh1',
                        help='Dataset name (default: ETTh1)')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction length (default: 96)')
    parser.add_argument('--method', type=str, default='dempster',
                        help='Evidence combination method (default: dempster)')
    parser.add_argument('--data_path', type=str, default='./dataset',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')
    
    args = parser.parse_args()
    
    if args.all:
        # Run all experiments
        datasets = ['ETTh1', 'ETTh2', 'Weather', 'Exchange']
        pred_lens = [96, 192, 336]
        evidence_methods = ['dempster', 'murphy', 'yager', 'average']
        
        results = run_all_experiments(
            datasets=datasets,
            pred_lens=pred_lens,
            evidence_methods=evidence_methods,
            data_path=args.data_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
        )
        
        # Generate comparison table
        table = generate_comparison_table(results)
        print("\n" + table)
        
        with open(os.path.join(args.output_dir, 'comparison_table.tex'), 'w') as f:
            f.write(table)
    else:
        # Run single experiment
        run_single_experiment(
            dataset=args.dataset,
            pred_len=args.pred_len,
            evidence_method=args.method,
            data_path=args.data_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
        )


if __name__ == '__main__':
    main()
