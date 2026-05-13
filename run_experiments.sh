#!/bin/bash
# Run experiments for Fuzzy BPA EGNN

# Configuration
DATA_PATH="./dataset"
OUTPUT_DIR="./results"
EPOCHS=100
BATCH_SIZE=32
LR=1e-4
DEVICE="cuda"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $DATA_PATH

# Datasets
DATASETS=("ETTh1" "ETTh2" "Weather" "Exchange")
PRED_LENS=(96 192 336)
METHODS=("dempster" "murphy" "yager" "average")

# Run all experiments
echo "Starting all experiments..."

for DATASET in "${DATASETS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            echo "Running: $DATASET, pred_len=$PRED_LEN, method=$METHOD"
            
            python run_experiments.py \
                --dataset $DATASET \
                --pred_len $PRED_LEN \
                --method $METHOD \
                --data_path $DATA_PATH \
                --output_dir $OUTPUT_DIR \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --lr $LR \
                --device $DEVICE
        done
    done
done

# Generate comparison table
echo "Generating comparison table..."
python -c "
import json
import os

results_path = os.path.join('$OUTPUT_DIR', 'all_results.json')
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
    
    # Generate table
    lines = ['\\\\begin{table}[h]', '\\\\centering', '\\\\caption{Comparison of Evidence Combination Methods}']
    lines.append('\\\\begin{tabular}{l|cccc}')
    lines.append('\\\\hline')
    lines.append('Dataset & Dempster & Murphy & Yager & Average \\\\\\\\')
    lines.append('\\\\hline')
    
    # Group results
    grouped = {}
    for key, res in results.items():
        if 'error' in res:
            continue
        parts = key.rsplit('_', 2)
        dataset = parts[0]
        pred_len = parts[1]
        method = parts[2]
        group_key = f'{dataset}_{pred_len}'
        if group_key not in grouped:
            grouped[group_key] = {}
        grouped[group_key][method] = res
    
    for group_key in sorted(grouped.keys()):
        methods = grouped[group_key]
        row = [group_key]
        for method in ['dempster', 'murphy', 'yager', 'average']:
            if method in methods:
                res = methods[method]
                row.append(f\"{res['test_mse']:.4f}/{res['test_mae']:.4f}\")
            else:
                row.append('-')
        lines.append(' & '.join(row) + ' \\\\\\\\')
    
    lines.append('\\\\hline')
    lines.append('\\\\end{tabular}')
    lines.append('\\\\end{table}')
    
    with open(os.path.join('$OUTPUT_DIR', 'comparison_table.tex'), 'w') as f:
        f.write('\\n'.join(lines))
    
    print('Table saved to $OUTPUT_DIR/comparison_table.tex')
"

echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
