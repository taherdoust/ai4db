# Thesis Figures and Tables Guide

Quick reference for creating tables/figures from FTv2 evaluation results.

## Data Sources

After running evaluation:
- `results_*.json` - Full evaluation results with breakdowns
- `ftv2_benchmark_100.meta.json` - Benchmark distributions
- Training logs - Loss curves and timing data

## Essential Tables for Thesis

### Table 1: Overall Model Comparison

**Source**: Compare multiple `results_*.json` files

```
Model               | Params | EM    | EX    | Time   | Status
--------------------|--------|-------|-------|--------|----------
FT Qwen 2.5 14B    | 14B    | TBD   | TBD   | 2-3s   | Training
FT Llama 3.1 8B    | 8B     | 32%   | 87%   | 1-2s   | Complete
FT Qwen 2.5 7B     | 7B     | 28%   | 84%   | 1-2s   | Complete
Baseline Qwen 14B  | 14B    | 15%   | 58%   | 2-3s   | Tested
GPT-4o-turbo       | --     | 45%   | 91%   | 3-5s   | API
```

**Extract from JSON**:
```python
import json

with open('results_llama8b_q2sql.json') as f:
    data = json.load(f)
    em = data['em_accuracy'] * 100
    ex = data['ex_accuracy'] * 100
    print(f"EM: {em:.0f}%, EX: {ex:.0f}%")
```

### Table 2: Query Complexity Breakdown

**Source**: `performance_breakdowns.query_complexity` from each JSON

```
Model          | EASY (EM/EX) | MEDIUM (EM/EX) | HARD (EM/EX)
---------------|--------------|----------------|-------------
FT Llama 8B    | 42%/95%      | 28%/89%        | 18%/75%
FT Qwen 7B     | 38%/92%      | 24%/85%        | 15%/71%
Baseline       | 22%/71%      | 12%/58%        | 8%/42%
```

**Extract from JSON**:
```python
breakdowns = data['performance_breakdowns']['query_complexity']
for category in ['EASY', 'MEDIUM', 'HARD']:
    stats = breakdowns[category]
    em = stats['em_accuracy'] * 100
    ex = stats['ex_accuracy'] * 100
    print(f"{category}: EM={em:.0f}%, EX={ex:.0f}%")
```

### Table 3: Spatial Complexity Breakdown

**Source**: `performance_breakdowns.spatial_complexity`

```
Model          | BASIC (EM/EX) | INTERMEDIATE (EM/EX) | ADVANCED (EM/EX)
---------------|---------------|----------------------|------------------
FT Llama 8B    | 38%/93%       | 30%/86%              | 22%/78%
FT Qwen 7B     | 35%/90%       | 26%/82%              | 18%/74%
Baseline       | 20%/68%       | 14%/55%              | 8%/38%
```

### Table 4: Schema Complexity Breakdown

**Source**: `performance_breakdowns.schema_complexity`

```
Model          | SINGLE_TABLE | SINGLE_SCHEMA | MULTI_SCHEMA
---------------|--------------|---------------|-------------
FT Llama 8B    | 40%/94%      | 32%/88%       | 20%/78%
FT Qwen 7B     | 36%/91%      | 28%/84%       | 16%/74%
Baseline       | 18%/65%      | 14%/58%       | 10%/45%
```

### Table 5: Training Time and Cost

**Source**: Training logs + manual calculation

```
Model Size | Time/Epoch | Total (3 epochs) | Academic Cost | Cloud Cost (typical)
-----------|------------|------------------|---------------|---------------------
7B         | 14h        | 42h              | $0            | $154-390
8B         | 17h        | 51h              | $0            | $187-475
14B        | 63h        | 189h             | $0            | $694-1,751
```

## Essential Figures for Thesis

### Figure 1: Loss Convergence Curves

**Data**: Training logs → extract loss values

```python
# From training logs
epochs = [0.25, 0.50, 0.75, 1.0, 1.25, 1.5]
train_loss = [0.329, 0.232, 0.168, 0.131, 0.112, ...]
eval_loss = [0.330, 0.322, 0.318, 0.097, 0.095, ...]

# Plot
import matplotlib.pyplot as plt
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, eval_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Convergence (Llama 3.1 8B)')
plt.savefig('loss_convergence.pdf')
```

### Figure 2: Performance Breakdown Heatmap

**Data**: All breakdowns from JSON

```python
import seaborn as sns
import pandas as pd

# Extract data
models = ['FT Llama 8B', 'FT Qwen 7B', 'Baseline']
categories = ['EASY', 'MEDIUM', 'HARD']

# Create matrix
data = [
    [95, 89, 75],  # FT Llama 8B
    [92, 85, 71],  # FT Qwen 7B
    [71, 58, 42]   # Baseline
]

df = pd.DataFrame(data, index=models, columns=categories)

# Plot heatmap
sns.heatmap(df, annot=True, fmt='d', cmap='RdYlGn', vmin=0, vmax=100)
plt.title('EX Accuracy (%) by Query Complexity')
plt.xlabel('Query Complexity')
plt.ylabel('Model')
plt.savefig('complexity_heatmap.pdf')
```

### Figure 3: Training Time Scaling

**Data**: Manual from training logs

```python
model_sizes = [7, 8, 14]
training_times = [42, 51, 189]

plt.figure(figsize=(8, 6))
plt.plot(model_sizes, training_times, 'o-', linewidth=2, markersize=10)
plt.xlabel('Model Size (Billions of Parameters)')
plt.ylabel('Training Time (Hours, 3 epochs)')
plt.title('Training Time Scaling')
plt.grid(True, alpha=0.3)
plt.xticks([7, 8, 14])
plt.savefig('training_time_scaling.pdf')
```

### Figure 4: Performance Gap by Dimension

**Data**: Calculate gaps between FT and baseline

```python
import numpy as np

dimensions = ['Query\nComplexity', 'Spatial\nComplexity', 
              'Schema\nComplexity', 'Overall']

# Average gaps across difficulty levels
ft_gaps = [24, 28, 29, 29]  # Average % improvement

x = np.arange(len(dimensions))
width = 0.6

plt.figure(figsize=(10, 6))
plt.bar(x, ft_gaps, width, color='steelblue', alpha=0.8)
plt.xlabel('Difficulty Dimension')
plt.ylabel('Performance Gap (%)')
plt.title('Fine-Tuned vs Baseline Performance Gap')
plt.xticks(x, dimensions)
plt.ylim(0, 35)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(ft_gaps):
    plt.text(i, v + 1, f'+{v}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('performance_gap.pdf')
```

### Figure 5: Cost Comparison

**Data**: Manual calculation

```python
platforms = ['IPAZIA\n(Academic)', 'Vast.ai\n(Spot)', 'Lambda\nLabs', 
             'Google\nCloud', 'AWS']
costs_7b = [0, 37, 54, 154, 172]
costs_14b = [0, 168, 244, 694, 775]

x = np.arange(len(platforms))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, costs_7b, width, label='7B Model (42h)', color='lightblue')
ax.bar(x + width/2, costs_14b, width, label='14B Model (189h)', color='darkblue')

ax.set_ylabel('Cost (USD)')
ax.set_title('Fine-Tuning Cost Comparison Across Platforms')
ax.set_xticks(x)
ax.set_xticklabels(platforms)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('cost_comparison.pdf')
```

## Python Script for Batch Processing

```python
#!/usr/bin/env python3
"""
Generate all thesis tables and figures from evaluation results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def load_results(result_files):
    """Load all result JSON files."""
    results = {}
    for file in result_files:
        with open(file) as f:
            model_name = file.stem.replace('results_', '')
            results[model_name] = json.load(f)
    return results

def create_overall_comparison_table(results):
    """Create Table 1: Overall comparison."""
    data = []
    for model_name, result in results.items():
        data.append({
            'Model': model_name,
            'EM': f"{result['em_accuracy']*100:.0f}%",
            'EX': f"{result['ex_accuracy']*100:.0f}%",
            'Samples': result['total_samples']
        })
    
    df = pd.DataFrame(data)
    print("Table 1: Overall Comparison")
    print(df.to_string(index=False))
    return df

def create_complexity_breakdown_table(results, dimension):
    """Create breakdown table for any dimension."""
    print(f"\nTable: {dimension} Breakdown")
    
    for model_name, result in results.items():
        breakdowns = result['performance_breakdowns'][dimension]
        print(f"\n{model_name}:")
        for category, stats in sorted(breakdowns.items()):
            em = stats['em_accuracy'] * 100
            ex = stats['ex_accuracy'] * 100
            total = stats['total']
            print(f"  {category:20s}: EM={em:5.1f}%, EX={ex:5.1f}% (n={total})")

def plot_loss_curves(log_file, output='loss_curves.pdf'):
    """Plot training and validation loss curves."""
    # Parse log file for loss values
    # This depends on your log format
    pass

def main():
    # Specify result files
    result_files = [
        Path('results_llama8b_q2sql.json'),
        Path('results_qwen7b_q2sql.json'),
        Path('results_baseline_qwen14b.json')
    ]
    
    # Load results
    results = load_results(result_files)
    
    # Generate tables
    create_overall_comparison_table(results)
    create_complexity_breakdown_table(results, 'query_complexity')
    create_complexity_breakdown_table(results, 'spatial_complexity')
    create_complexity_breakdown_table(results, 'schema_complexity')
    
    # Generate figures
    # plot_loss_curves('training.log')
    # plot_performance_heatmap(results)
    # plot_cost_comparison()
    
    print("\n✓ All tables and figures generated!")

if __name__ == '__main__':
    main()
```

## LaTeX Tips

### Including Tables

```latex
\begin{table}[htbp]
\centering
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Type} & \textbf{Params} & \textbf{EM} & \textbf{EX} & \textbf{Time} \\
\midrule
FT Llama 8B & Q2SQL & 8B & 32\% & 87\% & 1-2s \\
FT Qwen 7B & Q2SQL & 7B & 28\% & 84\% & 1-2s \\
\bottomrule
\end{tabular}
\caption{Model performance comparison}
\label{tab:model_comparison}
\end{table}
```

### Including Figures

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{loss_convergence.pdf}
\caption{Training and validation loss curves showing stable convergence}
\label{fig:loss_convergence}
\end{figure}
```

### Referencing

In text:
```latex
As shown in Table~\ref{tab:model_comparison}, fine-tuned models achieve 
84-87\% EX accuracy. Figure~\ref{fig:loss_convergence} demonstrates stable 
convergence without overfitting.
```

## Quick Checklist

- [ ] Generate benchmark with difficulty dimensions
- [ ] Evaluate all Q2SQL models
- [ ] Extract breakdown tables from JSON
- [ ] Create loss curve plots from logs
- [ ] Generate performance heatmaps
- [ ] Create cost comparison figure
- [ ] Update results.tex with actual values
- [ ] Cross-check all numbers match JSON outputs
- [ ] Add figure/table captions and labels
- [ ] Update references in text

---

**Note**: Replace placeholder values with actual results as evaluations complete.

**Author**: Ali Taherdoust  
**Date**: November 18, 2024

