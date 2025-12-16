"""
Generate visualization graphs for model performance metrics.

This script creates publication-quality plots for the research paper.

Usage:
    python visualize_results.py --results results_fri_gpu/metrics.json --output results_fri_gpu/
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


def load_results(results_path):
    """Load metrics from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def plot_comparison_bar(results, output_dir):
    """Create bar chart comparing all metrics across models."""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=metric.capitalize())
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'comparison_bar.png', bbox_inches='tight')
    print(f"✓ Saved: comparison_bar.png")
    plt.close()


def plot_f1_comparison(results, output_dir):
    """Create focused F1-score comparison."""
    models = list(results.keys())
    f1_scores = [results[model]['f1'] for model in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#d62728' if f1 < 0.5 else '#2ca02c' if f1 > 0.9 else '#ff7f0e' 
              for f1 in f1_scores]
    
    bars = ax.barh(models, f1_scores, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        ax.text(f1 + 0.02, i, f'{f1:.4f}', 
               va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('F1-Score', fontweight='bold')
    ax.set_title('F1-Score Comparison Across Models', fontweight='bold', fontsize=12)
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add reference lines
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
    ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (>0.7)')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Moderate (>0.5)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'f1_comparison.png', bbox_inches='tight')
    print(f"✓ Saved: f1_comparison.png")
    plt.close()


def plot_metrics_heatmap(results, output_dir):
    """Create heatmap of all metrics."""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    data = np.array([[results[model][metric] for metric in metrics] 
                     for model in models])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_yticklabels(models)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Performance Metrics Heatmap', fontweight='bold', fontsize=12)
    fig.colorbar(im, ax=ax, label='Score')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'metrics_heatmap.png', bbox_inches='tight')
    print(f"✓ Saved: metrics_heatmap.png")
    plt.close()


def plot_precision_recall_tradeoff(results, output_dir):
    """Plot precision vs recall for each model."""
    models = list(results.keys())
    precisions = [results[model]['precision'] for model in models]
    recalls = [results[model]['recall'] for model in models]
    f1s = [results[model]['f1'] for model in models]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot with size based on F1
    sizes = [f1 * 500 for f1 in f1s]
    scatter = ax.scatter(recalls, precisions, s=sizes, alpha=0.6, 
                        c=f1s, cmap='viridis', edgecolors='black', linewidth=1.5)
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (recalls[i], precisions[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Recall', fontweight='bold', fontsize=11)
    ax.set_ylabel('Precision', fontweight='bold', fontsize=11)
    ax.set_title('Precision-Recall Trade-off (Bubble size = F1-Score)', 
                fontweight='bold', fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Add diagonal line (F1 iso-lines would be hyperbolas)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Precision = Recall')
    ax.legend()
    
    # Colorbar for F1
    cbar = fig.colorbar(scatter, ax=ax, label='F1-Score')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'precision_recall.png', bbox_inches='tight')
    print(f"✓ Saved: precision_recall.png")
    plt.close()


def create_summary_table(results, output_dir):
    """Create a formatted table image of results."""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Prepare data
    data = []
    for model in models:
        row = [model] + [f"{results[model][m]:.4f}" for m in metrics]
        data.append(row)
    
    fig, ax = plt.subplots(figsize=(10, len(models) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=data,
                    colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(models) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Model Performance Summary', fontweight='bold', fontsize=14, pad=20)
    plt.savefig(Path(output_dir) / 'summary_table.png', bbox_inches='tight')
    print(f"✓ Saved: summary_table.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate visualization graphs')
    parser.add_argument('--results', default='results_fri_gpu/metrics.json',
                       help='Path to metrics JSON file')
    parser.add_argument('--output', default='results_fri_gpu/',
                       help='Output directory for graphs')
    args = parser.parse_args()
    
    results_path = Path(args.results)
    output_dir = Path(args.output)
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING VISUALIZATION GRAPHS")
    print("=" * 60)
    print(f"Input:  {results_path}")
    print(f"Output: {output_dir}\n")
    
    results = load_results(results_path)
    
    print("Creating graphs...")
    plot_comparison_bar(results, output_dir)
    plot_f1_comparison(results, output_dir)
    plot_metrics_heatmap(results, output_dir)
    plot_precision_recall_tradeoff(results, output_dir)
    create_summary_table(results, output_dir)
    
    print(f"\n✓ All graphs saved to {output_dir}")
    print("\nGenerated files:")
    print("  - comparison_bar.png       (All metrics comparison)")
    print("  - f1_comparison.png        (F1-score focus)")
    print("  - metrics_heatmap.png      (Heatmap visualization)")
    print("  - precision_recall.png     (Precision-Recall trade-off)")
    print("  - summary_table.png        (Results table)")


if __name__ == '__main__':
    main()
