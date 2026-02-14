"""
Visualization utilities for training curves, OOD metrics, and Neural Collapse
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.animation import FuncAnimation
import imageio


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history: dict with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: if provided, save figure to this path
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    if 'test_acc' in history and any(history['test_acc']):
        axes[1].plot(epochs, history['test_acc'], 'g-', label='Test Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history:
        axes[2].plot(epochs, history['lr'], 'purple', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig


def create_training_gif(history, save_path='results/figures/gifs/training_curves.gif', fps=10):
    """
    Create animated GIF showing training progress over epochs
    
    Args:
        history: training history dict
        save_path: where to save GIF
        fps: frames per second
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    frames = []
    num_epochs = len(history['train_loss'])
    
    # Create frame every N epochs
    step = max(1, num_epochs // 50)  # ~50 frames max
    
    for epoch_end in range(step, num_epochs + 1, step):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, epoch_end + 1)
        
        # Loss
        axes[0].plot(epochs, history['train_loss'][:epoch_end], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, history['val_loss'][:epoch_end], 'r-', label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=14)
        axes[0].set_ylabel('Loss', fontsize=14)
        axes[0].set_title(f'Loss (Epoch {epoch_end}/{num_epochs})', fontsize=16, fontweight='bold')
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(1, num_epochs)
        axes[0].set_ylim(0, max(history['train_loss'][:20]) * 1.1)  # Fixed y-axis
        
        # Accuracy
        axes[1].plot(epochs, history['train_acc'][:epoch_end], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, history['val_acc'][:epoch_end], 'r-', label='Val', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=14)
        axes[1].set_ylabel('Accuracy (%)', fontsize=14)
        axes[1].set_title(f'Accuracy (Epoch {epoch_end}/{num_epochs})', fontsize=16, fontweight='bold')
        axes[1].legend(fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(1, num_epochs)
        axes[1].set_ylim(0, 100)
        
        plt.tight_layout()
        
        # Save frame to buffer (CORRIGIDO)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)
        image_rgb = image[:, :, :3]  # Converter RGBA para RGB
        frames.append(image_rgb)
        
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"GIF saved: {save_path}")

def plot_ood_scores(results, save_dir=None):
    """
    Plot OOD detection results
    
    Args:
        results: dict with structure:
            {
                'config': {'epochs': [...], 'tpt_mask': [...]},
                'scorers': {
                    'MSP': {'auroc': [...], 'fpr95': [...]},
                    ...
                }
            }
        save_dir: directory to save plots
    """
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    epochs = results['config']['epochs']
    tpt_mask = results['config']['tpt_mask']
    scorers = results['scorers']
    
    # TPT region (where mask == 1)
    tpt_epochs = [e for e, m in zip(epochs, tpt_mask) if m == 1]
    tpt_start = min(tpt_epochs) if tpt_epochs else None
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: AUROC vs Epoch
    ax = axes[0]
    for scorer_name, data in scorers.items():
        if scorer_name == 'NECO':
            # NECO only in TPT
            ax.plot(tpt_epochs, data['auroc'], 
                   '--o', linewidth=2, markersize=8, 
                   label=f'{scorer_name} (TPT only)')
        else:
            ax.plot(epochs, data['auroc'], 
                   '-o', linewidth=2, markersize=6, 
                   label=scorer_name)
    
    # Shade TPT region
    if tpt_start:
        ax.axvspan(tpt_start, max(epochs), alpha=0.1, color='red', label='TPT Phase')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('AUROC', fontsize=14)
    ax.set_title('OOD Detection: AUROC vs Training Epoch', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    # Plot 2: FPR@95 vs Epoch
    ax = axes[1]
    for scorer_name, data in scorers.items():
        if scorer_name == 'NECO':
            ax.plot(tpt_epochs, data['fpr95'], 
                   '--o', linewidth=2, markersize=8, 
                   label=f'{scorer_name} (TPT only)')
        else:
            ax.plot(epochs, data['fpr95'], 
                   '-o', linewidth=2, markersize=6, 
                   label=scorer_name)
    
    # Shade TPT region
    if tpt_start:
        ax.axvspan(tpt_start, max(epochs), alpha=0.1, color='red')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('FPR@95 (%)', fontsize=14)
    ax.set_title('OOD Detection: FPR@95 vs Training Epoch', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'ood_scores_evolution.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_dir}/ood_scores_evolution.png")
    
    plt.show()
    
    # Plot 3: Final comparison (bar plot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scorer_names = list(scorers.keys())
    final_auroc = [scorers[s]['auroc'][-1] for s in scorer_names]
    final_fpr95 = [scorers[s]['fpr95'][-1] for s in scorer_names]
    
    # AUROC bars
    ax = axes[0]
    bars = ax.bar(scorer_names, final_auroc, color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
    ax.set_ylabel('AUROC', fontsize=14)
    ax.set_title(f'Final OOD Detection Performance (Epoch {epochs[-1]})', 
                fontsize=16, fontweight='bold')
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, final_auroc):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    # FPR@95 bars
    ax = axes[1]
    bars = ax.bar(scorer_names, final_fpr95, color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
    ax.set_ylabel('FPR@95 (%)', fontsize=14)
    ax.set_title(f'False Positive Rate @ 95% TPR (Epoch {epochs[-1]})', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, final_fpr95):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'ood_scores_final.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_dir}/ood_scores_final.png")
    
    plt.show()


if __name__ == "__main__":
    # Test with dummy data
    dummy_history = {
        'train_loss': np.linspace(2.0, 0.1, 100) + np.random.randn(100) * 0.05,
        'val_loss': np.linspace(2.0, 0.3, 100) + np.random.randn(100) * 0.08,
        'train_acc': np.linspace(10, 99, 100) + np.random.randn(100) * 2,
        'val_acc': np.linspace(10, 80, 100) + np.random.randn(100) * 3,
        'lr': [0.1 * (0.99 ** i) for i in range(100)]
    }
    
    print("Testing visualization functions...")
    plot_training_curves(dummy_history)
    print("plot_training_curves works!")
