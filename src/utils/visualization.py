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
        axes[0].set_title(f'Loss (Epoch {epoch_end}/{num_epochs})', fontsize=16)
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(1, num_epochs)
        axes[0].set_ylim(0, max(history['train_loss'][:20]) * 1.1)  # Fixed y-axis
        
        # Accuracy
        axes[1].plot(epochs, history['train_acc'][:epoch_end], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, history['val_acc'][:epoch_end], 'r-', label='Val', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=14)
        axes[1].set_ylabel('Accuracy (%)', fontsize=14)
        axes[1].set_title(f'Accuracy (Epoch {epoch_end}/{num_epochs})', fontsize=16)
        axes[1].legend(fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(1, num_epochs)
        axes[1].set_ylim(0, 100)
        
        plt.tight_layout()
        
        # Save frame to buffer
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"GIF saved: {save_path}")


def plot_final_metrics(history, save_path=None):
    """
    Plot final training metrics summary
    
    Args:
        history: training history dict
        save_path: where to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss with smoothing
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', alpha=0.3, label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', alpha=0.3, label='Val Loss')
    
    # Smoothed curves (moving average)
    window = min(10, len(epochs) // 10)
    if window > 1:
        train_smooth = np.convolve(history['train_loss'], np.ones(window)/window, mode='valid')
        val_smooth = np.convolve(history['val_loss'], np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(epochs)+1), train_smooth, 'b-', linewidth=2, label='Train (smoothed)')
        ax.plot(range(window, len(epochs)+1), val_smooth, 'r-', linewidth=2, label='Val (smoothed)')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    if 'test_acc' in history and any(history['test_acc']):
        ax.plot(epochs, history['test_acc'], 'g-', label='Test', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 0]
    ax.plot(epochs, history['lr'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Final metrics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    
    summary_text = f"""
    TRAINING SUMMARY
    {'='*40}
    
    Final Train Accuracy: {final_train_acc:.2f}%
    Final Val Accuracy: {final_val_acc:.2f}%
    
    Best Val Accuracy: {best_val_acc:.2f}%
    Best Epoch: {best_epoch}/{len(epochs)}
    
    Total Epochs: {len(epochs)}
    """
    
    if 'test_acc' in history and any(history['test_acc']):
        final_test_acc = history['test_acc'][-1]
        summary_text += f"\n    Final Test Accuracy: {final_test_acc:.2f}%"
    
    ax.text(0.1, 0.5, summary_text, fontsize=14, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig


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
    print("✓ plot_training_curves works!")
