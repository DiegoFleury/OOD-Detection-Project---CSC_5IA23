import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import imageio

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


def plot_training_curves(history, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    if 'test_acc' in history and any(history['test_acc']):
        axes[1].plot(epochs, history['test_acc'], 'g-', label='Test Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

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

    plt.show()
    return fig


def create_training_gif(history, save_path='results/figures/gifs/training_curves.gif', fps=10):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    frames = []
    num_epochs = len(history['train_loss'])
    step = max(1, num_epochs // 50)

    for epoch_end in range(step, num_epochs + 1, step):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, epoch_end + 1)

        axes[0].plot(epochs, history['train_loss'][:epoch_end], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, history['val_loss'][:epoch_end], 'r-', label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=14)
        axes[0].set_ylabel('Loss', fontsize=14)
        axes[0].set_title(f'Loss (Epoch {epoch_end}/{num_epochs})', fontsize=16, fontweight='bold')
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(1, num_epochs)
        axes[0].set_ylim(0, max(history['train_loss'][:20]) * 1.1)

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
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frames.append(np.asarray(buf)[:, :, :3])
        plt.close(fig)

    imageio.mimsave(save_path, frames, fps=fps)


def plot_ood_scores_per_dataset(results, save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    epochs = results['config']['epochs']
    tpt_mask = results['config']['tpt_mask']
    ood_datasets = results['config']['ood_datasets']
    scorers = results['scorers']

    tpt_epochs = [e for e, m in zip(epochs, tpt_mask) if m == 1]
    tpt_start = min(tpt_epochs) if tpt_epochs else None

    for scorer_name in scorers.keys():
        fig, axes = plt.subplots(len(ood_datasets), 2, figsize=(16, 5 * len(ood_datasets)))

        if len(ood_datasets) == 1:
            axes = axes.reshape(1, -1)

        for idx, dataset_name in enumerate(ood_datasets):
            data = scorers[scorer_name][dataset_name]
            is_neco = scorer_name == 'NECO'
            style = '--o' if is_neco else '-o'
            markersize = 8 if is_neco else 6
            base_epochs = tpt_epochs if is_neco else epochs

            x_auroc = base_epochs[:len(data['auroc'])]
            x_fpr   = base_epochs[:len(data['fpr95'])]

            ax = axes[idx, 0]
            ax.plot(x_auroc, data['auroc'], style, linewidth=2, markersize=markersize)
            if tpt_start:
                ax.axvspan(tpt_start, max(epochs), alpha=0.1, color='red')
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel('AUROC', fontsize=14)
            ax.set_title(f'{scorer_name} - {dataset_name}: AUROC', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.5, 1.0])

            ax = axes[idx, 1]
            ax.plot(x_fpr, data['fpr95'], style, linewidth=2, markersize=markersize)
            if tpt_start:
                ax.axvspan(tpt_start, max(epochs), alpha=0.1, color='red')
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel('FPR@95 (%)', fontsize=14)
            ax.set_title(f'{scorer_name} - {dataset_name}: FPR@95', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, f'ood_{scorer_name}_per_dataset.png'),
                        dpi=300, bbox_inches='tight')

        plt.show()


if __name__ == "__main__":
    dummy_history = {
        'train_loss': np.linspace(2.0, 0.1, 100) + np.random.randn(100) * 0.05,
        'val_loss': np.linspace(2.0, 0.3, 100) + np.random.randn(100) * 0.08,
        'train_acc': np.linspace(10, 99, 100) + np.random.randn(100) * 2,
        'val_acc': np.linspace(10, 80, 100) + np.random.randn(100) * 3,
        'lr': [0.1 * (0.99 ** i) for i in range(100)]
    }
    plot_training_curves(dummy_history)