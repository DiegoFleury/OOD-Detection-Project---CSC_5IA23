import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os

from src.models import ResNet18


class Trainer:
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader=None,
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Scheduler will be created in train() with correct T_max
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self):
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc='Validating')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs=200, save_dir='checkpoints', early_stopping_patience=20, checkpoint_frequency=25):

        os.makedirs(save_dir, exist_ok=True)
        
        # Set scheduler T_max based on total epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate(self.val_loader)
            
            # Test (if available)
            if self.test_loader is not None:
                _, test_acc = self.validate(self.test_loader)
            else:
                test_acc = 0.0
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            if self.test_loader is not None:
                print(f"Test Acc: {test_acc:.2f}%")
            print(f"LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                print(f"✓ New best validation accuracy: {val_acc:.2f}%")
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(
                    os.path.join(save_dir, 'resnet18_cifar100_best.pth')
                )
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint at specified frequency
            if (epoch + 1) % checkpoint_frequency == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'resnet18_cifar100_epoch{epoch+1}.pth')
                )
            
            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")
                break
        
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")
        print("=" * 50)
        
        return self.history
    
    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
        }
        
        # Only save scheduler if it exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler only if it was saved
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        print(f"Checkpoint loaded: {path}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")


def load_model(checkpoint_path, num_classes=100, device='cuda'):
    
    model = ResNet18(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    history = checkpoint.get('history', {})
    
    return model, history


if __name__ == "__main__":
    print("Trainer class ready for use!")
    print("Import with: from src.utils.training import Trainer")