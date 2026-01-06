"""
Utility functions for Brain Tumor MRI Classification
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    """Print model architecture summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*60 + "\n")

def plot_sample_images(dataset, class_names, num_samples=12):
    """Plot sample images from dataset"""
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx, ax in zip(indices, axes):
        img, label = dataset[idx]
        
        # Denormalize if needed
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(class_names[label])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/sample_images.png', dpi=200)
    print("Sample images saved to models/sample_images.png")
    plt.close()

def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    
    print(f"Checkpoint loaded from {path}")
    print(f"Epoch: {epoch}, Val Acc: {val_acc:.2f}%")
    
    return model, optimizer, epoch, val_acc

def get_class_distribution(dataset):
    """Get class distribution in dataset"""
    labels = [label for _, label in dataset]
    unique, counts = np.unique(labels, return_counts=True)
    
    print("\nClass Distribution:")
    print("-" * 30)
    for cls, count in zip(unique, counts):
        print(f"{dataset.classes[cls]}: {count} ({count/len(labels)*100:.1f}%)")
    print("-" * 30)
    
    return dict(zip(unique, counts))

def create_directory_structure():
    """Create project directory structure"""
    directories = [
        'data/Training',
        'data/Testing',
        'models',
        'notebooks',
        'src'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure created successfully!")

if __name__ == "__main__":
    # Create directories if running utils directly
    create_directory_structure()