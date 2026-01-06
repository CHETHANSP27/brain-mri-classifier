"""
Brain Tumor MRI Classification - Training Script
Python 3.13 Compatible
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Configuration
CONFIG = {
    'data_dir': 'data/Training',
    'test_dir': 'data/Testing',
    'model_save_path': 'models/brain_tumor_resnet50.pth',
    'batch_size': 32,
    'num_epochs': 25,
    'learning_rate': 0.001,
    'img_size': 224,
    'num_classes': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class BrainTumorClassifier:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        print(f"Using device: {self.device}")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Data transforms
        self.train_transforms = transforms.Compose([
            transforms.Resize((config['img_size'], config['img_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.test_transforms = transforms.Compose([
            transforms.Resize((config['img_size'], config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
    def _build_model(self):
        """Build ResNet50 model with custom classifier"""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze early layers
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.config['num_classes'])
        )
        
        return model.to(self.device)
    
    def load_data(self):
        """Load training and validation data"""
        print("Loading datasets...")
        
        train_dataset = datasets.ImageFolder(
            self.config['data_dir'], 
            transform=self.train_transforms
        )
        
        test_dataset = datasets.ImageFolder(
            self.config['test_dir'], 
            transform=self.test_transforms
        )
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.class_names = train_dataset.classes
        print(f"Classes: {self.class_names}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return running_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return running_loss / len(self.test_loader), 100. * correct / total
    
    def train(self):
        """Full training loop"""
        print("\nStarting training...\n")
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': self.class_names
                }, self.config['model_save_path'])
                print(f"✓ Model saved! (Val Acc: {val_acc:.2f}%)")
        
        training_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Training completed in {training_time/60:.2f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"{'='*50}\n")
        
        self.plot_history(history)
        return history
    
    def plot_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        print("Training plots saved to models/training_history.png")
        plt.close()

def main():
    """Main training function"""
    print("="*60)
    print("Brain Tumor MRI Classification - Training")
    print("="*60)
    
    # Initialize classifier
    classifier = BrainTumorClassifier(CONFIG)
    
    # Load data
    classifier.load_data()
    
    # Train model
    history = classifier.train()
    
    print("\n✓ Training complete! Model saved to:", CONFIG['model_save_path'])
    print("\nTo evaluate the model, run: python src/evaluate.py")

if __name__ == "__main__":
    main()