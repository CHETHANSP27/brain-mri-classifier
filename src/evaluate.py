"""
Model Evaluation Script with Grad-CAM Visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import os
from gradcam import apply_gradcam

# Configuration
CONFIG = {
    'test_dir': 'data/Testing',
    'model_path': 'models/brain_tumor_resnet50.pth',
    'img_size': 224,
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_model(model_path, device):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    
    # Initialize model architecture
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4)
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])
    
    print(f"✓ Model loaded successfully")
    print(f"Classes: {class_names}")
    
    return model, class_names

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model on test set"""
    print("\nEvaluating model...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\n" + "="*60)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)
    
    return all_preds, all_labels, cm

def plot_confusion_matrix(cm, class_names):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    save_path = 'models/confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {save_path}")
    plt.close()

def visualize_predictions_with_gradcam(model, test_dataset, device, class_names, num_samples=6):
    """Visualize predictions with Grad-CAM"""
    print("\nGenerating Grad-CAM visualizations...")
    
    # Get transform without normalization for visualization
    viz_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Select random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(indices):
        # Load image
        img_path = test_dataset.imgs[sample_idx][0]
        true_label = test_dataset.imgs[sample_idx][1]
        
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        # Apply Grad-CAM
        result = apply_gradcam(model, image, viz_transform, device, class_names)
        
        # Plot
        axes[idx, 0].imshow(result['original'])
        axes[idx, 0].set_title(f"Original\nTrue: {class_names[true_label]}", fontsize=10)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(result['heatmap'], cmap='jet')
        axes[idx, 1].set_title("Grad-CAM Heatmap", fontsize=10)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(result['overlay'])
        axes[idx, 2].set_title(
            f"Prediction: {result['predicted_class']}\n"
            f"Confidence: {result['confidence']:.1f}%",
            fontsize=10,
            color='green' if result['predicted_index'] == true_label else 'red'
        )
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    save_path = 'models/gradcam_examples.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Grad-CAM visualizations saved to {save_path}")
    plt.close()

def main():
    """Main evaluation function"""
    print("="*60)
    print("Brain Tumor MRI Classification - Evaluation")
    print("="*60)
    
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}\n")
    
    # Load model
    model, class_names = load_model(CONFIG['model_path'], device)
    
    # Load test data
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(CONFIG['test_dir'], transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=4)
    
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Evaluate
    predictions, labels, cm = evaluate_model(model, test_loader, device, class_names)
    
    # Visualize with Grad-CAM
    visualize_predictions_with_gradcam(model, test_dataset, device, class_names)
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)

if __name__ == "__main__":
    main()