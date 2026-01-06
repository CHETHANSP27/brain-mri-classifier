"""
Grad-CAM Implementation for Brain Tumor MRI
Generates explainability heatmaps
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained PyTorch model
            target_layer: Layer to compute gradients (e.g., model.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
        
        Returns:
            cam: Grad-CAM heatmap (numpy array)
            predicted_class: Predicted class index
            confidence: Prediction confidence
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = F.softmax(output, dim=1)[0, target_class].item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        
        # Weighted combination of activations
        cam = (weights * activations).sum(dim=0)  # (H, W)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class, confidence
    
    def overlay_heatmap(self, image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            image: Original PIL Image or numpy array
            heatmap: Grad-CAM heatmap (H, W)
            alpha: Blending factor
            colormap: OpenCV colormap
        
        Returns:
            overlay: Overlayed image (numpy array)
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize heatmap to match image
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Convert heatmap to color
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend images
        overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlay

def apply_gradcam(model, image, transform, device, class_names):
    """
    Convenience function to apply Grad-CAM to an image
    
    Args:
        model: Trained model
        image: PIL Image
        transform: Image preprocessing transform
        device: torch device
        class_names: List of class names
    
    Returns:
        result: Dictionary with prediction, heatmap, and overlay
    """
    # Get target layer (last conv layer of ResNet)
    target_layer = model.layer4[-1]
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate CAM
    cam, pred_class, confidence = gradcam.generate_cam(input_tensor)
    
    # Create overlay
    overlay = gradcam.overlay_heatmap(image, cam)
    
    result = {
        'predicted_class': class_names[pred_class],
        'predicted_index': pred_class,
        'confidence': confidence * 100,
        'heatmap': cam,
        'overlay': overlay,
        'original': np.array(image)
    }
    
    return result