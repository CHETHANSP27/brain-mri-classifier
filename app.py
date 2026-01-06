"""
Brain Tumor MRI Classification - Streamlit App
Deployed version with Grad-CAM explainability
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from gradcam import apply_gradcam

# Page configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
MODEL_PATH = 'models/brain_tumor_resnet50.pth'
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_INFO = {
    'glioma': {
        'description': 'Gliomas are tumors that originate from glial cells in the brain.',
        'severity': 'High - Requires immediate medical attention',
        'color': '#ff6b6b'
    },
    'meningioma': {
        'description': 'Meningiomas arise from the meninges, the membranes surrounding the brain.',
        'severity': 'Moderate - Usually benign but requires monitoring',
        'color': '#ffd93d'
    },
    'pituitary': {
        'description': 'Pituitary tumors develop in the pituitary gland at the base of the brain.',
        'severity': 'Moderate - Often treatable with medication or surgery',
        'color': '#6bcf7f'
    },
    'notumor': {
        'description': 'No tumor detected in the MRI scan.',
        'severity': 'Normal - No immediate concern',
        'color': '#95e1d3'
    }
}

@st.cache_resource
def load_model():
    """Load trained model (cached)"""
    try:
        # Initialize model
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)
        )
        
        # Load weights
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        model.eval()
        
        class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])
        
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def get_transform():
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def main():
    # Header
    st.markdown('<div class="main-header">🧠 Brain Tumor MRI Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered MRI Analysis with Explainable AI</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This application uses deep learning to classify brain tumors from MRI scans.\n\n"
            "**Model**: ResNet-50\n"
            "**Classes**: Glioma, Meningioma, Pituitary, No Tumor\n"
            "**Explainability**: Grad-CAM"
        )
        
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload a brain MRI image
        2. Click 'Analyze MRI'
        3. View prediction and heatmap
        4. Review clinical information
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning(
            "This is a research tool and NOT a substitute for professional medical diagnosis. "
            "Always consult qualified healthcare professionals."
        )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Choose a brain MRI scan",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a brain MRI image in PNG or JPEG format"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded MRI', use_container_width=True)
            
            analyze_button = st.button('🔍 Analyze MRI', type='primary')
        else:
            st.info("👆 Please upload an MRI image to begin analysis")
            analyze_button = False
    
    with col2:
        st.header("📊 Analysis Results")
        
        if uploaded_file and analyze_button:
            with st.spinner('🔄 Analyzing MRI scan...'):
                try:
                    # Load model
                    model, class_names = load_model()
                    transform = get_transform()
                    
                    # Apply Grad-CAM
                    result = apply_gradcam(model, image, transform, DEVICE, class_names)
                    
                    # Display results
                    pred_class = result['predicted_class']
                    confidence = result['confidence']
                    
                    # Prediction card
                    st.success(f"✅ Analysis Complete!")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Prediction", pred_class.upper())
                    with col_res2:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Grad-CAM visualization
                    st.subheader("🔥 Grad-CAM Heatmap")
                    st.image(result['overlay'], caption='Explainability Heatmap', use_container_width=True)
                    
                    st.caption(
                        "Red regions indicate areas the AI focused on when making its prediction. "
                        "This helps radiologists understand the model's decision-making process."
                    )
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        
        elif not uploaded_file:
            st.info("Upload an MRI image and click 'Analyze MRI' to see results")
    
    # Clinical information
    if uploaded_file and analyze_button:
        st.markdown("---")
        st.header("🏥 Clinical Information")
        
        pred_class = result['predicted_class']
        info = CLASS_INFO[pred_class]
        
        col_info1, col_info2 = st.columns([2, 1])
        
        with col_info1:
            st.markdown(f"**Description**: {info['description']}")
            st.markdown(f"**Severity**: {info['severity']}")
        
        with col_info2:
            if pred_class != 'notumor':
                st.warning("⚠️ Tumor detected - Seek medical consultation")
            else:
                st.success("✅ No tumor detected")
        
        # Additional info
        with st.expander("📚 Learn More"):
            if pred_class == 'glioma':
                st.markdown("""
                **Gliomas** are the most common type of primary brain tumor. They can be:
                - Low-grade (slow-growing)
                - High-grade (aggressive)
                
                **Treatment** may include surgery, radiation, and chemotherapy.
                """)
            elif pred_class == 'meningioma':
                st.markdown("""
                **Meningiomas** are usually benign (non-cancerous) and slow-growing.
                
                **Treatment** depends on:
                - Size and location
                - Growth rate
                - Symptoms
                """)
            elif pred_class == 'pituitary':
                st.markdown("""
                **Pituitary tumors** can affect hormone production.
                
                **Symptoms** may include:
                - Vision problems
                - Hormonal imbalances
                - Headaches
                """)
            else:
                st.markdown("""
                **No tumor detected** in this scan. Regular monitoring may still be recommended
                based on your medical history and symptoms.
                """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Developed for Research Purposes | Contact your healthcare provider for medical advice"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()