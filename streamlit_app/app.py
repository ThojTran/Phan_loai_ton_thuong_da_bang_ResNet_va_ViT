"""
Streamlit app for Skin Lesion Classification using ResNet18 and ViT models.
Allows users to upload images and get predictions.
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from transforms import get_resnet_transforms, IMAGENET_MEAN, IMAGENET_STD

# Define label mapping
LABEL_MAP = {
    "nv":    0,
    "mel":   1,
    "bkl":   2,
    "bcc":   3,
    "akiec": 4,
    "vasc":  5,
    "df":    6,
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

CLASS_NAMES = {
    "nv": "Nevi (Common mole)",
    "mel": "Melanoma",
    "bkl": "Benign keratosis",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratosis",
    "vasc": "Vascular lesion",
    "df": "Dermatofibroma",
}

CLASS_DESCRIPTIONS = {
    "nv": "Common, benign moles. Usually brown, symmetric, and less than 6mm.",
    "mel": "Malignant melanoma - most serious skin cancer type. Requires urgent medical attention.",
    "bkl": "Common, benign growths. Usually brown, waxy, or scaly.",
    "bcc": "Basal cell carcinoma - most common skin cancer. Usually slow-growing and treatable.",
    "akiec": "Actinic keratosis - precancerous lesion. Often rough, scaly patches.",
    "vasc": "Vascular lesions including angiomas. Usually red or purple.",
    "df": "Dermatofibroma - benign fibrous bump. Usually brownish and firm.",
}

# Set page config
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {
        background-color: #ffcccc;
        border-left: 4px solid #ff0000;
    }
    .medium-risk {
        background-color: #ffffcc;
        border-left: 4px solid #ffaa00;
    }
    .low-risk {
        background-color: #ccffcc;
        border-left: 4px solid #00aa00;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path, model_type="resnet"):
    """Load trained model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(512, len(LABEL_MAP))
    else:  # vit
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(768, len(LABEL_MAP))
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

def predict(image, model, device, model_type="resnet"):
    """Make prediction on image."""
    # Prepare image
    transform = get_resnet_transforms("val")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_class = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, pred_class].item()
    
    # Get all class probabilities
    class_probs = {
        REVERSE_LABEL_MAP[i]: probabilities[0, i].item()
        for i in range(len(LABEL_MAP))
    }
    
    return pred_class, confidence, class_probs

def get_risk_level(class_label):
    """Determine risk level based on diagnosis."""
    high_risk_classes = ["mel", "bcc", "akiec"]
    medium_risk_classes = ["bkl"]
    
    if class_label in high_risk_classes:
        return "HIGH", "🔴"
    elif class_label in medium_risk_classes:
        return "MEDIUM", "🟡"
    else:
        return "LOW", "🟢"

def display_results(pred_class, confidence, class_probs):
    """Display prediction results with visualization."""
    class_label = REVERSE_LABEL_MAP[pred_class]
    class_name = CLASS_NAMES[class_label]
    class_desc = CLASS_DESCRIPTIONS[class_label]
    
    risk_level, risk_emoji = get_risk_level(class_label)
    
    # Display main prediction
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"## {risk_emoji} Prediction Result")
        
        # Main result box
        if risk_level == "HIGH":
            style_class = "high-risk"
        elif risk_level == "MEDIUM":
            style_class = "medium-risk"
        else:
            style_class = "low-risk"
        
        st.markdown(f"""
        <div class="prediction-box {style_class}">
            <h3>{class_name.upper()}</h3>
            <p><strong>Medical Code:</strong> {class_label.upper()}</p>
            <p><strong>Risk Level:</strong> {risk_level}</p>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
            <p><strong>Description:</strong> {class_desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display probability distribution
    st.markdown("### Probability Distribution Across All Classes")
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    classes_sorted = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    classes_labels = [CLASS_NAMES[c[0]] for c in classes_sorted]
    classes_probs = [c[1] for c in classes_sorted]
    
    colors = ['#ff6b6b' if c in ["mel", "bcc", "akiec"] else '#ffd93d' if c in ["bkl"] else '#6bcf7f' 
              for c, _ in classes_sorted]
    
    bars = ax.barh(classes_labels, classes_probs, color=colors)
    ax.set_xlabel("Probability", fontsize=12)
    ax.set_title("Classification Probabilities", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, classes_probs)):
        ax.text(prob + 0.02, i, f'{prob:.2%}', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display table
    st.markdown("### Detailed Probabilities")
    prob_df = pd.DataFrame([
        {
            "Class": CLASS_NAMES[code],
            "Medical Code": code.upper(),
            "Probability": f"{prob:.4f}",
            "Percentage": f"{prob:.2%}"
        }
        for code, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    ])
    st.dataframe(prob_df, use_container_width=True, hide_index=True)
    
    # Medical disclaimer
    st.warning(
        "⚠️ **DISCLAIMER**: This is an AI-assisted classification tool for educational and screening purposes only. "
        "It should NOT be used as a substitute for professional medical diagnosis. "
        "Always consult with a dermatologist for proper diagnosis and treatment."
    )

def main():
    """Main application."""
    st.markdown('<div class="main-header">🔬 Skin Lesion Classification</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Using ResNet18 Models Trained on HAM10000 & DERM7PT</p>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_option = st.radio(
            "Select Model",
            options=["ResNet18 (HAM10000)", "ResNet18 (DERM7PT - Transfer Learning)"],
            help="Choose which trained model to use for prediction"
        )
        
        if model_option == "ResNet18 (HAM10000)":
            model_path = project_root / "checkpoints" / "resnet18_best.pth"
            model_type_key = "resnet"
            dataset_info = "Trained on HAM10000 dataset"
        else:
            model_path = project_root / "checkpoints" / "resnet18_derm7pt_best.pth"
            model_type_key = "resnet"
            dataset_info = "Fine-tuned via transfer learning on DERM7PT dataset"
        
        # Model checkpoint path
        st.subheader("Model Checkpoint")
        if model_path.exists():
            st.success(f"✅ {model_path.name}")
            st.caption(dataset_info)
        else:
            st.error(f"❌ Model not found: {model_path.name}")
        
        # Info section
        st.divider()
        st.subheader("ℹ️ About")
        st.markdown("""
        This application classifies skin lesions into 7 categories:
        - **Nevi (nv)**: Common benign moles
        - **Melanoma (mel)**: Malignant skin cancer
        - **Benign keratosis (bkl)**: Common benign growths
        - **Basal cell carcinoma (bcc)**: Common skin cancer
        - **Actinic keratosis (akiec)**: Precancerous lesion
        - **Vascular lesion (vasc)**: Blood vessel-related
        - **Dermatofibroma (df)**: Benign fibrous bump
        
        **Available Models:**
        - **HAM10000**: Pre-trained on 10,000 images
        - **DERM7PT**: Transfer learning fine-tuned model
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📚 About Dataset", "📖 Guidelines"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Upload a dermatological image for classification"
            )
        
        with col2:
            st.subheader("Or Use Example")
            use_example = st.checkbox("Use example image")
        
        # Process image
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.success("✅ Image uploaded successfully!")
        elif use_example:
            # Create a simple example image (placeholder)
            st.info("Example mode: Using a placeholder image")
            image = Image.new("RGB", (224, 224), color=(100, 150, 200))
        
        if image is not None:
            # Display uploaded image
            st.markdown("### Preview")
            col_img1, col_img2 = st.columns([1, 2])
            
            with col_img1:
                st.image(image, use_column_width=True, caption="Uploaded Image")
            
            with col_img2:
                st.info("Image ready for classification. Click 'Predict' to get results.")
            
            # Prediction button
            if st.button("🔍 Predict", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing image using {model_option}..."):
                    try:
                        # Try to load and use model
                        if model_path.exists():
                            model, device = load_model(str(model_path), model_type_key)
                            if model is not None:
                                pred_class, confidence, class_probs = predict(image, model, device, model_type_key)
                                display_results(pred_class, confidence, class_probs)
                        else:
                            st.error(f"❌ Model checkpoint not found: {model_path.name}")
                            st.info(f"Expected location: `{model_path}`")
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        import traceback
                        st.text(traceback.format_exc())
    
    with tab2:
        st.subheader("Dataset Information")
        st.markdown("""
        ### HAM10000 Dataset
        - **Total Images**: ~10,000 dermatological images
        - **Classes**: 7 types of skin lesions
        - **Source**: International Skin Imaging Collaboration
        - **Usage**: Primary training dataset
        
        ### DERM7PT Dataset
        - **Total Images**: ~2,750 images from 75 patients
        - **Classes**: 7 skin lesion types (same as HAM10000)
        - **Source**: Another dermatological dataset
        - **Usage**: Transfer learning / fine-tuning dataset
        
        ### Transfer Learning Approach
        - **Step 1**: Train ResNet18 on HAM10000 (source domain)
        - **Step 2**: Fine-tune on DERM7PT (target domain)
        - **Result**: Improved generalization to new dermatological images
        
        ### Model Comparison
        """)
        
        # Display model comparison
        comparison_data = {
            "Model": ["ResNet18 (HAM10000)", "ResNet18 (DERM7PT)"],
            "Training Data": ["HAM10000 (~10K images)", "Transfer from HAM10000 + Fine-tune on DERM7PT"],
            "Use Case": ["General skin lesion classification", "Domain-specific refinement"],
            "File": ["resnet18_best.pth", "resnet18_derm7pt_best.pth"]
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
        
        # Class distribution
        st.markdown("### 7 Skin Lesion Classes")
        dist_data = {
            "Class": [CLASS_NAMES[k] for k in LABEL_MAP.keys()],
            "Code": list(LABEL_MAP.keys()),
            "Risk Level": ["🟢 LOW", "🔴 HIGH", "🟡 MEDIUM", "🔴 HIGH", "🔴 HIGH", "🟢 LOW", "🟢 LOW"]
        }
        st.dataframe(pd.DataFrame(dist_data), use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("📋 Classification Guidelines")
        st.markdown("""
        ### ABCDE Rule for Melanoma Detection
        - **A - Asymmetry**: One half unlike the other
        - **B - Border**: Irregular or scalloped border
        - **C - Color**: Variable color across the lesion
        - **D - Diameter**: Larger than 6mm (pencil eraser)
        - **E - Evolving**: Changes in size, shape, or color
        
        ### When to Seek Medical Attention
        - Any new or changing skin lesion
        - Lesions that bleed or ooze
        - Itching or pain
        - Lesions that don't heal
        - Family history of skin cancer
        
        ### About the Models
        - **ResNet18**: A deep convolutional neural network with 18 layers
        - **Transfer Learning**: Leverages knowledge from HAM10000 to improve accuracy on DERM7PT
        - **Fine-tuning**: Adapts pre-trained weights to new domain for better performance
        
        ### Important Notes
        - This tool is for educational and screening purposes only
        - Always consult a dermatologist for professional diagnosis
        - AI predictions can be incorrect - professional evaluation is essential
        - Regular skin checks are recommended, especially for high-risk individuals
        - Risk levels are based on typical characteristics, not diagnosis certainty
        """)

if __name__ == "__main__":
    main()
