"""
Streamlit app for Skin Lesion Classification using ResNet18 and ViT models.
Supports Grad-CAM visualization for both ResNet18 and ViT-B/16.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Path setup TRƯỚC mọi local import ──────────────────────────────────────
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
# ───────────────────────────────────────────────────────────────────────────

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import re

try:
    from transforms import get_resnet_transforms
except ModuleNotFoundError:
    def get_resnet_transforms(split="val"):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

# ---------------------------------------------------------------------------
# LABEL MAPS & CLASS INFO
# ---------------------------------------------------------------------------

HAM10000_LABELS = {
    "nv": 0, "mel": 1, "bkl": 2, "bcc": 3,
    "akiec": 4, "vasc": 5, "df": 6,
}

DERM7PT_LABELS = {
    "bcc": 0, "mel": 1, "misc": 2, "nevus": 3, "sk": 4,
}

CLASS_NAMES = {
    "nv":    "Nevi (Common mole)",
    "mel":   "Melanoma",
    "bkl":   "Benign keratosis",
    "bcc":   "Basal cell carcinoma",
    "akiec": "Actinic keratosis",
    "vasc":  "Vascular lesion",
    "df":    "Dermatofibroma",
    "misc":  "Miscellaneous",
    "nevus": "Nevus",
    "sk":    "Seborrheic keratosis",
}

CLASS_DESCRIPTIONS = {
    "nv":    "Common, benign moles. Usually brown, symmetric, and less than 6mm.",
    "mel":   "Malignant melanoma — most serious skin cancer type. Requires urgent medical attention.",
    "bkl":   "Common, benign growths. Usually brown, waxy, or scaly.",
    "bcc":   "Basal cell carcinoma — most common skin cancer. Usually slow-growing and treatable.",
    "akiec": "Actinic keratosis — precancerous lesion. Often rough, scaly patches.",
    "vasc":  "Vascular lesions including angiomas. Usually red or purple.",
    "df":    "Dermatofibroma — benign fibrous bump. Usually brownish and firm.",
    "misc":  "Miscellaneous skin lesions that do not fit the main categories.",
    "nevus": "Nevus — Common, benign moles.",
    "sk":    "Seborrheic keratosis — Common, non-cancerous skin growth.",
}

HIGH_RISK   = {"mel", "bcc", "akiec"}
MEDIUM_RISK = {"bkl", "sk", "misc"}

# ---------------------------------------------------------------------------
# PAGE CONFIG & CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-box { padding: 20px; border-radius: 10px; margin: 10px 0; }
    .high-risk      { background-color: #ffcccc; border-left: 4px solid #ff0000; }
    .medium-risk    { background-color: #ffffcc; border-left: 4px solid #ffaa00; }
    .low-risk       { background-color: #ccffcc; border-left: 4px solid #00aa00; }
    .gradcam-note   { font-size: 0.85em; color: #555; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def _remap_timm_vit_keys(state_dict: dict) -> dict:
    """
    Remap timm ViT key names → torchvision vit_b_16 key names.
    Called only when checkpoint was saved from a timm model.
    """
    remap = {}
    for k, v in state_dict.items():
        k = k.replace("patch_embed.proj.",   "conv_proj.")
        k = k.replace("cls_token",           "class_token")
        k = k.replace("pos_embed",           "encoder.pos_embedding")
        k = re.sub(
            r"^blocks\.(\d+)\.",
            lambda m: f"encoder.layers.encoder_layer_{m.group(1)}.",
            k,
        )
        k = k.replace(".norm1.",       ".ln_1.")
        k = k.replace(".norm2.",       ".ln_2.")
        k = k.replace(".attn.proj.",   ".self_attention.out_proj.")
        k = k.replace(".mlp.fc1.",     ".mlp.0.")
        k = k.replace(".mlp.fc2.",     ".mlp.3.")
        k = k.replace("norm.",         "encoder.ln.")
        k = k.replace("head.",         "heads.head.")
        remap[k] = v
    return remap


@st.cache_resource
def load_model(model_path: str, model_type: str = "resnet", dataset_type: str = "ham10000"):
    """Load a trained ResNet18 or ViT-B/16 checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = HAM10000_LABELS if dataset_type == "ham10000" else DERM7PT_LABELS
    num_classes = len(label_map)

    # Build architecture
    if model_type == "resnet":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:  # vit
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    try:
        raw = torch.load(model_path, map_location=device)

        # Unwrap checkpoint wrappers
        if isinstance(raw, dict):
            for wrapper_key in ("model_state", "model_state_dict", "state_dict"):
                if wrapper_key in raw:
                    raw = raw[wrapper_key]
                    break

        # Strip common prefixes
        clean = {}
        for k, v in raw.items():
            for prefix in ("model.", "features."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            clean[k] = v

        # Remap timm → torchvision keys for ViT
        if model_type == "vit" and "patch_embed.proj.weight" in clean:
            clean = _remap_timm_vit_keys(clean)

        missing, unexpected = model.load_state_dict(clean, strict=False)
        if missing:
            st.warning(
                f"⚠️ Missing keys ({len(missing)}): "
                f"{missing[:3]}{'…' if len(missing) > 3 else ''}"
            )
        if unexpected:
            st.warning(
                f"⚠️ Unexpected keys ({len(unexpected)}): "
                f"{unexpected[:3]}{'…' if len(unexpected) > 3 else ''}"
            )

        model.to(device).eval()
        return model, device

    except Exception as e:
        import traceback
        st.error(f"Error loading model: {e}")
        st.code(traceback.format_exc(), language="bash")
        return None, device

# ---------------------------------------------------------------------------
# GRAD-CAM IMPLEMENTATIONS
# ---------------------------------------------------------------------------

class GradCAMResNet:
    """Grad-CAM for ResNet18 — hooks into layer4[-1]."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._activations = None
        self._gradients   = None
        target = model.layer4[-1]
        self._fh = target.register_forward_hook(self._save_act)
        self._bh = target.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _i, output):
        self._activations = output.detach()

    def _save_grad(self, _m, _gi, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1).squeeze())
        cam = cam.cpu().numpy()
        return cam / cam.max() if cam.max() > 0 else cam

    def remove_hooks(self):
        self._fh.remove()
        self._bh.remove()


class GradCAMViT:
    """
    Grad-CAM for ViT-B/16 (torchvision) — hooks into encoder.layers[-1].
    Spatial tokens (excluding CLS at index 0) are reshaped to a 2-D grid.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._activations = None
        self._gradients   = None
        target = model.encoder.layers[-1]
        self._fh = target.register_forward_hook(self._save_act)
        self._bh = target.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _i, output):
        self._activations = output.detach()   # (B, N_tokens, D)

    def _save_grad(self, _m, _gi, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()

        acts  = self._activations[0, 1:, :]   # drop CLS token
        grads = self._gradients[0, 1:, :]
        weights = grads.mean(dim=-1, keepdim=True)
        cam_tokens = torch.relu((weights * acts).sum(dim=-1))

        n = int(cam_tokens.shape[0] ** 0.5)
        cam_2d = cam_tokens.cpu().numpy().reshape(n, n)
        return cam_2d / cam_2d.max() if cam_2d.max() > 0 else cam_2d

    def remove_hooks(self):
        self._fh.remove()
        self._bh.remove()


def get_gradcam(model: nn.Module, model_type: str):
    return GradCAMResNet(model) if model_type == "resnet" else GradCAMViT(model)


def overlay_heatmap(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    orig_np = np.array(original_image.convert("RGB"))
    h, w = orig_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_rgb = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    return (alpha * heatmap_rgb + (1 - alpha) * orig_np).astype(np.uint8)

# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

def predict(image, model, device, dataset_type="ham10000"):
    label_map = HAM10000_LABELS if dataset_type == "ham10000" else DERM7PT_LABELS
    rev = {v: k for k, v in label_map.items()}
    tensor = get_resnet_transforms("val")(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    pred_idx   = probs.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()
    class_probs = {rev[i]: probs[0, i].item() for i in range(len(label_map))}
    return pred_idx, confidence, class_probs


def predict_gradcam(image, model, device, model_type, dataset_type="ham10000"):
    label_map = HAM10000_LABELS if dataset_type == "ham10000" else DERM7PT_LABELS
    rev = {v: k for k, v in label_map.items()}
    tensor = get_resnet_transforms("val")(image).unsqueeze(0).to(device).requires_grad_(True)

    model.train()   # gradients need to flow through BN
    gcam = get_gradcam(model, model_type)
    try:
        output     = model(tensor)
        probs      = torch.softmax(output.detach(), dim=1)
        pred_idx   = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
        class_probs = {rev[i]: probs[0, i].item() for i in range(len(label_map))}
        cam = gcam(tensor, pred_idx)
    finally:
        gcam.remove_hooks()
        model.eval()

    return pred_idx, confidence, class_probs, cam   # return raw CAM, not overlay

# ---------------------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------------------

def get_risk_level(label: str):
    if label in HIGH_RISK:   return "HIGH",   "🔴"
    if label in MEDIUM_RISK: return "MEDIUM", "🟡"
    return "LOW", "🟢"


def display_results(pred_class, confidence, class_probs, dataset_type,
                    cam=None, original_image=None, alpha=0.5):
    label_map = HAM10000_LABELS if dataset_type == "ham10000" else DERM7PT_LABELS
    rev = {v: k for k, v in label_map.items()}
    class_label = rev[pred_class]
    risk_level, risk_emoji = get_risk_level(class_label)
    style_class = {"HIGH": "high-risk", "MEDIUM": "medium-risk", "LOW": "low-risk"}[risk_level]

    st.markdown("---")

    # Prediction card
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"## {risk_emoji} Prediction Result")
        st.markdown(f"""
        <div class="prediction-box {style_class}">
            <h3>{CLASS_NAMES[class_label].upper()}</h3>
            <p><strong>Medical Code:</strong> {class_label.upper()}</p>
            <p><strong>Risk Level:</strong> {risk_level}</p>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
            <p><strong>Description:</strong> {CLASS_DESCRIPTIONS[class_label]}</p>
        </div>
        """, unsafe_allow_html=True)

    # Grad-CAM
    if cam is not None and original_image is not None:
        overlay = overlay_heatmap(original_image, cam, alpha=alpha)
        st.markdown("### 🔥 Grad-CAM — Region Attention Map")
        st.markdown(
            "<p class='gradcam-note'>Red/warm = high attention · Blue/cool = low attention</p>",
            unsafe_allow_html=True,
        )
        gc1, gc2 = st.columns(2)
        with gc1:
            st.image(original_image, caption="Original Image",   use_container_width=True)
        with gc2:
            st.image(overlay,        caption="Grad-CAM Overlay", use_container_width=True)

        fig_cb, ax_cb = plt.subplots(figsize=(5, 0.35))
        fig_cb.patch.set_alpha(0)
        cbar = plt.colorbar(cm.ScalarMappable(cmap="jet"), cax=ax_cb, orientation="horizontal")
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["Low attention", "Medium", "High attention"])
        ax_cb.tick_params(labelsize=8)
        st.pyplot(fig_cb, use_container_width=False)

    # Probability bar chart
    st.markdown("### Probability Distribution Across All Classes")
    classes_sorted = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    labels_ = [CLASS_NAMES[c] for c, _ in classes_sorted]
    probs_  = [p              for _, p in classes_sorted]
    colors_ = [
        "#ff6b6b" if c in HIGH_RISK else "#ffd93d" if c in MEDIUM_RISK else "#6bcf7f"
        for c, _ in classes_sorted
    ]
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels_) * 0.6)))
    bars = ax.barh(labels_, probs_, color=colors_)
    ax.set_xlabel("Probability", fontsize=12)
    ax.set_title("Classification Probabilities", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.12)
    for bar, p in zip(bars, probs_):
        ax.text(p + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{p:.2%}", va="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    # Detail table
    st.markdown("### Detailed Probabilities")
    st.dataframe(pd.DataFrame([
        {"Class": CLASS_NAMES[c], "Medical Code": c.upper(),
         "Probability": f"{p:.4f}", "Percentage": f"{p:.2%}"}
        for c, p in classes_sorted
    ]), use_container_width=True, hide_index=True)

    st.warning(
        "⚠️ **DISCLAIMER**: AI-assisted screening tool only. "
        "Always consult a dermatologist for proper diagnosis and treatment."
    )

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    st.markdown('<div class="main-header">🔬 Skin Lesion Classification</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:gray;'>"
        "ResNet18 &amp; ViT-B/16 — HAM10000 &amp; DERM7PT — with Grad-CAM</p>",
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        MODEL_OPTIONS = {
            "ResNet18 — HAM10000":           ("resnet18_best.pth",         "resnet", "ham10000"),
            "ResNet18 — DERM7PT (Transfer)": ("resnet18_derm7pt_best.pth", "resnet", "derm7pt"),
            "ViT-B/16 — HAM10000":           ("vit_ham10000_best.pth",     "vit",    "ham10000"),
            "ViT-B/16 — DERM7PT (Transfer)": ("vit_derm7pt_best.pth",      "vit",    "derm7pt"),
        }

        model_option = st.radio("Select Model", list(MODEL_OPTIONS.keys()))
        ckpt_name, model_type_key, dataset_type = MODEL_OPTIONS[model_option]
        model_path = project_root / "checkpoints" / ckpt_name

        st.subheader("Model Checkpoint")
        if model_path.exists():
            st.success(f"✅ {ckpt_name}")
        else:
            st.error(f"❌ Not found: {ckpt_name}")

        st.divider()

        enable_gradcam = st.checkbox(
            "🔥 Enable Grad-CAM", value=True,
            help="Overlay heatmap showing which regions drove the prediction.",
        )
        gradcam_alpha = st.slider(
            "Heatmap opacity", 0.2, 0.9, 0.5, 0.05,
            disabled=not enable_gradcam,
        )

        st.divider()
        st.subheader("ℹ️ About")
        st.markdown("""
**Models**
- **ResNet18** – CNN backbone, fast & robust
- **ViT-B/16** – Vision Transformer, global context

**Datasets**
- **HAM10000** – 10 000 images, 7 classes
- **DERM7PT** – 2 750 images, 5 classes (fine-tuned)

**Grad-CAM**
Highlights image regions most relevant to the prediction.
        """)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📚 About Datasets", "📖 Guidelines"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a dermatological image",
                type=["jpg", "jpeg", "png", "bmp"],
            )
        with c2:
            st.subheader("Or Use Example")
            use_example = st.checkbox("Use placeholder example image")

        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.success("✅ Image uploaded successfully!")
        elif use_example:
            st.info("Example mode: synthetic 224×224 image.")
            image = Image.new("RGB", (224, 224), color=(120, 80, 60))

        if image is not None:
            st.markdown("### Preview")
            p1, p2 = st.columns([1, 2])
            with p1:
                st.image(image, use_container_width=True, caption="Input Image")
            with p2:
                st.info("Click **Predict** to classify and (optionally) visualise Grad-CAM.")

            if st.button("🔍 Predict"):
                if not model_path.exists():
                    st.error(f"❌ Checkpoint not found: {model_path}")
                else:
                    with st.spinner(f"Loading {model_option} and analysing…"):
                        try:
                            model, device = load_model(
                                str(model_path), model_type_key, dataset_type
                            )
                            if model is None:
                                st.stop()

                            if enable_gradcam:
                                pred_class, confidence, class_probs, cam = predict_gradcam(
                                    image, model, device, model_type_key, dataset_type
                                )
                                display_results(
                                    pred_class, confidence, class_probs,
                                    dataset_type, cam=cam,
                                    original_image=image, alpha=gradcam_alpha,
                                )
                            else:
                                pred_class, confidence, class_probs = predict(
                                    image, model, device, dataset_type
                                )
                                display_results(
                                    pred_class, confidence, class_probs, dataset_type
                                )

                        except Exception as e:
                            import traceback
                            st.error(f"Error during prediction: {e}")
                            st.code(traceback.format_exc())

    with tab2:
        st.subheader("Dataset Information")
        st.markdown("""
### HAM10000
| Property | Value |
|---|---|
| Total images | ~10 000 |
| Classes | 7 |
| Source | ISIC Archive |

**Classes:** nv · mel · bkl · bcc · akiec · vasc · df

---

### DERM7PT
| Property | Value |
|---|---|
| Total images | ~2 750 |
| Patients | 75 |
| Classes | 5 |
| Source | University of Graz |

**Classes:** bcc · mel · misc · nevus · sk
        """)

    with tab3:
        st.subheader("📋 Classification Guidelines")
        st.markdown("""
### ABCDE Rule for Melanoma Detection
| Letter | Criterion | Warning sign |
|---|---|---|
| **A** | Asymmetry | One half unlike the other |
| **B** | Border | Irregular, ragged, or blurred |
| **C** | Color | Multiple shades of brown/black, or red/white/blue |
| **D** | Diameter | Larger than 6 mm (pencil eraser) |
| **E** | Evolving | Changes in size, shape, or color over time |

### How to Read Grad-CAM
- 🔴 **Warm colours (red/orange)** — regions with highest influence on prediction
- 🔵 **Cool colours (blue)** — regions with low or negative influence

### Risk Levels
- 🔴 **HIGH** — mel, bcc, akiec → consult a dermatologist urgently
- 🟡 **MEDIUM** — bkl, sk, misc → monitor; seek advice if in doubt
- 🟢 **LOW** — nv, vasc, df → typically benign; regular checks advised
        """)


if __name__ == "__main__":
    main()