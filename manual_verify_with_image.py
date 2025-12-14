import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import sys
import os

# Setup paths
sys.path.insert(0, "D:/IIT/Core") # Use Core utils
sys.path.insert(0, "D:/IIT/Website") # Use Website utils

try:
    from utils.model_utils import MultiLabelResNet
    from utils.grad_cam import GradCAM
    from utils.preprocessing import TARGET_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

TARGET_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

def run_test():
    print("=== MANUAL MODEL VERIFICATION ===")
    
    # 1. Define Paths
    MODEL_PATH = "D:/IIT/Website/models/best_model.pth"
    # Create a dummy image if no real one exists, or use a specific one if known.
    # We will create a dummy image for this test.
    TEST_IMAGE_PATH = r"D:\IIT\Dataset\test_image\00000001_000.png"
    OUTPUT_HEATMAP_PATH = "D:/IIT/Website/manual_heatmap_real.png"
    
    # Create dummy image (gray gradient) - DISABLED
    # img_data = np.zeros((512, 512), dtype=np.uint8)
    # for i in range(512):
    #     img_data[i, :] = i // 2
    # cv2.imwrite(TEST_IMAGE_PATH, img_data)
    # print(f"[OK] Created dummy X-ray at {TEST_IMAGE_PATH}")

    # 2. Load Model
    device = torch.device('cpu') # Use CPU for safety
    print(f"Loading model from {MODEL_PATH}...")
    
    try:
        num_classes = len(TARGET_PATHOLOGIES)
        model = MultiLabelResNet(num_classes=num_classes, backbone='convnext_large', pretrained=False)
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        model.to(device)
        print("[OK] Model loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        return

    # 3. Setup Inference
    transform = transforms.Compose([
        transforms.Resize(TARGET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # 4. Run Inference
    try:
        image = Image.open(TEST_IMAGE_PATH).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze(0).cpu().numpy()
            
        print("\n=== PREDICTIONS ===")
        top_indices = np.argsort(probs)[::-1][:3]
        for idx in top_indices:
            print(f"{TARGET_PATHOLOGIES[idx]}: {probs[idx]:.4f}")
            
    except Exception as e:
        print(f"[FAIL] Inference failed: {e}")
        return

    # 5. Generate Heatmap
    try:
        gradcam = GradCAM(model, 'base_model.features.7')
        top_idx = top_indices[0] # Top prediction
        print(f"\nGeneratig heatmap for: {TARGET_PATHOLOGIES[top_idx]}")
        
        heatmap = gradcam.generate_heatmap(input_tensor, top_idx)
        overlaid = gradcam.overlay_heatmap(TEST_IMAGE_PATH, heatmap)
        
        cv2.imwrite(OUTPUT_HEATMAP_PATH, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
        print(f"[OK] Heatmap saved to {OUTPUT_HEATMAP_PATH}")
        print("You can open this file to see the result.")
        
    except Exception as e:
        print(f"[FAIL] Heatmap generation failed: {e}")

if __name__ == "__main__":
    run_test()
