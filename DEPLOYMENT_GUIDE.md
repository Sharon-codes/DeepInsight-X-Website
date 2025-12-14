# Website Deployment Guide for ConvNeXt Model

## Files Created/Updated

### 1. Created `utils/` Directory Structure
- ✅ `utils/__init__.py` - Package initialization
- ✅ `utils/model_utils.py` - ConvNeXt-compatible MultiLabelResNet
- ✅ `utils/preprocessing.py` - Image preprocessing constants
- ✅ `utils/grad_cam.py` - Heatmap generation (copied from Core)

### 2. Updated `app.py`
- ✅ Prioritized ConvNeXt Large backbone in model loading
- ✅ Fallback to other backbones if ConvNeXt fails

## Deployment Steps

### Step 1: Copy Trained Model
After training completes on HPC, copy the model weights:

```bash
# On HPC
scp ~/Sharon/IIT/Core/models/best_model_v2.pth <your-local-ip>:d:/IIT/Website/models/best_model.pth

# Or if you prefer:
scp ~/Sharon/IIT/Core/models/best_model_v2.pth <your-local-ip>:d:/IIT/Core/models/best_model.pth
```

The website will auto-detect it in either location.

### Step 2: Test Locally
```bash
cd d:\IIT\Website
python app.py
```

Visit `http://localhost:8000` and upload a chest X-ray to test.

### Step 3: Verify Output
The website should display:
- ✅ **14 pathology predictions** with confidence scores
- ✅ **14 heatmap images** (one per disease)
- ✅ **Medical report** with findings

## Troubleshooting

### Error: "Failed to load model with any known backbone"
- Check that `best_model.pth` exists in `models/` directory
- Verify the model was trained with train_v2.py (ConvNeXt)

### Error: "No module named 'utils'"
- Ensure `utils/__init__.py` exists
- Run from Website directory: `cd d:\IIT\Website`

### Heatmaps Not Generating
- Check GradCAM target layer: `base_model.features.7` for ConvNeXt
- Verify model is on correct device (CPU/CUDA)

## Architecture Compatibility

The website now supports the **exact same architecture** as train_v2.py:
- Model: ConvNeXt Large
- Input: 224x224 RGB images
- Output: 14-class multi-label predictions
- Preprocessing: ImageNet normalization

## Next Steps

Once the model is deployed:
1. Test with multiple chest X-ray images
2. Verify heatmaps highlight relevant regions
3. Check that medical reports are accurate
4. Deploy to production hosting if desired
