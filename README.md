# Chest X-Ray AI - Web Interface

A professional web application for AI-powered chest X-ray analysis with interactive visualizations and intelligent chatbot.

## Features

- 🔬 **AI-Powered Analysis**: ConvNeXt Large model trained on 230K+ images (NIH, OpenI, ReXGradient)
- 🎨 **Grad-CAM Visualizations**: Visual explanations showing which regions influenced predictions
- 🤖 **Intelligent Chatbot**: Google Gemini-powered assistant for report interpretation
- 📊 **Multi-Dataset Training**: Combined NIH ChestX-ray14, OpenI, and ReXGradient datasets
- 🎯 **14 Pathology Detection**: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Google Gemini API key (for chatbot)

## Installation

1. **Clone the repository**
```bash
git clone <website-repo-url>
cd chest-xray-website
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

4. **Download the trained model**
- Place `best_model_v3.pth` in the `models/` folder
- Or the model will auto-detect from `../Core/models/`

## Usage

### Run Locally
```bash
python app.py
```
Visit: `http://localhost:8000`

### Upload & Analyze
1. Go to "Check Model" page
2. Upload a chest X-ray (.png, .jpg, .dcm)
3. View results: predictions, confidence scores, heatmaps
4. Chat with the AI about findings

## Project Structure

```
Website/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── models/               # Trained model files
│   └── best_model_v3.pth
├── static/
│   ├── css/
│   └── cases/            # Uploaded images & results
├── templates/            # HTML templates
│   ├── home.html
│   ├── check_model.html
│   └── case.html
└── utils/               # Utilities from Core
    ├── model_utils.py
    ├── grad_cam.py
    └── preprocessing.py
```

## Chatbot Features

The AI chatbot can:
- ✅ Explain findings in the report
- ✅ Clarify confidence scores
- ✅ Reference heatmap visualizations
- ❌ **Cannot** provide medical advice or treatment recommendations (safety feature)

## API Keys

### Google Gemini (Chatbot)
1. Get a free API key: https://makersuite.google.com/app/apikey
2. Add to `.env`: `GEMINI_API_KEY=your_key_here`

## Deployment

See `HOSTING_GUIDE.md` for deployment instructions (Render, Railway, PythonAnywhere, etc.)

## Model Performance

- **Dataset**: 230K+ images (NIH: 112K, OpenI: 7.5K, ReXGradient: 160K)
- **Architecture**: ConvNeXt Large with focal loss
- **Target Metrics**: AUROC > 0.90

## Troubleshooting

### Chatbot Not Working
- Verify API key in `.env`
- Check network connectivity: run `python test_connection.py`
- Error messages will guide you

### Model Not Loading
- Ensure `best_model_v3.pth` exists in `models/` or `../Core/models/`
- Check file size (should be ~300MB for ConvNeXt Large)

## License

For educational and research purposes.

## Related Repository

- **Training Pipeline**: See the Core repository for dataset processing and model training

## Contributors

- Nikita Lotlikar - Research & ML
- Sharon Melhi - Research & ML
