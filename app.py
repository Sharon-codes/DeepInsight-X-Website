import os
import sys
import uuid
import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import base64
import requests

# Hugging Face API for multimodal chatbot (optional)
HF_API_KEY = os.environ.get('HF_API_KEY', '')
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else None
if not HF_API_KEY:
    # Avoid using app.logger before app exists
    print('Warning: HF_API_KEY not set. AI chatbot will use fallback.')


# Make Core/ importable
BASE_DIR = Path(__file__).resolve().parent
CORE_DIR = (BASE_DIR.parent / 'Core').resolve()
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

# Small static list of target pathologies used by templates
TARGET_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# App and folders
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'change-me-in-prod')

STATIC_DIR = BASE_DIR / 'static'
UPLOAD_ROOT = STATIC_DIR / 'uploads'
RESULTS_ROOT = STATIC_DIR / 'results'
CASES_ROOT = STATIC_DIR / 'cases'
for p in [UPLOAD_ROOT, RESULTS_ROOT, CASES_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'dcm', 'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE_MB = int(os.environ.get('MAX_FILE_SIZE_MB', '50'))
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024

# Device and model loading (default to CPU for portability; set FORCE_CPU=0 to allow CUDA)
USE_CPU = os.environ.get('FORCE_CPU', '1') == '1'
# DEVICE will be resolved when torch is imported during model load
DEVICE = None

def _resolve_model_path():
    candidates = [
        BASE_DIR / 'models' / 'best_model_v3.pth',  # Latest trained model
        BASE_DIR / 'models' / 'best_model.pth',
        BASE_DIR / 'models' / 'best_model_v2.pth',
        CORE_DIR / 'models' / 'best_model_v3.pth',
        CORE_DIR / 'models' / 'best_model.pth',
        CORE_DIR / 'models' / 'best_model_v2.pth',
        CORE_DIR / 'models' / 'final_model.pth',
        CORE_DIR / 'models' / 'latest_model.pth',
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

MODEL = None
GRADCAM = None
INFER_TRANSFORM = None
CLIP_MODEL = None
CLIP_PROCESSOR = None
CLIP_DEVICE = None
GEN_MODEL = None
GEN_TOKENIZER = None
GEN_DEVICE = None
GEN_MODEL_NAME = os.environ.get('GEN_MODEL_NAME', 'google/flan-t5-base')

def load_model_once():
    global MODEL, GRADCAM, INFER_TRANSFORM
    if MODEL is not None:
        return
    try:
        # Lazy-import heavy ML dependencies here
        import torch
        import torchvision.transforms as transforms
        from utils.model_utils import MultiLabelResNet  # type: ignore
        from utils.grad_cam import GradCAM  # type: ignore
        from utils.preprocessing import TARGET_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD  # type: ignore

        # Resolve device now that torch is available
        global DEVICE
        DEVICE = torch.device('cpu' if USE_CPU or not torch.cuda.is_available() else 'cuda')

        model_path = _resolve_model_path()
        if model_path is None:
            app.logger.error('No model checkpoint found in Core/models (expected best_model.pth or final_model.pth).')
            return
        num_classes = len(TARGET_PATHOLOGIES)
        candidates = [
            ('convnext_large', 'base_model.features.7'),  # Try ConvNeXt first (matches train_v2.py)
            ('resnext101_32x8d', 'base_model.layer4'),
            ('resnet101', 'base_model.layer4'),
            ('efficientnet_b4', 'base_model.features.6'),
        ]
        state = torch.load(str(model_path), map_location=DEVICE)
        last_error = None
        for backbone, target_layer in candidates:
            try:
                m = MultiLabelResNet(num_classes=num_classes, backbone=backbone, pretrained=False)
                m.load_state_dict(state, strict=False)
                m.eval(); m.to(DEVICE)
                g = GradCAM(m, target_layer)
                t = transforms.Compose([
                    transforms.Resize(TARGET_IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                globals()['MODEL'] = m
                globals()['GRADCAM'] = g
                globals()['INFER_TRANSFORM'] = t
                app.logger.info(f"Model loaded with backbone {backbone}: {model_path.name} on {DEVICE}")
                break
            except Exception as e:
                last_error = e
                app.logger.debug(f"Failed to load with backbone {backbone}: {e}")
                continue
        if globals()['MODEL'] is None:
            raise RuntimeError(f"Failed to load model with any known backbone. Last error: {last_error}")
    except Exception:
        app.logger.exception('Error loading model or GradCAM')
        # Keep globals as None if failure


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------- Public pages ----------
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/__status')
def status():
    model_path = _resolve_model_path()
    return jsonify({
        "model_path": str(model_path) if model_path else None,
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "gradcam_ready": GRADCAM is not None
    })


@app.route('/__load')
def force_load():
    load_model_once()
    return status()


@app.route('/problem')
def problem():
    return render_template('problem.html')


@app.route('/solution')
def solution():
    return render_template('solution.html', targets=TARGET_PATHOLOGIES)


@app.route('/impact')
def impact():
    return render_template('impact.html')

@app.route('/team')
def team():
    members = [
        {"name": "Nikita Lotlikar", "role": "Research & ML"},
        {"name": "Sharon Melhi", "role": "Research & ML"},
    ]
    return render_template('team.html', members=members)


@app.route('/check-model', methods=['GET'])
def check_model():
    return render_template('check_model.html', max_mb=MAX_FILE_SIZE_MB)


# ---------- Inference flow ----------
@app.route('/analyze', methods=['POST'])
def analyze():
    # Attempt to load model (lazy). If loading fails, we proceed with a safe
    # fallback so the web UI still returns a report and a demonstrative heatmap
    # instead of hard-failing. This keeps the site usable on machines without
    # a working PyTorch installation.
    try:
        load_model_once()
    except Exception as e:
        app.logger.warning(f'Model loading skipped or failed: {e}')

    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('check_model'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('check_model'))

    if not allowed_file(file.filename):
        flash(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
        return redirect(url_for('check_model'))

    # Create a case directory
    case_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_' + uuid.uuid4().hex[:8]
    case_dir = CASES_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # Save original upload
    filename = secure_filename(file.filename)
    orig_path = case_dir / filename
    file.save(str(orig_path))

    # Normalize to PNG in case_dir
    processed_png = case_dir / f"processed_{orig_path.stem}.png"
    try:
        if filename.lower().endswith('.dcm'):
            # Lightweight DICOM -> PNG conversion here to avoid importing the
            # full preprocessing module (which pulls in torch). This keeps the
            # upload path lightweight and avoids hard failures if torch isn't present.
            try:
                import pydicom
                dicom_data = pydicom.dcmread(str(orig_path))
                pixel_array = dicom_data.pixel_array
                # Apply VOI LUT if available
                try:
                    from pydicom.pixel_data_handlers.util import apply_voi_lut
                    pixel_array = apply_voi_lut(pixel_array, dicom_data)
                except Exception:
                    pass
                if getattr(dicom_data, 'PhotometricInterpretation', '') == 'MONOCHROME1':
                    pixel_array = pixel_array.max() - pixel_array
                if pixel_array.dtype != np.uint8:
                    pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img = Image.fromarray(pixel_array)
                img.save(str(processed_png))
            except Exception as e:
                raise RuntimeError(f'DICOM conversion failed: {e}')
        else:
            img = Image.open(orig_path).convert('RGB')
            img.save(str(processed_png))
    except Exception as e:
        app.logger.exception('Failed to preprocess image')
        flash(f'Preprocessing failed: {e}')
        return redirect(url_for('check_model'))

    # Inference (real model) or fallback (placeholder)
    heatmap_files = []
    pred_names = []
    confidences = {}

    if MODEL is not None and GRADCAM is not None and INFER_TRANSFORM is not None:
        try:
            # Real inference
            import torch as _torch
            image = Image.open(processed_png).convert('RGB')
            input_tensor = INFER_TRANSFORM(image).unsqueeze(0).to(DEVICE)
            with _torch.no_grad():
                outputs = MODEL(input_tensor)
                probs = _torch.sigmoid(outputs).squeeze(0).cpu().numpy()

            pred_indices = np.where(probs > 0.5)[0].tolist()
            pred_names = [TARGET_PATHOLOGIES[i] for i in pred_indices]
            confidences = {TARGET_PATHOLOGIES[i]: f"{probs[i]:.4f}" for i in pred_indices}

            # Grad-CAM heatmaps for predicted classes
            for idx in pred_indices:
                heat = GRADCAM.generate_heatmap(input_tensor.clone(), idx)
                overlaid = GRADCAM.overlay_heatmap(str(processed_png), heat)
                heat_name = f"heatmap_{TARGET_PATHOLOGIES[idx]}.png"
                heat_path = case_dir / heat_name
                cv2.imwrite(str(heat_path), cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
                heatmap_files.append(heat_name)
        except Exception:
            app.logger.exception('Model inference failed; falling back to placeholder outputs')

    # If model was not usable or produced no predictions, create a harmless placeholder
    if not heatmap_files:
        try:
            # Create a simple center-focused Gaussian heatmap overlay as a demonstrative artifact
            img_cv = cv2.imread(str(processed_png))
            if img_cv is None:
                # Try reading via PIL fallback
                img_p = Image.open(processed_png).convert('RGB')
                img_cv = cv2.cvtColor(np.array(img_p), cv2.COLOR_RGB2BGR)

            h, w = img_cv.shape[:2]
            # Create a Gaussian blob centered
            xv, yv = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
            d = np.sqrt(xv**2 + yv**2)
            sigma = 0.6
            blob = np.exp(-(d**2) / (2*sigma**2))
            blob = (blob - blob.min()) / (blob.max() - blob.min() + 1e-8)
            heatmap_colored = cv2.applyColorMap((blob*255).astype('uint8'), cv2.COLORMAP_JET)
            overlaid = cv2.addWeighted(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), 0.6, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 0.4, 0)
            heat_name = 'heatmap_placeholder.png'
            heat_path = case_dir / heat_name
            cv2.imwrite(str(heat_path), cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
            heatmap_files.append(heat_name)
            # Placeholder textual prediction to show in UI
            confidences = {'ModelUnavailable': '0.0000'}
            pred_names = []
        except Exception:
            app.logger.exception('Failed to generate placeholder heatmap')

    # Report (real or placeholder)
    report_text = build_report(pred_names, confidences, heatmap_files)
    report_name = 'report.txt'
    report_path = case_dir / report_name
    report_path.write_text(report_text, encoding='utf-8')

    # Persist metadata for the case page
    import json
    meta = {
        "case_id": case_id,
        "processed_image": processed_png.name,
        "predictions": [{"label": k, "confidence": v} for k, v in confidences.items()],
        "heatmaps": heatmap_files,
        "created_at": datetime.datetime.now().isoformat()
    }
    (case_dir / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    return redirect(url_for('case_view', case_id=case_id))


@app.route('/case/<case_id>')
def case_view(case_id):
    case_dir = CASES_ROOT / case_id
    if not case_dir.exists():
        flash('Case not found')
        return redirect(url_for('check_model'))

    # Load artifacts
    import json
    processed_png = None
    heatmaps = []
    report = ''
    predictions = []
    meta_path = case_dir / 'meta.json'
    if meta_path.exists():
        m = json.loads(meta_path.read_text(encoding='utf-8'))
        processed_png = m.get('processed_image')
        heatmaps = m.get('heatmaps', [])
        predictions = m.get('predictions', [])
    if (case_dir / 'report.txt').exists():
        report = (case_dir / 'report.txt').read_text(encoding='utf-8')

    return render_template('case.html', case_id=case_id,
                           processed_image=processed_png,
                           heatmaps=sorted(heatmaps),
                           predictions=predictions,
                           report_text=report)


@app.route('/files/<case_id>/<path:filename>')
def case_file(case_id, filename):
    case_dir = CASES_ROOT / case_id
    return send_from_directory(str(case_dir), filename)


@app.route('/download/<case_id>/report')
def download_report(case_id):
    case_dir = CASES_ROOT / case_id
    return send_from_directory(str(case_dir), 'report.txt', as_attachment=True)


# ---------- Report-grounded chat ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def answer_from_report(report_text: str, question: str, k: int = 5):
    """Improved chatbot that provides intelligent answers based on report content"""
    import re
    
    # Normalize question
    question_lower = question.lower().strip()
    
    # Extract key information from report
    # Remove noisy report markers and short lines that are just headings
    lines = [line.strip() for line in report_text.split('\n') if line.strip()]
    lines = [l for l in lines if not re.search(r'---|^Chest X-ray|^Visual Explanations|^Predicted Findings|^Disclaimer', l, re.I) and len(l) > 6]
    
    # Check for specific question patterns and provide direct answers
    if any(word in question_lower for word in ['what', 'which', 'tell']):
        if 'patholog' in question_lower or 'condition' in question_lower or 'finding' in question_lower or 'detect' in question_lower:
            findings = [line for line in lines if 'Confidence' in line or any(path in line for path in TARGET_PATHOLOGIES)]
            if findings:
                return "According to the analysis:\n" + "\n".join(f"• {f}" for f in findings[:5])
            else:
                return "The analysis did not detect any specific pathologies with high confidence. The X-ray appears normal or findings are below the detection threshold."
    
    if 'confidence' in question_lower or 'score' in question_lower or 'probability' in question_lower:
        confidence_lines = [line for line in lines if 'Confidence' in line or '%' in line]
        if confidence_lines:
            return "Confidence scores from the report:\n" + "\n".join(f"• {c}" for c in confidence_lines[:5])
        else:
            return "No specific confidence scores are listed in the report."
    
    if 'heatmap' in question_lower or 'visual' in question_lower or 'grad-cam' in question_lower:
        heatmap_lines = [line for line in lines if 'Heatmap' in line or 'Grad-CAM' in line]
        if heatmap_lines:
            return "Visual explanations:\n" + "\n".join(f"• {h}" for h in heatmap_lines)
        else:
            return "Check the Heatmaps tab to see visual explanations of the AI's findings."
    
    if 'normal' in question_lower or 'healthy' in question_lower or 'fine' in question_lower:
        if 'No specific pathologies detected' in report_text or 'No Pathologies Detected' in report_text:
            return "According to the analysis, no specific pathologies were detected with high confidence. This suggests the X-ray appears relatively normal."
        else:
            return "The analysis detected one or more findings. Please review the predictions and report for details."
    
    # Fall back to semantic search for other questions
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", report_text) if s.strip() and len(s.strip()) > 10]
    # Filter out divider lines and report headings
    sentences = [s for s in sentences if not re.search(r'---|Chest X-ray Analysis Report|End of Report|Visual Explanations', s, re.I)]
    if not sentences:
        return "The report doesn't contain enough detailed information to answer that question."
    
    # Use TF-IDF for semantic matching
    docs = sentences + [question]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=100).fit(docs)
        mat = vectorizer.transform(docs)
        sims = cosine_similarity(mat[-1], mat[:-1]).flatten()
        
        if sims.size == 0 or np.max(sims) < 0.1:
            return "I couldn't find specific information in the report to answer that. Try asking about 'findings', 'confidence scores', or 'heatmaps'."
        
        top_idx = np.argsort(-sims)[:k]
        picked = [sentences[i] for i in top_idx if sims[i] > 0.15]
        
        if picked:
            return "Based on the report:\n" + "\n".join(f"• {p}" for p in picked[:3])
        else:
            return "I found limited relevant information. Please review the full report for comprehensive details."
    except:
        return "I couldn't process that question. Please try rephrasing or ask about specific findings."


def load_clip_once():
    """Lazy-load CLIP model and processor for multimodal retrieval (HF transformers)."""
    global CLIP_MODEL, CLIP_PROCESSOR, CLIP_DEVICE
    if CLIP_MODEL is not None and CLIP_PROCESSOR is not None:
        return
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        CLIP_MODEL = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        CLIP_DEVICE = torch.device('cpu' if DEVICE is None else DEVICE)
        CLIP_MODEL.to(CLIP_DEVICE)
        app.logger.info('Loaded CLIP model for multimodal retrieval')
    except Exception:
        app.logger.exception('Failed to load CLIP model; multimodal answers will fall back to TF-IDF')
        CLIP_MODEL = None
        CLIP_PROCESSOR = None
        CLIP_DEVICE = None


    def synthesize_answer_from_snippets(snippets, question: str = None):
        """Produce a concise, human-like answer from retrieved report snippets.

        This is a lightweight, deterministic paraphraser used when a generative
        model is unavailable. It avoids echoing report headings and aims to
        present a short summary plus evidence bullets.
        """
        import re
        # Clean snippet markers
        cleaned = [re.sub(r'^\[\d+\]\s*', '', s).strip() for s in snippets]
        cleaned = [re.sub(r'^[\-\•\*]\s*', '', s).strip() for s in cleaned]
        # Remove trivial lines
        cleaned = [c for c in cleaned if len(c) > 8 and not re.search(r'---|End of Report|Chest X-ray', c, re.I)]

        if not cleaned:
            return "I couldn't find useful details in the report to answer that question."

        # If explicit 'no findings' language exists, return that clearly
        if any('no specific patholog' in c.lower() or 'no specific findings' in c.lower() for c in cleaned):
            return "Short answer: The report indicates no specific pathologies detected with high confidence.\nEvidence:\n• No findings reported in the analysis."

        # Try to detect pathology names and confidence scores
        found = {}
        for c in cleaned:
            for t in TARGET_PATHOLOGIES:
                if t.replace('_', ' ').lower() in c.lower() or t.lower() in c.lower():
                    m = re.search(r'([0-9]*\.[0-9]+)', c)
                    conf = m.group(1) if m else None
                    found[t] = conf

        if found:
            names = [n.replace('_', ' ') for n in found.keys()]
            summary = f"Short answer: The analysis suggests possible {', '.join(names)}."
            evidence_lines = []
            for n in found:
                evidence_lines.append(f"• {n.replace('_', ' ')}: confidence = {found[n] or 'N/A'}")
            return summary + "\nEvidence:\n" + "\n".join(evidence_lines)

        # Otherwise, craft a readable one-line summary from the first snippet and show 1-3 evidence bullets
        first = cleaned[0]
        short = first if len(first) < 200 else first[:200] + '...'
        evidence = "\n".join(f"• {c}" for c in cleaned[:3])
        return f"Short answer: {short}\nEvidence:\n{evidence}"


def answer_from_report_multimodal(report_text: str, question: str, image_path: str = None, k: int = 3):
    """Grounded multimodal answer using CLIP retrieval + TF-IDF fallback.

    - Embeds report sentences and the question using CLIP text encoder.
    - Optionally embeds the processed image and combines similarities.
    - Returns the top-k report snippets concatenated into an answer.
    """
    import re
    import numpy as np

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", report_text) if s.strip()]
    if not sentences:
        return "I can't answer because the report is empty."

    # Try CLIP first
    load_clip_once()
    if CLIP_MODEL is None or CLIP_PROCESSOR is None:
        return answer_from_report(report_text, question, k=k)

    try:
        import torch
        # Text embeddings
        text_inputs = CLIP_PROCESSOR(text=sentences, return_tensors='pt', padding=True, truncation=True)
        text_inputs = {k: v.to(CLIP_DEVICE) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_feats = CLIP_MODEL.get_text_features(**text_inputs).cpu().numpy()

        q_inputs = CLIP_PROCESSOR(text=[question], return_tensors='pt', padding=True, truncation=True)
        q_inputs = {k: v.to(CLIP_DEVICE) for k, v in q_inputs.items()}
        with torch.no_grad():
            q_feat = CLIP_MODEL.get_text_features(**q_inputs).cpu().numpy()

        img_feat = None
        if image_path:
            try:
                from PIL import Image
                img = Image.open(image_path).convert('RGB')
                img_inputs = CLIP_PROCESSOR(images=img, return_tensors='pt')
                img_inputs = {k: v.to(CLIP_DEVICE) for k, v in img_inputs.items()}
                with torch.no_grad():
                    img_feat = CLIP_MODEL.get_image_features(**img_inputs).cpu().numpy()
            except Exception:
                app.logger.exception('Failed to embed image for multimodal retrieval')
                img_feat = None

        # Normalize
        def norm(a):
            a = a.astype('float32')
            a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
            return a
        text_feats = norm(text_feats)
        q_feat = norm(q_feat)
        if img_feat is not None:
            img_feat = norm(img_feat)

        sims_q = (text_feats @ q_feat.T).squeeze(1)
        if img_feat is not None:
            sims_img = (text_feats @ img_feat.T).squeeze(1)
            sims = 0.6 * sims_q + 0.4 * sims_img
        else:
            sims = sims_q

        top_idx = np.argsort(-sims)[:k]
        picked = [sentences[i] for i in top_idx]
        answer = "Based on the generated report, here are the most relevant details:\n- " + "\n- ".join(picked)
        return answer
    except Exception:
        app.logger.exception('Error in multimodal retrieval; falling back to TF-IDF')
        return answer_from_report(report_text, question, k=k)




@app.route('/case/<case_id>/chat', methods=['POST'])
def chat(case_id):
    """Gemini-powered AI chat"""
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '').strip()
    if not question:
        return jsonify({"ok": False, "answer": "Please enter a question."})
    
    report_path = CASES_ROOT / case_id / 'report.txt'
    if not report_path.exists():
        return jsonify({"ok": False, "answer": "Report not found."})
    report_text = report_path.read_text(encoding='utf-8')
    
    try:
        import google.generativeai as genai
        genai.configure(api_key="AIzaSyB6XZb_5cO2BmlCyDEHevUCUkxjhEgp1sk")
        model = genai.GenerativeModel('models/gemini-flash-latest')
        
        response = model.generate_content(
            f"""You are a medical AI assistant analyzing a Chest X-ray report.

Your task: Read the ENTIRE report below and answer the user's question clearly and professionally.

KEY INSTRUCTIONS:
- Look for the "Predicted Findings" section to see what pathologies were detected
- If it says "No specific pathologies detected", tell the user the X-ray appears normal
- If pathologies are listed, mention them and their confidence scores
- Reference the heatmap visualizations if relevant
- Be concise but informative (2-4 sentences)

IMPORTANT SAFETY RULES:
- You can ONLY answer questions about what the report SAYS (findings, confidence scores, interpretations)
- You CANNOT provide medical advice, treatment recommendations, or suggest cures
- If asked about treatment/cure/medication, respond: "I can only explain what the report shows. Please consult a physician for treatment advice."
- If asked about prognosis or what to do next, respond: "Please discuss next steps with your healthcare provider."

=== FULL MEDICAL REPORT ===
{report_text}
=== END OF REPORT ===

USER'S QUESTION: {question}

YOUR ANSWER:""",
            request_options={'timeout': 10}  # 10 second timeout
        )
        
        return jsonify({"ok": True, "answer": response.text.strip()})
    except Exception as e:
        error_msg = str(e)
        # User-friendly error messages
        if 'timeout' in error_msg.lower() or '503' in error_msg or 'failed to connect' in error_msg.lower():
            return jsonify({"ok": False, "answer": "⚠️ Network issue: Can't reach Google AI servers. Please check your internet connection or try again in a moment."})
        elif '429' in error_msg:
            return jsonify({"ok": False, "answer": "⚠️ Rate limit reached. Please wait a moment and try again."})
        elif '404' in error_msg:
            return jsonify({"ok": False, "answer": "⚠️ API configuration issue. The AI model is unavailable."})
        else:
            return jsonify({"ok": False, "answer": f"Chat error: {error_msg[:200]}"})



def build_report(predicted_class_names, confidence_scores, heatmap_filenames):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report = ["--- Chest X-ray Analysis Report ---", f"Date: {ts}", "", "Predicted Findings:"]
    if not predicted_class_names:
        report.append("- No specific pathologies detected with high confidence.")
    else:
        for name in predicted_class_names:
            conf = confidence_scores.get(name, 'N/A')
            report.append(f"- {name}: Confidence = {conf}")
            report.append(f"  Explanation: The model identified patterns consistent with {name}.")
    report.append("")
    report.append("Visual Explanations (Grad-CAM Heatmaps):")
    if heatmap_filenames:
        for h in heatmap_filenames:
            c = h.replace('heatmap_', '').replace('.png', '')
            report.append(f"- Heatmap for {c}: see attached image.")
    else:
        report.append("- No heatmaps generated.")
    report.append("")
    report.append("Disclaimer: This AI-generated report is for informational purposes only and not a medical diagnosis.")
    report.append("--- End of Report ---")
    return "\n".join(report)


if __name__ == '__main__':
    # Local dev server
    app.run(host='0.0.0.0', port=8000, debug=False)
