# Chest X‑ray AI Website

A modular Flask web app to:
- Explain the project (Home, Problem, Solution, Team)
- Let users upload a chest X‑ray (DICOM/PNG/JPG)
- Run the Core model to produce predictions and Grad‑CAM heatmaps
- Generate a structured report
- Provide a report‑grounded chat assistant (answers strictly based on the generated report)

## Run locally

Prerequisites:
- Python 3.10+
- The Core folder present alongside Website (Website imports Core/utils)
- A trained model file in `Core/models` (e.g., `best_model.pth` or `final_model.pth`)

Install dependencies:

```powershell
python -m venv .venv ; .\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

If you want the local multimodal chat (image+text) to work without external APIs, install the Hugging Face `transformers` package which provides CLIP:

```powershell
pip install transformers==4.38.0
```

Start the server:

```powershell
$env:FLASK_SECRET_KEY="dev-key" ; python app.py
```

Open http://localhost:8000

## Deployment

Any WSGI host will work (Gunicorn, Azure Web Apps, etc.). The WSGI entry is `wsgi:application`.

Example (Linux):

```bash
gunicorn -w 2 -b 0.0.0.0:8000 wsgi:application
```

Environment variables:
- `FLASK_SECRET_KEY`: secret for sessions (required in production)
- `MAX_FILE_SIZE_MB`: upload size cap (default 50)

## Notes
- This app imports from `../Core/utils`. Don’t move the Core folder; or set `PYTHONPATH` to include it.
- Model checkpoint autodetects `best_model.pth`, `final_model.pth`, or `latest_model.pth` in `Core/models`.
- Chat answers are strictly grounded in the generated report using TF‑IDF based retrieval. No external LLMs are used.
