# Website Hosting Guide

## ✅ Website Status

**Good News**: Your website is fully functional and ready to host! It includes:
- ✅ Complete Flask web application
- ✅ HTML templates (10 pages including home, upload, results)
- ✅ Model loading and inference system
- ✅ Grad-CAM visualization
- ✅ Report generation
- ✅ Production-ready configuration (wsgi.py, Procfile)

## Prerequisites

Before hosting, you need:
1. **Trained model** at `Core/models/best_model.pth` (from training)
2. **Dependencies installed** (see requirements.txt)

## Local Testing (Before Hosting)

### 1. Install Dependencies
```powershell
cd d:\IIT\Website
pip install -r requirements.txt
```

### 2. Test Locally
```powershell
python app.py
```
Visit: http://localhost:8000

### 3. Test Model Loading
Visit: http://localhost:8000/__status

Should show:
```json
{
  "model_path": "path/to/best_model.pth",
  "device": "cpu",
  "model_loaded": true,
  "gradcam_ready": true
}
```

## Hosting Options

### Option 1: Local Network Hosting (Easiest)

**Use Case**: Share with people on your local network (home/office)

**Steps**:
1. Start the server:
```powershell
cd d:\IIT\Website
python app.py
```

2. Find your local IP:
```powershell
ipconfig
# Look for "IPv4 Address" (e.g., 192.168.1.100)
```

3. Share with others:
```
http://YOUR_IP_ADDRESS:8000
Example: http://192.168.1.100:8000
```

**Pros**: 
- ✅ Free
- ✅ Easy setup
- ✅ Works immediately

**Cons**:
- ❌ Only accessible on same network
- ❌ Computer must stay on
- ❌ Not secure (HTTP only)

---

### Option 2: Heroku (Free/Paid Cloud Hosting)

**Use Case**: Public access, good for demos and portfolios

**Setup**:
1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli

2. Login and create app:
```powershell
cd d:\IIT\Website
heroku login
heroku create your-app-name
```

3. Deploy:
```powershell
git init
git add .
git commit -m "Initial commit"
heroku git:remote -a your-app-name
git push heroku main
```

4. Your website will be at: `https://your-app-name.herokuapp.com`

**Important**: Model file must be <500MB for Heroku free tier

**Pros**:
- ✅ Public access (anyone can visit)
- ✅ HTTPS (secure)
- ✅ Free tier available
- ✅ Easy deployment

**Cons**:
- ❌ Free tier sleeps after 30 min inactivity (wakes up on visit)
- ❌ Model file size limit

---

### Option 3: Render (Free Cloud Hosting)

**Use Case**: Similar to Heroku, better free tier

**Setup**:
1. Create account: https://render.com
2. Connect your GitHub repository
3. Create new "Web Service"
4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn wsgi:app`

**Pros**:
- ✅ Free tier (750 hours/month)
- ✅ Auto-deploy from GitHub
- ✅ HTTPS
- ✅ Better than Heroku free tier

**Cons**:
- ❌ Spins down after inactivity
- ❌ Slower cold starts

---

### Option 4: Google Cloud / AWS / Azure (Production)

**Use Case**: High traffic, production deployment

**Platforms**:
- **Google Cloud Run**: $0.40/million requests
- **AWS EC2**: ~$10-30/month for small instance
- **Azure App Service**: ~$13/month basic tier

**Pros**:
- ✅ Scalable
- ✅ Always on
- ✅ Professional setup
- ✅ Custom domain support

**Cons**:
- ❌ More complex setup
- ❌ Costs money
- ❌ Requires more configuration

---

### Option 5: ngrok (Quick Public Access)

**Use Case**: Temporary public link for demos/testing

**Setup**:
1. Download ngrok: https://ngrok.com/download

2. Start your app:
```powershell
cd d:\IIT\Website
python app.py
```

3. In another terminal:
```powershell
ngrok http 8000
```

4. Get public URL (valid for 8 hours):
```
Forwarding: https://abc123.ngrok.io -> http://localhost:8000
```

**Pros**:
- ✅ Instant public access
- ✅ HTTPS
- ✅ No deployment needed
- ✅ Great for demos

**Cons**:
- ❌ Temporary URL (changes each time)
- ❌ Computer must stay on
- ❌ Free tier has limits

---

## Recommended Approach

### For Testing/Development:
**Use ngrok** - Get public link instantly, perfect for showing to others

### For Portfolio/Long-term:
**Use Render** - Free, always accessible, professional

### For Production:
**Use Google Cloud Run** - Scalable, pay-as-you-go, professional

## Security Considerations

⚠️ **Before Public Hosting**:

1. **Change Flask secret key** (in app.py line 33):
```python
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-random-secret-key-here')
```

2. **Turn off debug mode** for production (app.py line 423):
```python
app.run(host='0.0.0.0', port=8000, debug=False)  # Change to False
```

3. **Add file upload limits** (already done in app.py line 44):
```python
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
```

4. **Use environment variables** for sensitive config:
```powershell
$env:FLASK_SECRET_KEY="your-secret-key"
$env:FORCE_CPU="1"
python app.py
```

## Production Deployment Checklist

- [ ] Trained model exists at `Core/models/best_model.pth`
- [ ] All dependencies in requirements.txt are installed
- [ ] Test locally at http://localhost:8000
- [ ] Test model loading at http://localhost:8000/__status
- [ ] Upload test image and verify predictions work
- [ ] Change Flask secret key (for security)
- [ ] Set debug=False (for production)
- [ ] Choose hosting platform
- [ ] Deploy and test public URL
- [ ] Verify file uploads work on hosted version
- [ ] Share link and test from different device

## Troubleshooting

### Model doesn't load:
- Check: Model file exists at `Core/models/best_model.pth`
- Check: File size is appropriate for hosting platform
- Visit: http://your-url/__status to see error

### Predictions fail:
- Check: Image format is PNG/JPG/JPEG
- Check: File size < 50MB
- Check: Model loaded successfully

### Hosting platform build fails:
- Check: All dependencies in requirements.txt
- Check: Python version compatibility (Python 3.8+)
- Check: Procfile or start command configured correctly

## Quick Start Commands

**Local Testing**:
```powershell
cd d:\IIT\Website
python app.py
# Visit http://localhost:8000
```

**ngrok Public Link** (after running above):
```powershell
ngrok http 8000
# Copy the https://xxx.ngrok.io URL
```

**Check Status**:
```powershell
curl http://localhost:8000/__status
```

## Next Steps

1. **Complete training** to get the model file
2. **Test locally** first using `python app.py`
3. **Choose hosting option** based on your needs
4. **Deploy and share** your medical AI web app!

The website is production-ready and waiting for your trained model! 🚀
