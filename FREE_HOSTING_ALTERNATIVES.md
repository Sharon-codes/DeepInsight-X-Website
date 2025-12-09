# Free Hosting Alternatives to Render

If Render doesn't work for you, here are the best free alternatives for hosting a Python Flask application.

## 1. PythonAnywhere (Highly Recommended for Flask)
**Website:** [www.pythonanywhere.com](https://www.pythonanywhere.com/)

This is specifically designed for Python hosting and is very beginner-friendly.

*   **Pros:**
    *   ✅ **Forever Free** (Beginner account).
    *   ✅ specific support for Flask/Django.
    *   ✅ No credit card required.
    *   ✅ Access to a bash console in the browser.
*   **Cons:**
    *   ❌ Domain will be `yourusername.pythonanywhere.com`.
    *   ❌ You have to log in once every 3 months to click a "Run until 3 months from now" button to keep it active.
    *   ❌ Limited CPU/Bandwidth on free tier.
    *   ❌ **Important:** Free tier only allows outgoing requests to a specific whitelist of sites (installing packages works, but your app can't call random external APIs unless you upgrade).

**How to Deploy:**
1.  Sign up.
2.  Go to "Web" tab -> "Add a new web app".
3.  Select "Flask" and your Python version.
4.  Upload your code (you can use the "Files" tab or `git clone` in the "Consoles" tab).
5.  Update the WSGI configuration file (provided by them) to point to your `app.py`.

## 2. Vercel
**Website:** [vercel.com](https://vercel.com/)

Mostly known for frontend, but supports Python Serverless Functions.

*   **Pros:**
    *   ✅ Very generous free tier.
    *   ✅ Extremely fast global CDN.
    *   ✅ Custom domain support (even on free tier).
    *   ✅ Git integration (auto-deploys on push).
*   **Cons:**
    *   ❌ "Serverless" architecture means your app sleeps and wakes up (cold starts).
    *   ❌ Requires a `vercel.json` config file.
    *   ❌ File uploads are tricky (ephemeral filesystem - files disappear after the request finishes). You would need to upload images to an external service like Cloudinary or AWS S3 instead of saving them locally to `static/uploads`.

## 3. Koyeb
**Website:** [koyeb.com](https://www.koyeb.com/)

A newer platform similar to Render/Heroku.

*   **Pros:**
    *   ✅ Free tier available (Eco Dynos).
    *   ✅ No credit card required for starter.
    *   ✅ Native Docker support.
*   **Cons:**
    *   ❌ Newer, so fewer tutorials online.
    *   ❌ Limits on RAM/CPU.

## 4. Google Cloud Run (Free Tier)
**Website:** [cloud.google.com/run](https://cloud.google.com/run)

Enterprise-grade serverless.

*   **Pros:**
    *   ✅ 2 million requests per month free.
    *   ✅ Scalable.
    *   ✅ Professional.
*   **Cons:**
    *   ❌ **Requires Credit Card** for signup (identity verification).
    *   ❌ Setup is more complex (requires Docker container).
    *   ❌ Can be intimidating interface.

## 5. ngrok (For Demos Only)
**Website:** [ngrok.com](https://ngrok.com/)

Not for permanent hosting, but perfect for showing a friend *right now*.

*   **Pros:**
    *   ✅ Instant.
    *   ✅ Runs on your own computer (so it's fast and has access to your GPU).
*   **Cons:**
    *   ❌ URL changes every time you restart.
    *   ❌ Your computer must stay on.

---

## 🏆 Recommendation

1.  **If you want simple & permanent:** Go with **PythonAnywhere**. It's the most "standard" way to host a simple Flask app for free.
2.  **If you want custom domains & speed:** Go with **Vercel**, BUT you will need to change how you handle file uploads (since you can't save files to the server disk permanently).
