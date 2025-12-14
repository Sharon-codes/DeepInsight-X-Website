import google.generativeai as genai
import sys

genai.configure(api_key="AIzaSyB6XZb_5cO2BmlCyDEHevUCUkxjhEgp1sk")
model = genai.GenerativeModel('models/gemini-flash-latest')

try:
    print("Testing connection to Google AI...")
    response = model.generate_content("Say hello", request_options={'timeout': 10})
    print(f"✓ SUCCESS: {response.text}")
    sys.exit(0)
except Exception as e:
    print(f"✗ FAILED: {e}")
    print("\nTroubleshooting:")
    print("1. Check if https://ai.google.dev opens in your browser")
    print("2. Temporarily disable firewall/antivirus")
    print("3. Check proxy settings")
    sys.exit(1)
