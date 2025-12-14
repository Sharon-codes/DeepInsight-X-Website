import google.generativeai as genai

genai.configure(api_key="AIzaSyB6XZb_5cO2BmlCyDEHevUCUkxjhEgp1sk")

print("Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"  - {m.name}")
