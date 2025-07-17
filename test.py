import os
import google.generativeai as genai
from dotenv import load_dotenv

# Step 1: Load environment variables
print("Loading .env variables...")
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Step 2: Configure Gemini
genai.configure(api_key=API_KEY)

# Step 3: Initialize Model
model = genai.GenerativeModel("gemini-1.5-pro")

# Step 4: Test Generation
prompt = "Explain what Terraform is in one sentence."
response = model.generate_content(prompt)
print("\nGemini API Response:\n")
print(response.text)
