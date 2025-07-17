import os
from dotenv import load_dotenv
import google.generativeai as genai

# Step 1: Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Step 2: Configure Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Step 3: Get user prompt and cost
print(" README Generator Ready.")
infra_prompt = input("Describe the infrastructure (same as before):\n")
cost_input = input("Enter estimated monthly cost (e.g., $20.59):\n")

# Step 4: System prompt to guide the agent
system_prompt = f"""
Generate a professional README.md for a Terraform project based on the following:

Infrastructure Description:
\"\"\"
{infra_prompt}
\"\"\"

Include these sections:
- Title
- Overview of infrastructure
- Terraform Requirements
- How to Deploy (init, plan, apply)
- Estimated Cost: {cost_input}
- License (MIT)
Output in markdown format only.
"""

# Step 5: Generate README
try:
    response = model.generate_content(system_prompt)
    readme_content = response.text.strip()

    with open("README.md", "w") as f:
        f.write(readme_content)

    print("\n README.md generated successfully!")
    print(" Preview:")
    print("\n" + readme_content[:300] + "...\n")

except Exception as e:
    print(" Error generating README:", e)
