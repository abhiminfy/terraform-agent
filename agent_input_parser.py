import os
import re
import boto3
import google.generativeai as genai
from dotenv import load_dotenv

def highlight_placeholders(terraform_code: str) -> str:
    substitutions = {
        r'ami\s*=\s*".+?"': 'ami = "ami-09ac0b140f63d3458"',
        r'username\s*=\s*".+?"': 'username = "adminuser"',
        r'password\s*=\s*".+?"': 'password = "MyS3cur3P@ssw0rd!"',
        r'db_name\s*=\s*".+?"': 'db_name = "<terraform-db-subnet-group>"',
        r'identifier\s*=\s*".+?"': 'identifier = "terraform-mysql-db"',
        r'key_name\s*=\s*".+?"': 'key_name = "terraform-key"',
        r'subnet_id\s*=\s*".+?"': 'subnet_id = "subnet-0be2812d720cb27e1"',
        r'vpc_security_group_ids\s*=\s*\[.+?\]': 'vpc_security_group_ids = ["sg-087ec30e65e4381af"]',
        r'availability_zone\s*=\s*".+?"': 'availability_zone = "us-east-1a"'
    }
    for pattern, replacement in substitutions.items():
        terraform_code = re.sub(pattern, replacement, terraform_code)
    return terraform_code

def estimate_cost():
    ec2_price = 0.0116  # USD/hour for t2.micro
    rds_price = 0.017   # USD/hour for db.t3.micro MySQL

    total_hourly = ec2_price + rds_price
    monthly_cost = round(total_hourly * 24 * 30, 2)

    print("\n Estimated Cost Breakdown (Per Hour):")
    print(f"  - EC2 (t2.micro): ${ec2_price}/hour")
    print(f"  - RDS (db.t3.micro, MySQL): ${rds_price}/hour")
    print(f"\n Total Estimated Monthly Cost: ${monthly_cost}")

# Step 1: Load environment variables
print("Loading environment variables...")
load_dotenv()

# Step 2: Get API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: Missing API key. Please set GOOGLE_API_KEY in your .env file.")
    exit(1)

# Step 3: Configure Gemini
try:
    genai.configure(api_key=API_KEY)
except Exception as config_error:
    print(f"ERROR: Failed to configure Gemini: {config_error}")
    exit(1)

# Step 4: Load model
try:
    model = genai.GenerativeModel("gemini-1.5-pro")
except Exception as model_error:
    print(f"ERROR: Failed to initialize model: {model_error}")
    exit(1)

# System prompt
system_prompt = (
    "You are an AI DevOps assistant. Convert the user's infrastructure request into valid Terraform code. "
    "Do not include explanations or CLI commands. Only output pure HCL Terraform code. "
    "Use default values where necessary, and always follow Terraform syntax correctly."
)

# Step 5: Input & Response
if __name__ == "__main__":
    print("\nGemini Terraform Agent Ready.\n")
    user_prompt = input("Describe your infrastructure needs:\n")
    print("\nGenerating Terraform output...\n")

    try:
        response = model.generate_content([
            {"role": "user", "parts": [system_prompt + "\n" + user_prompt]}
        ])
        raw_code = response.text
        cleaned_code = highlight_placeholders(raw_code)

        # Save to file
        with open("generated.tf", "w") as f:
            f.write(cleaned_code)
        print(" Terraform code saved to 'generated.tf'.")

        # Show code
        print("\nFinal Terraform Code:\n")
        print(cleaned_code)

        # Estimate cost
        estimate_cost()

    except Exception as e:
        print("\nERROR: Gemini API call failed -", e)


