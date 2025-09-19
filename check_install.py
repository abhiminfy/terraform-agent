# check_install.py
import importlib
import subprocess
import sys

# map a module import name -> pip package name
checks = {
    # Strands SDK (core + companion tools)
    "strands_agents": "strands-agents",
    "strands_agents_tools": "strands-agents-tools",
    # Gemini client
    "google.generativeai": "google-generativeai",
    # Web / infra libs
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "pydantic": "pydantic",
    "httpx": "httpx",
    "dotenv": "python-dotenv",  # import name 'dotenv'
    "jwt": "PyJWT",  # import name 'jwt'
    "slowapi": "slowapi",
    "prometheus_client": "prometheus-client",
    "celery": "celery",
    "git": "gitpython",  # import name 'git'
    "infracost": "infracost",  # optional if used
}

missing = []
for mod, pkg in checks.items():
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(pkg)

if not missing:
    print("All checked packages are already installed ✅")
    sys.exit(0)

# dedupe and keep ordering stable
to_install = []
seen = set()
for p in missing:
    if p not in seen:
        seen.add(p)
        to_install.append(p)

pip_cmd = [sys.executable, "-m", "pip", "install"] + to_install
print("Installing missing packages:")
print(" ".join(pip_cmd))
# run install
try:
    subprocess.check_call(pip_cmd)
    print("\nInstall finished. ✅")
    print("If you run inside Docker, rebuild your image or restart the container.")
except subprocess.CalledProcessError as e:
    print("\nInstall failed with exit code", e.returncode)
    print("You can try running the printed pip command yourself for more logs.")
