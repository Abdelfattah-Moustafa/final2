import subprocess
import sys

def check(name, code):
    try:
        p = subprocess.run([sys.executable, "-u", "-c", code], capture_output=True, text=True, timeout=15)
        print(f"{name}: status={p.returncode}")
        if p.stdout.strip():
            print(f"  stdout: {p.stdout.strip()}")
        if p.stderr.strip():
            print(f"  stderr: {p.stderr.strip()}")
    except Exception as e:
        print(f"{name}: timeout or error {e}")

code_template = """
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'app', 'server'))
import {module}
from asl_service import SignDB
db = SignDB()
print("instantiated ok")
"""

modules = [
    "bcrypt",
    "jose",
    "socketio",
    "fastapi",
    "sqlalchemy",
    "uvicorn",
    "email_validator"
]

for m in modules:
    code = code_template.format(module=m)
    check(m, code)
