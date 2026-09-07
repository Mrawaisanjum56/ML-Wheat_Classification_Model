import os
import subprocess
import sys
import venv
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV_DIR = BACKEND / ".venv"

def venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def create_venv_if_needed():
    if not VENV_DIR.exists():
        print("📦 Creating virtual environment...")
        venv.create(VENV_DIR, with_pip=True)

def install_requirements():
    py = str(venv_python())
    print("⬇️ Installing backend dependencies...")
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([py, "-m", "pip", "install", "-r", str(BACKEND / "requirements.txt")])

def run_backend():
    py = str(venv_python())
    print("🚀 Starting FastAPI backend on http://127.0.0.1:8000")
    subprocess.call([py, "-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"], cwd=str(BACKEND))

def run_frontend():
    print("🌐 Starting frontend on http://127.0.0.1:5500")
    # Uses Python built-in static server
    subprocess.call([sys.executable, "-m", "http.server", "5500"], cwd=str(FRONTEND))

def main():
    create_venv_if_needed()
    install_requirements()

    t1 = threading.Thread(target=run_backend, daemon=True)
    t2 = threading.Thread(target=run_frontend, daemon=True)

    t1.start()
    time.sleep(1.5)
    t2.start()

    print("\n✅ App is starting...")
    print("Frontend: http://127.0.0.1:5500")
    print("Backend : http://127.0.0.1:8000")
    print("Docs    : http://127.0.0.1:8000/docs")
    print("\nPress CTRL+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")

if __name__ == "__main__":
    main()