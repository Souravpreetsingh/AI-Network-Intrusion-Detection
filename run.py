#!/usr/bin/env python3
"""
Main Entry Point for AI-Based Network Intrusion Detection System.

Usage:
    python run.py train          # Train model and save to models/model.pkl
    python run.py frontend       # Launch the web server (frontend + API)
    python run.py app            # Launch Streamlit app (legacy)
    python run.py api            # Launch Flask API (legacy, API-only)
    python run.py generate       # Generate synthetic dataset
    python run.py all            # Generate data, train, and launch frontend
"""
import sys
import os
import subprocess
import traceback


def print_banner():
    banner = """
    =====================================================
      AI-Based Network Intrusion Detection System
      Using Machine Learning
    =====================================================
    """
    print(banner)


def cmd_generate():
    """Generate synthetic CICIDS2017 dataset."""
    try:
        print("[INFO] Generating synthetic dataset...")
        from generate_sample_data import main
        main()
    except ImportError as e:
        print(f"[ERROR] Missing library: {e}")
        print("       Please install required packages: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to generate dataset: {e}")
        traceback.print_exc()
        sys.exit(1)


def cmd_train():
    """Train the Random Forest model and save it to models/model.pkl."""
    try:
        print("[INFO] Running training pipeline...")
        from train_pipeline import run_training_pipeline
        result = run_training_pipeline()
        if result is None or not result.get("success"):
            print("[ERROR] Training failed. See error messages above.")
            sys.exit(1)
    except ImportError as e:
        print(f"[ERROR] Missing library: {e}")
        print("       Please install required packages: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Training pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)


def cmd_app():
    """Launch the Streamlit frontend (legacy)."""
    try:
        print("[INFO] Launching Streamlit application...")
        streamlit_script = os.path.join(os.path.dirname(__file__), "app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_script], check=True)
    except FileNotFoundError as e:
        print(f"[ERROR] Streamlit not found: {e}")
        print("       Install it with: pip install streamlit")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Streamlit exited with code {e.returncode}")
        sys.exit(1)


def cmd_api():
    """Launch the Flask API backend (legacy, API-only mode)."""
    try:
        print("[INFO] Launching Flask API backend...")
        api_script = os.path.join(os.path.dirname(__file__), "backend", "app.py")
        subprocess.run([sys.executable, api_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] API server exited with code {e.returncode}")
        sys.exit(1)


def cmd_frontend():
    """Launch the unified web server (frontend + API).

    Starts the Flask server that serves both the static frontend pages
    and the REST API endpoints.  The server checks for a trained model
    at models/model.pkl and loads it automatically.
    """
    try:
        print("[INFO] Launching Aegis AI-IDS web server...")
        print("[INFO] Frontend available at http://localhost:8501")
        server_script = os.path.join(os.path.dirname(__file__), "backend", "app.py")
        subprocess.run([sys.executable, server_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Frontend server exited with code {e.returncode}")
        sys.exit(1)


def cmd_all():
    """Run full pipeline: generate synthetic data, train model, launch frontend."""
    print("[INFO] Running full pipeline...")
    cmd_generate()
    cmd_train()
    cmd_frontend()


def main():
    print_banner()

    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    commands = {
        "generate": cmd_generate,
        "train": cmd_train,
        "app": cmd_app,
        "api": cmd_api,
        "frontend": cmd_frontend,
        "all": cmd_all,
    }

    if command in commands:
        commands[command]()
    else:
        print(f"[ERROR] Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
