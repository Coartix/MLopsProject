import subprocess
import argparse
import signal
import sys

def start_streamlit():
    return subprocess.Popen(["streamlit", "run", "Home.py"])

def start_gunicorn():
    return subprocess.Popen(["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "api:app", "--reload", "--timeout", "1000"])

def train_models():
    subprocess.run(["python3", "train_model.py"])
    subprocess.run(["python3", "train_autoencoder.py"])

def signal_handler(sig, frame):
    print('Stopping all services...')
    streamlit_process.terminate()
    gunicorn_process.terminate()
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run web services and optionally train models.')
    parser.add_argument('--train', action='store_true', help='Train the model if set')
    args = parser.parse_args()

    # Train models if requested
    if args.train:
        train_models()

    # Start Streamlit and Gunicorn
    streamlit_process = start_streamlit()
    gunicorn_process = start_gunicorn()

    # Handle CTRL+C
    signal.signal(signal.SIGINT, signal_handler)

    # Wait for subprocesses
    streamlit_process.wait()
    gunicorn_process.wait()
