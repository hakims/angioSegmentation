
import subprocess
import sys
import shutil
import importlib.util
from pathlib import Path

REQUIREMENTS_FILE = "requirements.txt"
SAM_CHECKPOINT = Path("dr_sam_core/checkpoints/sam_vit_h.pth")
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

def install_package(package, index_url=None):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if isinstance(package, list):
        cmd += package
    else:
        cmd.append(package)
    if index_url:
        cmd += ["--index-url", index_url]
    subprocess.check_call(cmd)

def is_installed(module):
    return importlib.util.find_spec(module) is not None

def check_and_install_torch():
    try:
        import torch
        if shutil.which("nvidia-smi") and not torch.cuda.is_available():
            print("⚠️ PyTorch is CPU-only but GPU is available. Upgrading to CUDA version...")
            install_package(["torch", "torchvision"], "https://download.pytorch.org/whl/cu118")
        else:
            print("✅ PyTorch already installed and valid.")
    except ImportError:
        print("📦 PyTorch not found. Installing...")
        has_gpu = shutil.which("nvidia-smi") is not None
        index_url = "https://download.pytorch.org/whl/cu118" if has_gpu else "https://download.pytorch.org/whl/cpu"
        install_package(["torch", "torchvision"], index_url)

def install_core_requirements():
    print(f"📋 Installing required packages from {REQUIREMENTS_FILE}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])

def download_sam_weights():
    if SAM_CHECKPOINT.exists() and SAM_CHECKPOINT.stat().st_size > 10_000_000:
        print(f"✅ SAM model already downloaded at {SAM_CHECKPOINT}")
        return
    SAM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading SAM weights to {SAM_CHECKPOINT}...")
    import requests
    with requests.get(SAM_URL, stream=True) as r:
        r.raise_for_status()
        with open(SAM_CHECKPOINT, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("✅ SAM model download complete.")

def main():
    print("🚀 Setting up Dr-SAM dependencies...\n")
    install_core_requirements()
    check_and_install_torch()
    download_sam_weights()
    print("\n🎉 All dependencies installed and verified!")

if __name__ == "__main__":
    main()
