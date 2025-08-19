# The most utility of all the utilities

import os
import yaml
import torch
from PIL import Image
from torchvision import transforms
import platform
import subprocess

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_image_files(image_dir):
    return sorted([
        f for f in os.listdir(image_dir)
        if not f.startswith(".") and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

def open_image_non_blocking(img_path):
    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            proc = subprocess.Popen(
                ["qlmanage", "-p", img_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return proc
        elif system == "Linux":
            proc = subprocess.Popen(["xdg-open", img_path])
            return proc
        elif system == "Windows":
            # mspaint is simple and usually available
            proc = subprocess.Popen(["mspaint", img_path])
            return proc
        else:
            print("Unknown OS; using PIL fallback (blocking).")
            Image.open(img_path).show()
            return None
    except Exception as e:
        print(f"Failed to open image {img_path}: {e}")
        return None

def parse_arg(value, options):
    if all(isinstance(opt, bool) for opt in options):
        return value.lower() in ["true", "1", "yes"]
    if value.isdigit():
        idx = int(value) - 1
        if 0 <= idx < len(options):
            return options[idx]
    if value in options:
        return value
    raise ValueError(f"Invalid input '{value}'. Choose from: {options} or use index.")

def pick_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        import torch_directml
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch_directml.is_available():
            device = torch_directml.device()
            print("Using DirectML for GPU acceleration")
        else:
            print("Using CPU")
    except ImportError:
        print("DirectML not available, using CPU")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
    return device

def make_unique_path(output_dir, base, ext=".png"):
    # try base first
    candidate = os.path.join(output_dir, base + ext)
    if not os.path.exists(candidate):
        return candidate
    # then add a numeric suffix
    i = 1
    while True:
        candidate = os.path.join(output_dir, f"{base}__{i:03d}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1

def build_transforms(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])