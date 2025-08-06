from flask import Flask, request, send_file, render_template
from io import BytesIO
import torch
import yaml
from torchvision.utils import save_image
import os

from models.generator import ConditionalSpriteGenerator
from utils.metadata_config import MetadataEncoder

app = Flask(__name__)

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

encoder = MetadataEncoder(config)
z_dim = config["train"]["z_dim"]
metadata_keys = list(config["metadata"].keys())

# Load model
model = ConditionalSpriteGenerator(z_dim, encoder.meta_dim)
model.load_state_dict(torch.load(config["train"]["save_path"], map_location="cpu"))
model.eval()

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    if not request.is_json:
        return {"error": "Request must be JSON"}, 400
    data = request.json
    try:
        meta = {}
        for key in metadata_keys:
            val = data.get(key)
            options = config["metadata"][key]
            meta[key] = parse_arg(val, options)

        z = torch.randn(1, z_dim)
        meta_vec = encoder.encode(meta).unsqueeze(0)
        with torch.no_grad():
            output = model(z, meta_vec)

        buffer = BytesIO()
        save_image((output + 1) / 2, buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")
    except Exception as e:
        return str(e), 400

@app.route("/config")
def get_config():
    try:
        return config["metadata"]
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)
