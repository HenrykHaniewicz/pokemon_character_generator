from flask import Flask, request, send_file, render_template
from io import BytesIO
import torch
from torchvision.utils import save_image
import os

from models.generator import ConditionalSpriteGenerator
from utils.metadata_config import MetadataEncoder
from utils.util import parse_arg, load_config

def create_app(test_config=None):
    app = Flask(__name__)

    # Load config
    config = load_config("config.yaml")

    encoder = MetadataEncoder(config)
    z_dim = config["train"]["z_dim"]
    metadata_keys = list(config["metadata"].keys())

    # Check for model
    model_path = config["train"]["save_path"]
    model_available = os.path.exists(model_path)

    model = None
    if model_available:
        model = ConditionalSpriteGenerator(z_dim, encoder.meta_dim)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
        model.eval()

    @app.route("/")
    def home():
        return render_template("index.html", model_available=model_available)

    @app.route("/generate", methods=["POST"])
    def generate():
        if not model_available:
            return "Model file is missing. Cannot generate character.", 500

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

    return app

# Only run server if script is executed directly
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
