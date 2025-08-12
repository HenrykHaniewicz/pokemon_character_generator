import sys
import os
import torch
import yaml
import errno
from torchvision.utils import save_image
from models.generator import ConditionalSpriteGenerator
from utils.metadata_config import MetadataEncoder
from utils.util import make_unique_path, load_config, parse_arg

def generate_character(args):
    config = load_config("config.yaml")

    encoder = MetadataEncoder(config)
    metadata_keys = list(config["metadata"].keys())

    if len(args) != len(metadata_keys):
        print(f"Usage: python generate_character.py {' '.join(metadata_keys)}")
        print(f"Example: python generate_character.py {' '.join(str(v[0]) for v in config['metadata'].values())}")
        sys.exit(1)
        return

    meta = {}
    for key, val in zip(metadata_keys, args):
        options = config["metadata"][key]
        try:
            meta[key] = parse_arg(val, options)
        except ValueError as e:
            print(f"Error for '{key}': {e}")
            sys.exit(1)
            return

    z_dim = config["train"]["z_dim"]
    output_dir = config["generate"]["output_dir"]

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST or not os.path.isdir(output_dir):
            raise

    # Load model
    model = ConditionalSpriteGenerator(z_dim, encoder.meta_dim)
    model.load_state_dict(torch.load(config["train"]["save_path"], map_location="cpu", weights_only=False))
    model.eval()

    # Generate sprite
    z = torch.randn(1, z_dim)
    meta_vec = encoder.encode(meta).unsqueeze(0)
    with torch.no_grad():
        output = model(z, meta_vec)

    base = "_".join(str(meta[k]) for k in metadata_keys)
    save_path = make_unique_path(output_dir, base, ext=".png")
    save_image((output + 1) / 2, save_path)
    print(f"✅ Saved sprite to: {save_path}")

if __name__ == "__main__":
    generate_character(sys.argv[1:])
