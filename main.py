import torch
from models.generator import ConditionalSpriteGenerator
from models.trainer import train_conditional_generator, train_conditional_generator_with_quality
from utils.dataset import SpriteDataset, GradedSpriteDataset
from utils.metadata_config import MetadataEncoder, random_metadata
from utils.util import pick_device, load_config, make_unique_path
from torchvision import transforms
from torch.utils.data import DataLoader
import json
import os
from torchvision.utils import save_image
import yaml
import argparse

config = load_config("config.yaml")

def generate_samples(generator, z_dim, encoder, output_dir, device, num_samples,
                     metadata_keys, allow_grading=False):
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)
    grades = {}

    for i in range(num_samples):
        z = torch.randn(1, z_dim).to(device)
        meta = random_metadata(encoder)
        meta_vec = encoder.encode(meta).unsqueeze(0).to(device)

        # If graded model, always condition on quality=10 → 1.0
        if getattr(generator, "expects_quality", False):
            q = torch.tensor([[1.0]], device=device, dtype=meta_vec.dtype)
            meta_vec = torch.cat([meta_vec, q], dim=1)

        with torch.no_grad():
            output = generator(z, meta_vec)

        base = "_".join(str(meta[k]) for k in metadata_keys)
        file_path = make_unique_path(output_dir, base, ext=".png")
        save_image((output.cpu() + 1) / 2, file_path)
        print(f"\nGenerated {file_path} with original meta: {meta}")

        if allow_grading:
            graded_meta = {}
            print("Please enter corrected metadata for this image (press Enter to keep original):")
            for key in metadata_keys:
                default_val = meta.get(key, "")
                user_input = input(f"  {key} (default: {default_val}): ").strip()
                graded_meta[key] = default_val if user_input == "" else user_input

            while True:
                q = input("Rate the image quality from 1 (poor) to 10 (excellent): ").strip()
                try:
                    qv = int(q)
                    if 1 <= qv <= 10:
                        break
                except:
                    pass
                print("Please enter an integer 1–10.")

            grades[file_path] = {
                "meta_original": meta,
                "meta_corrected": graded_meta,
                "quality_rating": qv
            }

    if allow_grading:
        out_json = os.path.join(output_dir, "grades.json")
        with open(out_json, "w") as f:
            json.dump(grades, f, indent=2)
        print(f"\nSaved feedback ratings to {out_json}")


def build_transforms():
    return transforms.Compose([
        transforms.Resize((192, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graded-train", action="store_true", help="Use the graded (quality-aware) training pipeline.")
    args = parser.parse_args()

    encoder = MetadataEncoder(config)
    z_dim = config["train"]["z_dim"]
    batch_size = config["train"]["batch_size"]
    epochs = config["train"]["epochs"]
    device = pick_device()
    transform = build_transforms()

    metadata_keys = list(config["metadata"].keys())

    if not args.graded_train:
        dataset = SpriteDataset(config["train"]["data_dir"], config["train"]["labels_file"], transform, encoder=encoder)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config["train"]["cores"])

        model = ConditionalSpriteGenerator(z_dim, encoder.meta_dim).to(device)
        model.expects_quality = False
        train_conditional_generator(model, dataloader, z_dim, device, epochs)

    else:
        graded_file = config["quality_graded_train"].get("graded_file")
        if not graded_file:
            graded_file = os.path.join(config["generate"]["output_dir"], "grades.json")
        include_orig = None
        if config["quality_graded_train"].get("include_original", False):
            include_orig = (config["train"]["data_dir"], config["train"]["labels_file"])

        dataset = GradedSpriteDataset(
            graded_file=graded_file,
            transform=transform,
            encoder=encoder,
            include_original=include_orig,
            min_quality=config["quality_graded_train"].get("min_quality", 7)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config["train"]["cores"])

        # +1 is due to the quality data
        model = ConditionalSpriteGenerator(z_dim, encoder.meta_dim + 1).to(device)
        model.expects_quality = True

        train_conditional_generator_with_quality(
            model,
            dataloader,
            z_dim,
            device,
            epochs,
            lambda_outline=config["quality_graded_train"].get("lambda_outline", 0.5),
            lambda_perceptual=config["quality_graded_train"].get("lambda_perceptual", 1.0),
            use_quality_condition=True,
            lambda_quality=0.0
        )


    generate_samples(
        model,
        z_dim,
        encoder,
        config["generate"]["output_dir"],
        device,
        config["generate"]["num_samples"],
        metadata_keys,
        allow_grading=config["generate"].get("allow_grading", False)
    )


if __name__ == "__main__":
    device = pick_device()

    encoder = MetadataEncoder(config)
    z_dim = config["train"]["z_dim"]

    model = ConditionalSpriteGenerator(z_dim, encoder.meta_dim + 1)
    model.load_state_dict(torch.load(config["train"]["save_path"], map_location="cpu", weights_only=False))

    model.expects_quality = True

    metadata_keys = list(config["metadata"].keys())

    generate_samples(
        model,
        z_dim,
        encoder,
        config["generate"]["output_dir"],
        device,
        config["generate"]["num_samples"],
        metadata_keys,
        allow_grading=config["generate"].get("allow_grading", False)
    )

    # main()
