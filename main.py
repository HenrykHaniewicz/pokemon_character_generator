import torch
from models.generator import ConditionalSpriteGenerator
from models.trainer import train_conditional_generator
from utils.dataset import SpriteDataset
from utils.metadata_config import MetadataEncoder, random_metadata
from torchvision import transforms
from torch.utils.data import DataLoader
import json
import os
from torchvision.utils import save_image
import yaml


def generate_samples_with_grading(generator, z_dim, encoder, output_dir, device, allow_grading=False):
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)
    grades = {}
    for i in range(10):
        z = torch.randn(1, z_dim).to(device)
        meta = random_metadata(encoder)
        meta_vec = encoder.encode(meta).unsqueeze(0).to(device)
        with torch.no_grad():
            output = generator(z, meta_vec)
        output_cpu = output.cpu()
        file_path = os.path.join(output_dir, f"gen_{i:03}.png")
        save_image((output_cpu + 1) / 2, file_path)
        print(f"Generated {file_path} with meta: {meta}")

        if allow_grading:
            grade = input("Rate the sprite from 1 (poor) to 4 (excellent): ")
            grades[file_path] = {"meta": meta, "grade": int(grade)}

    if allow_grading:
        with open(os.path.join(output_dir, "grades.json"), "w") as f:
            json.dump(grades, f, indent=2)

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    encoder = MetadataEncoder(config)

    z_dim = config["train"]["z_dim"]
    batch_size = config["train"]["batch_size"]
    epochs = config["train"]["epochs"]
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
    transform = transforms.Compose([
        transforms.Resize((192, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = SpriteDataset(config["train"]["data_dir"], config["train"]["labels_file"], transform, encoder=encoder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = ConditionalSpriteGenerator(z_dim, encoder.meta_dim).to(device)
    train_conditional_generator(model, dataloader, z_dim, device, epochs)

    generate_samples_with_grading(
        model,
        z_dim,
        encoder,
        config["generate"]["output_dir"],
        device,
        allow_grading=config["generate"].get("allow_grading", False)
    )

if __name__ == "__main__":
    main()