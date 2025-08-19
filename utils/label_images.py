import os
import json

from utils.util import open_image_non_blocking, load_config, get_image_files

def load_existing_labels(label_path):
    if os.path.exists(label_path):
        with open(label_path) as f:
            return json.load(f)
    return {}

def prompt_choice(prompt_text, options):
    while True:
        print(f"{prompt_text}")
        for i, opt in enumerate(options, start=1):
            print(f"{i}. {opt}")
        raw = input("> ").strip().lower()

        # Try match by number (1-indexed)
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]

        # Try match by name
        if raw in options:
            return raw

        print("Invalid input. Please choose by number or name.\n")

def prompt_yes_no(prompt_text):
    while True:
        raw = input(f"{prompt_text} (y/n): ").strip().lower()
        if raw in ["y", "yes"]:
            return True
        elif raw in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'.\n")

def prompt_user_for_metadata(fname, metadata_config):
    print(f"\nLabeling: {fname}")
    metadata = {}

    for key, options in metadata_config.items():
        # Handle boolean options
        if all(isinstance(opt, bool) for opt in options):
            val = prompt_yes_no(f"{key.replace('_', ' ').capitalize()}?")
            metadata[key] = val
        else:
            val = prompt_choice(f"{key.replace('_', ' ').capitalize()}:", options)
            metadata[key] = val

    return metadata

def label_images(image_dir, label_path, metadata_config):
    labels = load_existing_labels(label_path)
    existing = set(labels.keys())
    image_files = get_image_files(image_dir)

    for fname in image_files:
        if fname in existing:
            continue
        img_path = os.path.join(image_dir, fname)
        proc = open_image_non_blocking(img_path)

        metadata = prompt_user_for_metadata(fname, metadata_config)
        labels[fname] = metadata

        if proc:
            try:
                proc.terminate()
            except Exception:
                pass

        with open(label_path, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"Saved metadata for {fname}\n")

if __name__ == "__main__":
    config = load_config()
    label_images(
        image_dir=config["train"]["data_dir"],
        label_path=config["train"]["labels_file"],
        metadata_config=config["metadata"]
    )
