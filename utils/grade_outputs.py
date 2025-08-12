import os
import json
import argparse

from utils.util import load_config, get_image_files, open_image_non_blocking

def load_existing_grades(grades_path):
    if os.path.exists(grades_path):
        try:
            with open(grades_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: could not parse existing {grades_path}: {e}")
    return {}

def prompt_choice(prompt_text, options):
    """Choose from a fixed list of options (case-insensitive, supports numeric selection)."""
    options_norm = [str(o).lower() for o in options]
    while True:
        print(f"{prompt_text}")
        for i, opt in enumerate(options, start=1):
            print(f"{i}. {opt}")
        raw = input("> ").strip().lower()

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]

        if raw in options_norm:
            return options[options_norm.index(raw)]

        print("Invalid input. Please choose by number or name.\n")

def prompt_yes_no(prompt_text):
    while True:
        raw = input(f"{prompt_text} (y/n): ").strip().lower()
        if raw in ["y", "yes"]:
            return True
        if raw in ["n", "no"]:
            return False
        print("Please enter 'y' or 'n'.\n")

def prompt_quality():
    while True:
        raw = input("Rate image quality 1–10: ").strip()
        try:
            q = int(raw)
            if 1 <= q <= 10:
                return q
        except:
            pass
        print("Please enter an integer 1–10.\n")

def prompt_user_for_metadata(fname, metadata_config):
    """
    metadata_config expected like your label_images: a dict:
      key -> list of options (booleans or strings/ints)
    """
    print(f"\nGrading metadata for: {fname}")
    metadata = {}

    for key, options in metadata_config.items():
        if all(isinstance(opt, bool) for opt in options):
            val = prompt_yes_no(f"{key.replace('_', ' ').capitalize()}?")
            metadata[key] = val
        else:
            val = prompt_choice(f"{key.replace('_', ' ').capitalize()}:", options)
            metadata[key] = val

    return metadata

def parse_meta_from_filename(fname, metadata_config):
    """Parse original metadata values from a filename like male_red_blue_old_True__001.png"""
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")
    keys = list(metadata_config.keys())

    if len(parts) < len(keys):
        print(f"Warning: {fname} does not match expected metadata format")
        return {}

    # Only take the first N parts for metadata; ignore any extra suffix (e.g., __001)
    parts = parts[:len(keys)]

    meta = {}
    for key, val in zip(keys, parts):
        opts = metadata_config[key]
        if all(isinstance(opt, bool) for opt in opts):
            meta[key] = val.lower() in ["true", "1", "yes", "y"]
        elif all(isinstance(opt, int) for opt in opts):
            try:
                meta[key] = int(val)
            except ValueError:
                meta[key] = val
        elif all(isinstance(opt, float) for opt in opts):
            try:
                meta[key] = float(val)
            except ValueError:
                meta[key] = val
        else:
            meta[key] = val
    return meta

def grade_outputs(sample_dir, grades_path, metadata_config, overwrite=False):
    """
    Iterate images in sample_dir, prompt for metadata + quality, and write to grades.json
    Keys are full file paths to be consistent with graded dataset loader.
    """
    os.makedirs(sample_dir, exist_ok=True)
    grades = load_existing_grades(grades_path)
    already = set(grades.keys())

    image_files = get_image_files(sample_dir)
    if not image_files:
        print(f"No images found in {sample_dir}.")
        return

    for fname in image_files:
        img_path = os.path.join(sample_dir, fname)

        if (img_path in already) and not overwrite:
            print(f"Skipping (already graded): {fname}")
            continue

        proc = open_image_non_blocking(img_path)

        meta_corrected = prompt_user_for_metadata(fname, metadata_config)
        quality = prompt_quality()
        meta_original = parse_meta_from_filename(fname, metadata_config)
        
        grades[img_path] = {
            "meta_original": meta_original,
            "meta_corrected": meta_corrected,
            "quality_rating": quality
        }

        # Close previewer if possible
        if proc:
            try:
                proc.terminate()
            except Exception:
                pass

        with open(grades_path, "w") as f:
            json.dump(grades, f, indent=2)
        print(f"Saved grade for {fname}\n")

    print(f"All done. Grades saved to {grades_path}")

def main():
    parser = argparse.ArgumentParser(description="Grade generated outputs with metadata + quality.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--samples", default=None,
                        help="Directory with generated images. Defaults to config.generate.output_dir")
    parser.add_argument("--out", default=None,
                        help="Path to grades.json. Defaults to <samples>/grades.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-grade images even if already present in grades.json")
    args = parser.parse_args()

    config = load_config(args.config)
    sample_dir = args.samples or config["generate"]["output_dir"]
    grades_path = args.out or os.path.join(sample_dir, "grades.json")
    metadata_config = config["metadata"]

    grade_outputs(sample_dir, grades_path, metadata_config, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
