# RPG Maker XP Pokémon Character Generator

A neural network (conditional GAN) that generates Pokémon-style character sprites for RPG Maker XP, conditioned on configurable attributes such as gender, colors, clothing, and more.  
Supports **human-graded training** for improved quality control.

---

## How It Works

- **Training**: The network takes both input sprites and their associated metadata (e.g., gender, primary color, has_hat, pose) and trains a conditional generator model.  
- **Generation**: You can produce new sprites either right after training or at any time later using the trained model.  
- **Grading**: Generated sprites can be graded interactively (metadata corrections + quality score) and used in a *graded training pipeline* to improve future results.

---

## Key Features

- **Metadata labeling** for your dataset with `label_images.py`.
- **Two training modes**:
  - **Original** — standard conditional GAN training from metadata.
  - **Graded** — uses human-graded metadata and quality ratings as an extra conditioning signal.
- **Sample generation & grading** directly after training in `main.py`.
- **Standalone grading** with `grade_outputs.py` for generated images without retraining first.
- **Configurable attributes** stored in `config.yaml` and read by all scripts.
- **Flexible dataset handling** — graded and original data can be trained separately or combined.
- Optional **Flask web interface** for interactive character generation.

---

## Usage

### 1. Label your dataset (if needed)

```bash
python -m utils.label_images
```

### Train the model (if needed)

Original pipeline (no quality rating):

```bash
python main.py
```

Graded pipeline (quality-aware, uses grades.json from grading step):

```bash
python main.py --graded-train
```

This loads your labeled data and trains the generator.

### Generate & grade samples immediately after training

If `allow_grading` is set to true in `config.yaml`, `main.py` will:

- Generate samples.
- Prompt the user to correct metadata.
- Ask for a quality rating (1–10).
- Save feedback to `<output_dir>/grades.json`.


Standalone grading (no training)

You can grade any folder of generated sprites later, without retraining:

```bash
python -m utils.grade_outputs --samples <path/to/generated> --out <path/to/generated>/grades.json
```

This will:

- Open each image.
- Prompt for corrected metadata.
- Prompt for a quality rating.
- Save results in `grades.json`.

### Run the web app

```bash
python app.py
```

Open your browser at http://127.0.0.1:5000 to access the interactive character generator.

### Generate a character (via CLI, optional)

```bash
python generate_character.py <gender> <top_color> <bottom_color> <age> <hat:true|false>
```

Example:

```bash
python generate_character.py male red blue old true
```

## Requirements

- Python 3.8+
- torch
- torchvision
- pillow
- pyyaml
- flask

Install dependencies with:

```bash
pip install -r requirements.txt
```