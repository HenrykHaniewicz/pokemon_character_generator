# RPG Maker XP Pokémon Character Generator

Uses a neural network (conditional GAN) to generate Pokémon-style character sprites for RPG Maker XP, based on configurable attributes like gender, color, pose, and more.

---

## How It Works

Neural network training encodes both input images and metadata (e.g., gender, primary color, hat, etc.) and feeds them into a conditional generator model. The model learns to produce sprite-style characters conditioned on this metadata.

You can generate characters through a Flask web interface that dynamically builds forms from your config and lets you generate, preview, and download sprites with a smooth UI including dark mode and loading spinners.

---

## Key Features

- Label your own dataset with custom metadata using `label_images.py`
- Train a conditional GAN with `main.py`
- Serve a Flask web app for interactive character generation with:
  - Dynamic form generation from your config  
  - Dark mode toggle with smooth fade  
  - Loading spinner while generating  
  - Download button for generated sprites
- Configurable attributes stored in `config.yaml`

---

## Usage

### Label your dataset (if needed)

```bash
python label_images.py
```

### Train the model (if needed)

```bash
python main.py
```

This loads your labeled data and trains the generator.

### Run the web app

```bash
python app.py
```

Open your browser at http://127.0.0.1:5000 to access the character generator.

### Generate a character (via CLI, optional)

```bash
python generate_character.py <gender> <top_color> <bottom_color> <age> <hat:true|false>
```

Example:

```bash
python generate_character.py male red blue old true
```

### Requirements

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

### Notes

The Flask app serves a responsive web UI with dark mode, spinner, and download button.
Generated images are returned from the model and previewed live.
CSS and static assets are served from the static/ folder.
Model file location is configured in config.yaml. The app shows a default message if the model is missing.
Outputs (if generated via CLI) are saved in the generated_sprites/ folder.
Customize attributes and metadata options in config.yaml.
Works best with properly labeled and consistent sprite inputs.