# RPG Maker XP Pokémon Character Generator

Uses a neural network (conditional GAN) to generate Pokémon-style character sprites for RPG Maker XP, based on configurable attributes like gender, color, pose, and more.

## How It Works

Neural network training is achieved by encoding both the input image and metadata (e.g., gender, primary color, hat, etc.) and feeding them into a conditional generator model. The model learns to produce sprite-style characters conditioned on this metadata.

## Key Features

- Label your own dataset with custom metadata using `label_images.py`
- Train a conditional GAN with `main.py`
- Generate new character sprites using `generate_character.py`
- Configurable attributes stored in `config.yaml`

## Usage

### Label your dataset if needed

```bash
python label_images.py
```

### Train the model if needed

```bash
python main.py
```

This loads your labeled data and trains the generator.


### Generate a character

```bash
python generate_character.py <gender> <top_color> <bottom_color> <age> <hat:true|false>
```

Example:

```bash
python generate_character.py male red blue old true
```

### Requirements

Python 3.8+
torch
torchvision
pillow
pyyaml


Install dependencies with:

```bash
pip install -r requirements.txt
```

### Notes

Outputs are saved in the `generated_sprites/` folder.
You can customize attributes in `config.yaml`.
Works best with properly labeled and consistent sprite inputs.