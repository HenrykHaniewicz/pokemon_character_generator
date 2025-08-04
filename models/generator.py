import torch
import torch.nn as nn

class ConditionalSpriteGenerator(nn.Module):
    def __init__(self, z_dim, meta_dim):
        super().__init__()
        if (z_dim < 1 or meta_dim < 1):
            raise ValueError("Issue when loading ConditionalSpriteGenerator. Dimensions must be positive.")
        self.fc = nn.Sequential(
            nn.Linear(z_dim + meta_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 192 * 128),
            nn.Tanh()
        )

    def forward(self, z, metadata):
        x = torch.cat([z, metadata], dim=1)
        x = self.fc(x)
        return x.view(-1, 3, 192, 128)