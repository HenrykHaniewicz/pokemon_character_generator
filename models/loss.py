import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # We will extract features from these layers (after ReLU)
        self.slice1 = vgg[:4]   # relu1_2
        self.slice2 = vgg[4:9]  # relu2_2
        self.slice3 = vgg[9:16] # relu3_3
        self.slice4 = vgg[16:23]# relu4_3

        self.layer_weights = layer_weights or [1.0, 0.75, 0.5, 0.25]
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        # Assume inputs are in [0, 1], convert to VGG input format
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        loss = 0.0
        for i, (slice_fn, w) in enumerate(zip([self.slice1, self.slice2, self.slice3, self.slice4], self.layer_weights)):
            x = slice_fn(x)
            y = slice_fn(y)
            loss += w * F.l1_loss(x, y)
        return loss
