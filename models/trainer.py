import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.loss import VGGPerceptualLoss

def sobel_edge_detector(img):
    # img shape: (B, C, H, W)
    gray = img.mean(dim=1, keepdim=True)

    # Pad with replication to avoid artificial edges at border
    gray_padded = F.pad(gray, (1, 1, 1, 1), mode='replicate')

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                           dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                           dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)

    grad_x = F.conv2d(gray_padded, sobel_x)
    grad_y = F.conv2d(gray_padded, sobel_y)

    edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return edges

def train_conditional_generator(model, dataloader, z_dim, device, epochs, lambda_outline=0.5, lambda_perceptual=1.0):
    if len(dataloader) == 0:
        print("Dataloader is empty. Skipping training.")
        torch.save(model.state_dict(), "models/latest.pt")
        return

    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(epochs):
        total_loss = 0.0
        for real_imgs, meta_vecs in dataloader:
            real_imgs = real_imgs.to(device)  # assumed normalized to [-1, 1]
            meta_vecs = meta_vecs.to(device)
            z = torch.randn(real_imgs.size(0), z_dim).to(device)

            outputs = model(z, meta_vecs)

            # Scale outputs and targets from [-1, 1] → [0, 1] for VGG
            outputs_01 = (outputs + 1) / 2
            real_imgs_01 = (real_imgs + 1) / 2

            # Ensure 3 channels for VGG
            if outputs_01.shape[1] == 1:
                outputs_vgg = outputs_01.repeat(1, 3, 1, 1)
                real_imgs_vgg = real_imgs_01.repeat(1, 3, 1, 1)
            else:
                outputs_vgg = outputs_01
                real_imgs_vgg = real_imgs_01

            # Losses
            perceptual_loss = perceptual_loss_fn(outputs_vgg, real_imgs_vgg)
            outline_loss = F.l1_loss(sobel_edge_detector(outputs), sobel_edge_detector(real_imgs))

            # Total loss
            loss = lambda_perceptual * perceptual_loss + lambda_outline * outline_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} (Perceptual: {perceptual_loss.item():.4f}, Outline: {outline_loss.item():.4f})")

    torch.save(model.state_dict(), "models/sprite_gen.pt")