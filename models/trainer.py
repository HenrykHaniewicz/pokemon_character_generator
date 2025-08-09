import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    perceptual_losses= []
    outline_losses = []
    total_losses = []
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
        perceptual_losses.append(perceptual_loss.item())
        outline_losses.append(outline_loss.item())
        total_losses.append(avg_loss)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} (Perceptual: {perceptual_loss.item():.4f}, Outline: {outline_loss.item():.4f})")
    plt.plot(range(1, epochs + 1), total_losses, label='Total Loss')
    plt.plot(range(1, epochs + 1), perceptual_losses, label='Perceptual Loss')
    plt.plot(range(1, epochs + 1), outline_losses, label='Outline Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("models/loss_plot.png")
    plt.close()
    torch.save(model.state_dict(), "models/sprite_gen.pt")


def train_conditional_generator_with_quality(model, dataloader, z_dim, device, epochs, lambda_outline=0.5, lambda_perceptual=1.0,
                                                use_quality_condition=True, lambda_quality=0.0):
    if len(dataloader) == 0:
        print("Dataloader is empty. Skipping training.")
        torch.save(model.state_dict(), "models/latest.pt")
        return

    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    perceptual_losses = []
    outline_losses = []
    quality_losses = []
    total_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        for real_imgs, meta_vecs, quality in dataloader:
            real_imgs = real_imgs.to(device)  # assumed normalized to [-1, 1]
            meta_vecs = meta_vecs.to(device)
            quality = quality.float().to(device).unsqueeze(1) / 10.0  # normalize 1-10 → 0-1

            # Optionally add quality to conditioning vector
            if use_quality_condition:
                meta_vecs = torch.cat([meta_vecs, quality], dim=1)

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
            
            # Optional: quality regression loss if model predicts quality (needs quality head in model)
            if lambda_quality > 0 and hasattr(model, "predict_quality"):
                pred_quality = model.predict_quality(outputs).squeeze(1)
                quality_loss = F.mse_loss(pred_quality, quality.squeeze(1))
            else:
                quality_loss = torch.tensor(0.0, device=device)

            loss = lambda_perceptual * perceptual_loss + lambda_outline * outline_loss + lambda_quality * quality_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        perceptual_losses.append(perceptual_loss.item())
        outline_losses.append(outline_loss.item())
        quality_losses.append(quality_loss.item() if lambda_quality > 0 else 0.0)
        total_losses.append(avg_loss)

        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} "f"(Perceptual: {perceptual_loss.item():.4f}, Outline: {outline_loss.item():.4f}, Quality: {quality_loss.item():.4f})")

    # Plot
    plt.plot(range(1, epochs + 1), total_losses, label='Total Loss')
    plt.plot(range(1, epochs + 1), perceptual_losses, label='Perceptual Loss')
    plt.plot(range(1, epochs + 1), outline_losses, label='Outline Loss')
    if lambda_quality > 0:
        plt.plot(range(1, epochs + 1), quality_losses, label='Quality Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("models/loss_plot_graded.png")
    plt.close()

    torch.save(model.state_dict(), "models/sprite_gen_graded.pt")
