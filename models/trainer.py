import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def sobel_edge_detector(img):
    # img shape: (B, C, H, W), expect RGB image scaled between -1 and 1 or 0 and 1
    # Convert to grayscale (simple average)
    gray = img.mean(dim=1, keepdim=True)

    # Sobel kernels
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return edges

def train_conditional_generator(model, dataloader, z_dim, device, epochs, lambda_outline=1.0):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(epochs):
        total_loss = 0.0
        for real_imgs, meta_vecs in dataloader:
            real_imgs = real_imgs.to(device)
            meta_vecs = meta_vecs.to(device)
            z = torch.randn(real_imgs.size(0), z_dim).to(device)
            outputs = model(z, meta_vecs)

            img_loss = criterion(outputs, real_imgs)

            real_edges = sobel_edge_detector(real_imgs)
            gen_edges = sobel_edge_detector(outputs)
            outline_loss = criterion(gen_edges, real_edges)

            loss = img_loss + lambda_outline * outline_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} (Img: {img_loss.item():.4f}, Outline: {outline_loss.item():.4f})")

    torch.save(model.state_dict(), "models/sprite_gen.pt")