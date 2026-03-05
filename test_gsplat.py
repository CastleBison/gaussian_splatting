import torch
import gsplat
from gsplat import rasterization
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda")
H, W = 256, 256

means_list = []
colors_list = []

# Create a colorful cube
for x in range(-2, 3):
    for y in range(-2, 3):
        for z in range(-2, 3):
            means_list.append([x * 0.5, y * 0.5, z * 0.5])
            colors_list.append([
                (x + 2) / 4,
                (y + 2) / 4,
                (z + 2) / 4
            ])

# Add a sphere pattern
for i in range(50):
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    r = 2.5
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    means_list.append([x, y, z])
    colors_list.append([1, 0.5, 0])

means = torch.tensor(means_list, dtype=torch.float32, device=device)
colors = torch.tensor(colors_list, dtype=torch.float32, device=device)
scales = torch.ones(len(means), 3, device=device) * 0.25
quats = torch.zeros(len(means), 4, device=device)
quats[:, 0] = 1
opacities = torch.ones(len(means), device=device) * 0.8

viewmat = torch.eye(4, device=device).unsqueeze(0)
viewmat[:, :3, 3] = torch.tensor([0, 0, -6], device=device)

focal = W / 2
Ks = torch.tensor([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], device=device).unsqueeze(0)

rendered_image, alpha, info = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmat,
    Ks=Ks,
    width=W,
    height=H,
)

print(f"Gaussians: {len(means)}")
print(f"Rendered image shape: {rendered_image.shape}")

img = rendered_image[0].cpu().detach().numpy()
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title("Gaussian Splatting - Colorful Cube + Sphere")
plt.axis('off')
plt.savefig("result.png", dpi=150, bbox_inches='tight')
print("Saved to result.png")
