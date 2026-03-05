import torch
import gsplat
from gsplat import rasterization
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda")
torch.manual_seed(42)

H, W = 128, 128

target_means = torch.tensor([
    [0.0, 0.0, 3.0],
    [0.5, 0.5, 3.5],
    [-0.5, -0.5, 3.2],
    [0.3, -0.3, 3.8],
    [-0.3, 0.4, 3.3],
], device=device)

target_colors = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
], device=device)

means = torch.randn(5, 3, device=device) * 0.5 + 1.0
means[:, 2] = 4.0

scales = torch.ones(5, 3, device=device) * 0.2
quats = torch.zeros(5, 4, device=device)
quats[:, 0] = 1

colors = torch.rand(5, 3, device=device)
opacities = torch.ones(5, device=device) * 0.5

focal = W / 2
Ks = torch.tensor([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], device=device).unsqueeze(0)

target_view = torch.eye(4, device=device).unsqueeze(0)
target_view[:, :3, 3] = torch.tensor([0, 0, 0], device=device)

rendered, alpha, _ = rasterization(
    means=target_means, quats=quats[:5], scales=scales[:5],
    opacities=opacities[:5], colors=target_colors,
    viewmats=target_view, Ks=Ks, width=W, height=H,
)
target_img = rendered[0].detach()

means.requires_grad = True
scales.requires_grad = True
colors.requires_grad = True
opacities.requires_grad = True

optimizer = torch.optim.Adam([means, scales, colors, opacities], lr=0.05)

print("Training Gaussian Splatting...")
print(f"Initial means:\n{means}")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for step in range(50):
    optimizer.zero_grad()
    
    rendered, alpha, _ = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        viewmats=target_view, Ks=Ks, width=W, height=H,
    )
    
    loss = ((rendered[0] - target_img) ** 2).mean()
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
        img = rendered[0].cpu().detach().numpy()
        idx = step // 10
        axes[0, idx].imshow(np.clip(img, 0, 1))
        axes[0, idx].set_title(f"Step {step}")
        axes[0, idx].axis('off')

axes[1, 4].imshow(target_img.cpu().numpy())
axes[1, 4].set_title("Target")
axes[1, 4].axis('off')

plt.tight_layout()
plt.savefig("training_result.png", dpi=150)
print("\nFinal means:")
print(means)
print("Saved to training_result.png")
