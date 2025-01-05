import torch
import numpy as np
import matplotlib.pyplot as plt

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def target_distribution(x):
    """
    Computes the target distribution as a mixture of two 2D Gaussians.

    Parameters:
    -----------
    x : torch.Tensor
        A 2D tensor of shape (n_samples, 2) representing input points.

    Returns:
    --------
    torch.Tensor
        The target distribution values at the input points.
    """
    mu1 = torch.tensor([30.0, 30.0], device=device)
    mu2 = torch.tensor([-30.0, -30.0], device=device)
    sigma = 5.0

    dist1 = torch.exp(-torch.sum((x - mu1) ** 2, dim=1) / (2 * sigma**2))
    dist2 = torch.exp(-torch.sum((x - mu2) ** 2, dim=1) / (2 * sigma**2))

    normalization = 2 * (2 * np.pi * sigma**2)
    return (dist1 + dist2) / normalization

# Generate grid for evaluating the target distribution
x = np.linspace(-50, 50, 100)
y = np.linspace(-50, 50, 100)
X, Y = np.meshgrid(x, y)
grid = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32, device=device)

# Compute the target distribution on the grid
Z = target_distribution(grid).reshape(100, 100).cpu().numpy()

# Plot the target distribution as a contour plot
plt.figure(figsize=(12, 6))
contour = plt.contourf(X, Y, Z, levels=20, cmap="viridis")
plt.colorbar(contour, label="Density")
plt.title("Target Distribution Density")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("target_distribution_contourf.pdf")
plt.show()
