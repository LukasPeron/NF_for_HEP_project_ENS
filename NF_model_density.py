import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

def kde_density(samples, x_grid, y_grid):
    """
    Estimate the density of samples using Kernel Density Estimation (KDE).

    Parameters:
    -----------
    samples : np.ndarray
        Array of shape (n_samples, dim) containing samples.
    x_grid : np.ndarray
        2D grid of x-coordinates for density evaluation.
    y_grid : np.ndarray
        2D grid of y-coordinates for density evaluation.

    Returns:
    --------
    np.ndarray
        2D array of the same shape as x_grid and y_grid containing density estimates.
    """
    kde = gaussian_kde(samples.T)  # Transpose to match expected input shape [dim, n_samples]
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])  # Combine grid points
    densities = kde(grid_points)  # Evaluate KDE at grid points
    return densities.reshape(x_grid.shape)

class PlanarFlow(nn.Module):
    """
    Defines a single planar flow transformation for normalizing flows.
    """
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(dim, device=device))
        self.w = nn.Parameter(torch.randn(dim, device=device))
        self.b = nn.Parameter(torch.randn(1, device=device))

    def forward(self, z):
        linear = torch.matmul(z, self.w) + self.b
        activation = torch.tanh(linear)
        z_out = z + self.u * activation.unsqueeze(1)
        return z_out

    def log_det_jacobian(self, z):
        linear = torch.matmul(z, self.w) + self.b
        activation = torch.tanh(linear)
        grad_activation = 1 - activation**2
        psi = grad_activation.unsqueeze(1) * self.w
        det = torch.abs(1 + torch.sum(psi * self.u, dim=1))
        return torch.log(torch.clamp(det, min=1e-6))

class NormalizingFlow(nn.Module):
    """
    Implements a normalizing flow model as a sequence of planar flows.
    """
    def __init__(self, dim, n_flows):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])

    def forward(self, z):
        log_jacobian = 0
        for flow in self.flows:
            z = flow(z)
            log_jacobian += flow.log_det_jacobian(z)
        return z, log_jacobian

def load_trained_model(checkpoint_path, dim, n_flows):
    """
    Load a trained Normalizing Flow model from a checkpoint.

    Parameters:
    -----------
    checkpoint_path : str
        Path to the checkpoint file containing the saved model state.
    dim : int
        Dimensionality of the data.
    n_flows : int
        Number of flow layers in the Normalizing Flow model.

    Returns:
    --------
    NormalizingFlow
        The loaded Normalizing Flow model in evaluation mode.
    """
    model = NormalizingFlow(dim, n_flows).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def sample_from_nf(model, n_samples):
    """
    Sample points from a Normalizing Flow model.

    Parameters:
    -----------
    model : NormalizingFlow
        The trained Normalizing Flow model.
    n_samples : int
        Number of samples to generate.

    Returns:
    --------
    np.ndarray
        An array of shape (n_samples, dim) containing samples.
    """
    base_samples = torch.normal(0, 5, size=(n_samples, 2), device=device)
    nf_samples, _ = model(base_samples)
    return nf_samples.detach().cpu().numpy()

def nf_density_example():
    """
    Load the trained NF model, sample from it, estimate density using KDE, 
    and plot the results as a contour plot.
    """
    # Model parameters
    checkpoint_path = 'best_model.pth'  # Path to the saved model checkpoint
    dim = 2
    n_flows = 40

    # Load the trained model
    trained_model = load_trained_model(checkpoint_path, dim, n_flows)

    # Sample from the NF model
    n_samples = 100000
    nf_samples = sample_from_nf(trained_model, n_samples)

    # Generate grid for density estimation
    grid_size = 100
    x = np.linspace(-50, 50, grid_size)
    y = np.linspace(-50, 50, grid_size)
    X, Y = np.meshgrid(x, y)

    # Estimate density using KDE
    nf_density = kde_density(nf_samples, X, Y)

    # Plot density using contourf
    plt.figure(figsize=(12, 6))
    plt.contourf(X, Y, nf_density, levels=50, cmap="viridis")
    plt.title("NF Density Estimation from Samples")
    plt.xlabel("x")
    plt.ylabel("y")

    # Save and show the plot
    plt.savefig("nf_density_contour.pdf")
    plt.show()

# Execute the visualization
if __name__ == "__main__":
    nf_density_example()
