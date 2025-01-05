import torch
import torch.nn as nn
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

def base_distribution(n_samples):
    """
    Samples points from the base distribution: a 2D Gaussian centered at 0.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate.

    Returns:
    --------
    torch.Tensor
        A tensor of shape (n_samples, 2) representing the samples.
    """
    return torch.normal(0, 5, size=(n_samples, 2)).to(device)

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

def target_density(x):
    """
    Wrapper for the target distribution function.

    Parameters:
    -----------
    x : torch.Tensor
        A 2D tensor of shape (n_samples, 2) representing input points.

    Returns:
    --------
    torch.Tensor
        The target density at the input points.
    """
    return target_distribution(x)

def mcmc_with_nf(model, n_samples, initial_sample=None):
    """
    Perform Metropolis-Hastings MCMC using a Normalizing Flow model for proposals.

    Parameters:
    -----------
    model : NormalizingFlow
        The trained Normalizing Flow model.
    n_samples : int
        Number of MCMC samples to generate.
    initial_sample : list or np.ndarray, optional
        Initial sample to start the MCMC. If not provided, sampled from the base distribution.

    Returns:
    --------
    np.ndarray
        An array of shape (n_samples, dim) containing the MCMC samples.
    """
    samples = []
    accepted = 0

    # Initialize the first sample
    if initial_sample is None:
        current_sample = base_distribution(1)  # Single sample from base distribution
    else:
        current_sample = torch.tensor(initial_sample, dtype=torch.float32, device=device).unsqueeze(0)

    current_density = target_density(current_sample).item()

    for _ in range(n_samples):
        # Propose a new sample using the NF model
        z0 = base_distribution(1)  # Sample from base distribution
        proposal, _ = model(z0)    # Transform using NF model
        
        # Compute target density for the proposal
        proposal_density = target_density(proposal).item()

        # Apply Metropolis-Hastings acceptance criterion
        alpha = min(1, proposal_density / current_density)
        if torch.rand(1).item() < alpha:
            current_sample = proposal
            current_density = proposal_density
            accepted += 1

        samples.append(current_sample.detach().cpu().numpy())

    acceptance_rate = accepted / n_samples
    print(f"MCMC Acceptance Rate: {acceptance_rate:.4f}")

    return np.vstack(samples)

def mcmc_example():
    """
    Perform MCMC using a trained Normalizing Flow model and visualize the results.
    """
    checkpoint_path = 'best_model.pth'  # Path to the saved model
    dim = 2
    n_flows = 40

    # Load the trained model
    trained_model = load_trained_model(checkpoint_path, dim, n_flows)

    # Run MCMC
    n_mcmc_samples = 10000
    mcmc_samples = mcmc_with_nf(trained_model, n_mcmc_samples)

    # Generate grid for target distribution
    x = np.linspace(-50, 50, 100)
    y = np.linspace(-50, 50, 100)
    X, Y = np.meshgrid(x, y)
    grid = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32, device=device)

    Z = target_distribution(grid).reshape(100, 100).cpu().numpy()

    # Plot MCMC samples and target density
    plt.figure(figsize=(12, 6))
    plt.contourf(X, Y, Z, levels=20, cmap="viridis")
    plt.scatter(mcmc_samples[:, 0], mcmc_samples[:, 1], alpha=0.3, color='red', label='MCMC Samples')
    plt.title("MCMC Samples from Normalizing Flow")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("mcmc_samples_nf.pdf")
    plt.show()

# Run the MCMC example
if __name__ == "__main__":
    mcmc_example()
