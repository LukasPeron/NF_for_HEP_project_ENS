import torch
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

def kde_density(samples, x_grid, y_grid):
    """
    Estimate the density of MCMC samples using Kernel Density Estimation (KDE).

    Parameters:
    -----------
    samples : np.ndarray
        Array of shape (n_samples, dim) containing MCMC samples.
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

def mcmc_example():
    """
    Perform MCMC using a trained Normalizing Flow model and visualize the results.
    """
    # Model and sampling parameters
    checkpoint_path = 'best_model.pth'
    dim = 2
    n_flows = 40
    n_mcmc_samples = 10000

    # Load the trained model
    trained_model = load_trained_model(checkpoint_path, dim, n_flows)

    # Run MCMC to generate samples
    mcmc_samples = mcmc_with_nf(trained_model, n_mcmc_samples)

    # Generate grid for KDE and density estimation
    grid_size = 100
    x = np.linspace(-50, 50, grid_size)
    y = np.linspace(-50, 50, grid_size)
    X, Y = np.meshgrid(x, y)

    # Perform KDE for density estimation
    mcmc_density = kde_density(mcmc_samples, X, Y)

    # Plot the density of MCMC samples
    plt.figure(figsize=(12, 6))
    contour = plt.contourf(X, Y, mcmc_density, levels=50, cmap="viridis")
    plt.colorbar(contour, label='Density')
    plt.title("Density of MCMC Samples")
    plt.xlabel("x")
    plt.ylabel("y")

    # Save and display the plot
    plt.savefig("mcmc_density.pdf")
    plt.show()

# Run the MCMC example
if __name__ == "__main__":
    mcmc_example()
