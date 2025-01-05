import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Update matplotlib's font size for better readability
matplotlib.rcParams.update({'font.size': 20})

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

    return (dist1 + dist2) / (2 * (2 * np.pi * sigma**2))

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
        """
        Applies the planar flow transformation.

        Parameters:
        -----------
        z : torch.Tensor
            Input tensor of shape (n_samples, dim).

        Returns:
        --------
        torch.Tensor
            Transformed tensor of the same shape as input.
        """
        linear = torch.matmul(z, self.w) + self.b
        activation = torch.tanh(linear)
        z_out = z + self.u * activation.unsqueeze(1)
        return z_out

    def log_det_jacobian(self, z):
        """
        Computes the log determinant of the Jacobian for the planar flow.

        Parameters:
        -----------
        z : torch.Tensor
            Input tensor of shape (n_samples, dim).

        Returns:
        --------
        torch.Tensor
            Log determinant of the Jacobian for each input sample.
        """
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
        """
        Applies the sequence of planar flows to the input.

        Parameters:
        -----------
        z : torch.Tensor
            Input tensor of shape (n_samples, dim).

        Returns:
        --------
        torch.Tensor
            Transformed tensor after all flows.
        torch.Tensor
            Sum of log determinants of Jacobians for all flows.
        """
        log_jacobian = 0
        for flow in self.flows:
            z = flow(z)
            log_jacobian += flow.log_det_jacobian(z)
        return z, log_jacobian

def train():
    """
    Trains the normalizing flow model to approximate the target distribution.

    Saves the best model during training and generates visualizations of the 
    flow's progress at regular intervals.
    """
    model = NormalizingFlow(dim=2, n_flows=40).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    best_loss = float('inf')

    for epoch in range(1, 20001):
        z0 = base_distribution(5000)
        z_k, log_jacobian = model(z0)
        log_p_target = torch.log(target_distribution(z_k))
        log_p_base = -0.5 * torch.sum(z0**2, dim=1)

        loss = -torch.mean(log_p_target + log_jacobian - log_p_base)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
            }, 'best_model.pth')

        if epoch % 1000 == 0 or epoch == 1:
            visualize(model, epoch, z0, z_k)

    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    visualize(model, 'best', z0, z_k, save=True)

    plt.figure(figsize=(12, 6))
    plt.loglog(range(1, 20001), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid()
    plt.savefig("loss_curve_loglog.pdf")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 20001), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid()
    plt.savefig("loss_curve_linear.pdf")
    plt.show()

def visualize(model, epoch, z0, z_k, save=False):
    """
    Visualizes the target distribution and the samples generated by the normalizing flow.

    Parameters:
    -----------
    model : NormalizingFlow
        The trained normalizing flow model.
    epoch : int or str
        The epoch number or 'best' for the best model visualization.
    z0 : torch.Tensor
        Samples from the base distribution.
    z_k : torch.Tensor
        Transformed samples from the normalizing flow.
    save : bool, optional
        Whether to save the visualization to a file.
    """
    x = np.linspace(-50, 50, 100)
    y = np.linspace(-50, 50, 100)
    X, Y = np.meshgrid(x, y)
    grid = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32, device=device)

    Z = target_distribution(grid).reshape(100, 100).cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.contourf(X, Y, Z, levels=20, cmap="viridis")
    z_k_np = z_k.detach().cpu().numpy()
    plt.scatter(z_k_np[:, 0], z_k_np[:, 1], color="red", alpha=0.6, label="NF Samples")
    plt.title(f"Target Distribution and NF Samples (Epoch {epoch})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    if save:
        plt.savefig(f"best_model_plot.pdf")
        plt.savefig(f"best_model_plot.png")
    if epoch == 1:
        plt.savefig(f"base_distribution.pdf")
        plt.savefig(f"base_distribution.png")
    plt.show()

# Run the training
train()
