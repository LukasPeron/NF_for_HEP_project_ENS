import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Update matplotlib's font size for better readability
matplotlib.rcParams.update({'font.size': 20})

def target_distribution_2d(x, y, mu1, sigma1, mu2, sigma2):
    """
    Computes the value of a target distribution consisting of two Gaussian wells.

    Parameters:
    -----------
    x : float or np.ndarray
        x-coordinate(s) where the target distribution is evaluated.
    y : float or np.ndarray
        y-coordinate(s) where the target distribution is evaluated.
    mu1 : list or tuple
        Mean [x, y] of the first Gaussian well.
    sigma1 : float
        Standard deviation of the first Gaussian well.
    mu2 : list or tuple
        Mean [x, y] of the second Gaussian well.
    sigma2 : float
        Standard deviation of the second Gaussian well.

    Returns:
    --------
    np.ndarray
        The combined value of the two Gaussian wells at the specified (x, y) coordinates.
    """
    well1 = np.exp(-((x - mu1[0])**2 + (y - mu1[1])**2) / (2 * sigma1**2))
    well2 = np.exp(-((x - mu2[0])**2 + (y - mu2[1])**2) / (2 * sigma2**2))
    return well1 + well2

# Parameters for the Gaussian wells
mu1 = [30, 30]
sigma1 = 5
mu2 = [-30, -30]
sigma2 = 5

# Generate a grid for evaluating the target distribution
x = np.linspace(-50, 50, 100)
y = np.linspace(-50, 50, 100)
X, Y = np.meshgrid(x, y)
Z = target_distribution_2d(X, Y, mu1, sigma1, mu2, sigma2)

# Number of points to draw
n_points = 10000

# Draw points uniformly over the 2D domain
uniform_points = np.random.uniform(-50, 50, (n_points, 2))

# Calculate the target distribution value for each point
probabilities = target_distribution_2d(uniform_points[:, 0], uniform_points[:, 1], mu1, sigma1, mu2, sigma2)

# Normalize probabilities to use as alpha values for transparency
alpha_values = probabilities / np.max(probabilities)

# Plot the target distribution and reweighted points
fig, ax = plt.subplots(figsize=(12, 6))
ax.contourf(X, Y, Z, levels=50, cmap='viridis')
ax.scatter(uniform_points[:, 0], uniform_points[:, 1], c='red', alpha=alpha_values, label='Reweighted Points')
ax.set_title('Shoot and Reweight Method for 2D Distribution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc="lower right")
plt.savefig('shoot_and_reweight_2d.png')
plt.savefig('shoot_and_reweight_2d.pdf')
plt.show()
