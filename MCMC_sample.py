import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def target_distribution_2d(x, y, mu1, sigma1, mu2, sigma2, distance):
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
    distance : float
        Distance separating the two Gaussian wells (affects their relative positioning).

    Returns:
    --------
    np.ndarray
        The combined value of the two Gaussian wells at the specified (x, y) coordinates.
    """
    well1 = np.exp(-((x - mu1[0])**2 + (y - mu1[1])**2) / (2 * sigma1**2))
    well2 = np.exp(-((x - mu2[0])**2 + (y - mu2[1])**2) / (2 * sigma2**2))
    return well1 + well2

def metropolis_hasting_2d(mu1, sigma1, mu2, sigma2, distance, proposal_std, n_samples):
    """
    Implements the Metropolis-Hastings algorithm for sampling a 2D target distribution.

    This function also animates the sampling process and saves the animation as an MP4 file.

    Parameters:
    -----------
    mu1 : list or tuple
        Mean [x, y] of the first Gaussian well in the target distribution.
    sigma1 : float
        Standard deviation of the first Gaussian well.
    mu2 : list or tuple
        Mean [x, y] of the second Gaussian well in the target distribution.
    sigma2 : float
        Standard deviation of the second Gaussian well.
    distance : float
        Distance separating the two Gaussian wells (affects their relative positioning).
    proposal_std : float
        Standard deviation of the Gaussian proposal distribution for generating new samples.
    n_samples : int
        Number of samples to generate in the MCMC chain.

    Returns:
    --------
    np.ndarray
        A 2D array of shape (n_samples, 2) containing the sampled points.

    Notes:
    ------
    The animation is saved to a file named 'metropolis_hasting_2d_animation_large_gap.mp4'.
    """
    chain = np.zeros((n_samples, 2))
    current = np.random.uniform(-50, 50, 2)

    # Create plot and initialize animation
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.linspace(-50, 50, 100)
    y = np.linspace(-50, 50, 100)
    X, Y = np.meshgrid(x, y)
    Z = target_distribution_2d(X, Y, mu1, sigma1, mu2, sigma2, distance)
    ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    samples, = ax.plot([], [], 'ro', alpha=0.6, label='MCMC Samples')
    ax.set_title('Metropolis-Hastings MCMC Sampling for 2D Distribution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    # Update function for animation
    def update(i):
        nonlocal current
        candidate = current + np.random.normal(0, proposal_std, 2)
        acceptance_ratio = target_distribution_2d(
            candidate[0], candidate[1], mu1, sigma1, mu2, sigma2, distance
        ) / target_distribution_2d(
            current[0], current[1], mu1, sigma1, mu2, sigma2, distance
        )
        
        if np.random.uniform(0, 1) < acceptance_ratio:
            current = candidate
        
        chain[i] = current
        samples.set_data(chain[:i+1, 0], chain[:i+1, 1])
        plt.draw()
        return samples,

    ani = animation.FuncAnimation(fig, update, frames=n_samples, blit=True, repeat=False)
    ani.save('metropolis_hasting_2d_animation_large_gap.mp4', writer='ffmpeg')

    return chain

# Parameters
mu1 = [30, 30]
sigma1 = 5
mu2 = [-30, -30]
sigma2 = 5
distance = 50
proposal_std = 5.0
n_samples = 5000

# Run Metropolis-Hastings algorithm for 2D distribution
chain_2d = metropolis_hasting_2d(mu1, sigma1, mu2, sigma2, distance, proposal_std, n_samples)
