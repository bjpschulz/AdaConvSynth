import torch
import torch.nn.functional as F
import numpy as np


def gabor_filters(n_filters=64, kernel_size=(7, 7)):
    """
    Generate a set of Gabor filters with the specified kernel size and number of filters using PyTorch.

    Parameters:
        kernel_size (tuple): Size of the Gabor kernel (height, width).
        n_filters (int): Number of Gabor filters to generate.

    Returns:
        list: A list of Gabor filter kernels as PyTorch tensors.
    """
    filters = []

    # Parameters for Gabor filter
    sigma = 2.0  # Standard deviation of the Gaussian envelope
    lambd = torch.pi / 4  # Wavelength of the sinusoidal factor
    gamma = 0.5  # Spatial aspect ratio
    psi = 0  # Phase offset

    y, x = torch.meshgrid(
        torch.arange(kernel_size[0]) - kernel_size[0] // 2,
        torch.arange(kernel_size[1]) - kernel_size[1] // 2,
        indexing="ij",
    )

    for i in range(n_filters):
        theta = torch.tensor((i / n_filters) * torch.pi)  # Orientations evenly spaced
        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

        gaussian = torch.exp(-0.5 * ((x_theta**2 + gamma**2 * y_theta**2) / sigma**2))
        sinusoid = torch.cos(2 * torch.pi * x_theta / lambd + psi)
        gabor_kernel = gaussian * sinusoid

        filters.append(gabor_kernel)

    filters = torch.stack(filters, dim=0).unsqueeze(1)

    filters /= torch.linalg.vector_norm(filters, dim=(-3, -2, -1), keepdim=True)

    return filters


# Example usage:
# filters = gabor_filters(kernel_size=(7, 7), n_filters=64)


def dct_filters(num_filters: int, kernel_size: int):
    """
    Generates a set of DCT-III filters in PyTorch.

    Args:
        num_filters (int): Number of filters to generate.
        kernel_size (int): Size of the kernel (assumed square).

    Returns:
        torch.Tensor: A tensor of shape (num_filters, 1, kernel_size, kernel_size).
    """
    filters = []
    for i in range(num_filters):
        # Create a 2D grid
        x = np.linspace(0, np.pi, kernel_size, endpoint=False)
        y = np.linspace(0, np.pi, kernel_size, endpoint=False)
        X, Y = np.meshgrid(x, y)

        # Generate a DCT-III basis function
        filter_matrix = np.cos((2 * i + 1) * X / (2 * kernel_size)) * np.cos(
            (2 * i + 1) * Y / (2 * kernel_size)
        )

        # Normalize to have zero mean and unit variance
        filter_matrix -= filter_matrix.mean()
        filter_matrix /= filter_matrix.std() + 1e-8

        filters.append(filter_matrix)

    # Convert to PyTorch tensor and reshape for convolution
    filters = torch.tensor(filters, dtype=torch.float32).unsqueeze(1)
    filters /= torch.linalg.vector_norm(filters, dim=(-3, -2, -1), keepdim=True)

    return filters


# Example Usage:
# num_filters = 32
# kernel_size = 7
# dct_filters = dct_filters(num_filters, kernel_size)
# print(dct_filters.shape)  # Should be (8, 1, 4, 4)


def mixed_finite_difference_kernels():
    """
    Manually constructs 2D finite difference filters to match the uploaded image.

    Returns:
        torch.Tensor: A tensor of shape (num_filters, 1, kernel_size, kernel_size),
                      containing predefined derivative filters.
    """
    filters = []

    # Define each filter explicitly based on the provided image structure

    # First-order derivative in x (∂/∂x)
    Dx = torch.tensor([[-1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float32)

    # First-order derivative in y (∂/∂y)
    Dy = torch.tensor([[-1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32)

    # Second-order derivative in x (∂²/∂x²)
    Dxx = torch.tensor([[1, -1, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.float32)

    # Second-order derivative in y (∂²/∂y²)
    Dyy = torch.tensor([[1, 0, 0], [-1, 0, 0], [1, 0, 0]], dtype=torch.float32)

    # Mixed derivative (∂²/∂x∂y)
    Dxy = torch.tensor([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32)

    # Append all filters into a list
    filters.extend([Dx, Dy, Dxx, Dyy, Dxy])

    # Convert to PyTorch tensor and reshape to (num_filters, 1, kernel_size, kernel_size)
    filters = torch.stack(filters).unsqueeze(1)

    return filters


# Example Usage:
finite_diff_kernels = mixed_finite_difference_kernels()
