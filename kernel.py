import numpy as np

def get_kernel(kernel, **kwargs):
    """Get the kernel function. Supported kernels are 'linear' and 'gaussian'.

    Args:
        kernel (str): The kernel name.

    Raises:
        ValueError: If the kernel is not recognized.

    Returns:
        function: The kernel function.
    """
    if kernel == 'linear':
        return linear_kernel(**kwargs)
    elif kernel == 'gaussian':
        return gaussian_kernel(**kwargs)
    else:
        raise ValueError('Kernel not recognized.')

def linear_kernel(**kwargs):
    def f(x1, x2):
        return np.inner(x1, x2)
    return f

def gaussian_kernel(gamma, **kwargs):
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f
