import numpy as np

class TridiagCoefGenerator:
    def __init__(self):
        pass

    def generate_tridiag_coef_x(self, sim_params, bottom_surface):
        """
        Generate tridiagonal coefficients in the x direction.

        Args:
            sim_params (SimulationParameters): The simulation parameters object containing the necessary parameters.
            bottom_surface (np.ndarray): The 2D array containing the bottom surface data.

        Returns:
            np.ndarray: A 2D array of shape (height, width, 4) containing the tridiagonal coefficients in the x direction.
        """
        width = sim_params.WIDTH
        height = sim_params.HEIGHT
        Bcoef = sim_params.Bcoef
        dx = sim_params.dx

        data = np.zeros((height, width, 4), dtype=np.float32)

        # Extract depth from bottom_surface
        depth_here = -bottom_surface[:, :, 2]
        depth_plus = np.roll(depth_here, -1, axis=1)
        depth_minus = np.roll(depth_here, 1, axis=1)

        d_dx = (depth_plus - depth_minus) / (2.0 * dx)
        a = depth_here * d_dx / (6.0 * dx) - (Bcoef + 1.0 / 3.0) * depth_here**2 / (dx * dx)
        b = 1.0 + 2.0 * (Bcoef + 1.0 / 3.0) * depth_here**2 / (dx * dx)
        c = -depth_here * d_dx / (6.0 * dx) - (Bcoef + 1.0 / 3.0) * depth_here**2 / (dx * dx)

        # Boundary conditions
        b[:, :3] = 1.0
        b[:, -3:] = 1.0
        a[:, :3] = 0.0
        a[:, -3:] = 0.0
        c[:, :3] = 0.0
        c[:, -3:] = 0.0

        # Near dry conditions
        mask = bottom_surface[:, :, 3] < 0
        a[mask] = 0.0
        b[mask] = 1.0
        c[mask] = 0.0

        data[:, :, 0] = a
        data[:, :, 1] = b
        data[:, :, 2] = c

        return data

    def generate_tridiag_coef_y(self, sim_params, bottom_surface):
        """
        Generate tridiagonal coefficients in the y direction.

        Args:
            sim_params (SimulationParameters): The simulation parameters object containing the necessary parameters.
            bottom_surface (np.ndarray): The 2D array containing the bottom surface data.

        Returns:
            np.ndarray: A 2D array of shape (height, width, 4) containing the tridiagonal coefficients in the y direction.
        """
        width = sim_params.WIDTH
        height = sim_params.HEIGHT
        Bcoef = sim_params.Bcoef
        dy = sim_params.dy

        data = np.zeros((height, width, 4), dtype=np.float32)

        # Extract depth from bottom_surface
        depth_here = -bottom_surface[:, :, 2]
        depth_plus = np.roll(depth_here, -1, axis=0)
        depth_minus = np.roll(depth_here, 1, axis=0)

        d_dy = (depth_plus - depth_minus) / (2.0 * dy)
        a = depth_here * d_dy / (6.0 * dy) - (Bcoef + 1.0 / 3.0) * depth_here**2 / (dy * dy)
        b = 1.0 + 2.0 * (Bcoef + 1.0 / 3.0) * depth_here**2 / (dy * dy)
        c = -depth_here * d_dy / (6.0 * dy) - (Bcoef + 1.0 / 3.0) * depth_here**2 / (dy * dy)

        # Boundary conditions
        b[:3, :] = 1.0
        b[-3:, :] = 1.0
        a[:3, :] = 0.0
        a[-3:, :] = 0.0
        c[:3, :] = 0.0
        c[-3:, :] = 0.0

        # Near dry conditions
        mask = bottom_surface[:, :, 3] < 0
        a[mask] = 0.0
        b[mask] = 1.0
        c[mask] = 0.0

        data[:, :, 0] = a
        data[:, :, 1] = b
        data[:, :, 2] = c

        return data
