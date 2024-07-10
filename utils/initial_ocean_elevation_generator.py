import numpy as np

class InitialOceanElevationGenerator:
    def generate_initial_ocean_elevation(self, sim_params, bottom_surface):
        """
        Generates initial ocean elevation data based on the provided parameters.

        Args:
            sim_params (SimulationParameters): The simulation parameters object containing the necessary parameters.
            bottom_surface (np.ndarray): The 2D array containing the bottom surface data.

        Returns:
            np.ndarray: A 2D array of shape (height, width, 4) containing the initial ocean elevation data.
        """
        width = sim_params.WIDTH
        height = sim_params.HEIGHT
        centerX = width // 2
        centerY = height // 2
        amplitude = sim_params.amplitude

        # Create a 2D array to store ocean elevation data
        data = np.zeros((height, width, 4), dtype=np.float32)

        # Define sigma, which affects the spread of the initial elevation
        sigma = 24.0  # make smaller for shorter IC

        # Create grid of x and y coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Calculate the differences between the current coordinates and the center
        dx = x - centerX
        dy = y - centerY

        # Get the bottom surface elevation at the current point and neighbors
        bottom = bottom_surface[:, :, 2]
        bottom_right = np.roll(bottom, -1, axis=1)
        bottom_left = np.roll(bottom, 1, axis=1)
        bottom_up = np.roll(bottom, -1, axis=0)
        bottom_down = np.roll(bottom, 1, axis=0)

        # Calculate the initial elevation as a Gaussian function centered on the middle of the 2D space
        elev_correction = 0.0 * (np.maximum(bottom_right, 0.0) + np.maximum(bottom_left, 0.0) + np.maximum(bottom_up, 0.0) + np.maximum(bottom_down, 0.0))
        elevation = elev_correction + 0.0 * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

        # Apply the elevation correction condition
        elevation = np.where((elevation < bottom) | (bottom > 0), bottom, elevation)

        # Store the results in the data array
        data[:, :, 0] = elevation
        data[:, :, 1] = 0.0  # velocity in x direction
        data[:, :, 2] = 0.0  # velocity in y direction
        data[:, :, 3] = 0.0  # scalar transport

        return data