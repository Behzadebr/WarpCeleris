import numpy as np

class DepthSurfaceGenerator:
    def __init__(self):
        pass

    def load_depth(self, file_path, width, height):
        """
        Load depth data from a file and return it as a 2D array.

        Args:
            file_path (str): The path to the depth file.
            width (int): The width of the depth data grid.
            height (int): The height of the depth data grid.

        Returns:
            np.ndarray: A 2D array containing the depth data.
        """
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                depth_data = np.zeros((width, height), dtype=np.float32)
                for i, line in enumerate(lines[:height]):
                    depth_values = line.split()
                    if len(depth_values) != width:
                        raise ValueError(f"Depth file at {file_path} is not in the correct format.")
                    depth_data[:, i] = np.array(depth_values, dtype=np.float32)
            return depth_data
        except IOError as e:
            print(f"Could not find depth file at {file_path}: {e}")
            return None
        except ValueError as e:
            print(e)
            return None

    def handle_boundary(self, bathy2d):
        """
        Handle the boundaries of the depth data by correcting edges.

        Args:
            bathy2d (np.ndarray): The 2D array containing the depth data.
        """
        width, height = bathy2d.shape

        # Correcting edges by mirroring the inner cells
        bathy2d[:2, :] = bathy2d[4:2:-1, :]
        bathy2d[-2:, :] = bathy2d[-4:-2][::-1, :]
        bathy2d[:, :2] = bathy2d[:, 4:2:-1]
        bathy2d[:, -2:] = bathy2d[:, -4:-2][:, ::-1]

    def generate_depth_surface(self, sim_params, file_path):
        """
        Generate a depth surface based on depth data loaded from a file.

        Args:
            sim_params (SimulationParameters): The simulation parameters object containing the necessary parameters.
            file_path (str): The path to the depth file.

        Returns:
            np.ndarray: A 2D array of shape (height, width, 4) containing the depth surface data.
        """
        width = sim_params.WIDTH
        height = sim_params.HEIGHT
        bathy2d = self.load_depth(file_path, width, height)

        if bathy2d is None:
            return None

        self.handle_boundary(bathy2d)

        data = np.zeros((height, width, 4), dtype=np.float32)

        # Populate the data array with depth information
        # Vectorized calculation of BN, BE, BS, BW
        bn = 0.5 * bathy2d + 0.5 * np.roll(bathy2d, -1, axis=1)
        be = 0.5 * bathy2d + 0.5 * np.roll(bathy2d, -1, axis=0)
        bs = 0.5 * bathy2d + 0.5 * np.roll(bathy2d, 1, axis=1)
        bw = 0.5 * bathy2d + 0.5 * np.roll(bathy2d, 1, axis=0)

        # Make sure edges are handled correctly
        bn[:, -1] = bathy2d[:, -1]
        be[-1, :] = bathy2d[-1, :]
        bs[:, 0] = bathy2d[:, 0]
        bw[0, :] = bathy2d[0, :]

        data[:, :, 0] = bn
        data[:, :, 1] = be
        data[:, :, 2] = bathy2d
        data[:, :, 3] = 99

        # Boolean near-dry check
        length_check = 3  # check within three points

        mask = (bathy2d >= 0)
        for i in range(1, length_check + 1):
            mask[:-i, :] |= (bathy2d[i:, :] >= 0)
            mask[i:, :] |= (bathy2d[:-i, :] >= 0)
            mask[:, :-i] |= (bathy2d[:, i:] >= 0)
            mask[:, i:] |= (bathy2d[:, :-i] >= 0)
        data[mask, 3] = -99

        return data