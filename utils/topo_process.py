import os
import numpy as np
from scipy.interpolate import griddata
import warp as wp
from kernels.fill_bottom import Fill_Bottom
from utils.simulation_parameters import SimulationParameters

class TopoProcess:
    def __init__(self, filename=None, datatype=None, path=None, sim_params=None, array_manager=None):
        """
        Initialize the TopoProcess object.
        """
        self.filename = filename
        self.datatype = datatype
        self.path = path
        self.sim_params = sim_params or SimulationParameters()
        self.array_manager = array_manager  # Reference to WarpArrayManager

        # Load and process bathy data
        self.load_and_process_data()

        # Handle boundary conditions
        self.handle_boundary()

        # Create Bottom field and launch the Warp kernel
        self.create_bottom_field()

    def z(self):
        """
        Load the bathymetry data based on datatype.
        """
        if self.datatype == 'xyz':
            if self.path is not None:
                filepath = os.path.join(self.path, self.filename)
            else:
                filepath = self.filename
            data = np.loadtxt(filepath)
            return data
        elif self.datatype == 'celeris':
            filepath = os.path.join(self.path, 'bathy.txt')
            bathy = np.loadtxt(filepath)
            return bathy * -1.0
        else:
            raise ValueError('Unsupported data format. Supported formats are "xyz" and "celeris".')

    def topofield(self):
        """
        Generate topo fields based on datatype.
        """
        WIDTH = self.sim_params.WIDTH
        HEIGHT = self.sim_params.HEIGHT
        dx = self.sim_params.dx
        dy = self.sim_params.dy

        if self.datatype == 'celeris':
            x_out, y_out = np.meshgrid(np.arange(0.0, WIDTH * dx, dx),
                                       np.arange(0.0, HEIGHT * dy, dy))
            foo = self.z()
            dem = foo.T  # Transpose to match grid
            return x_out, y_out, dem
        elif self.datatype == 'xyz':
            dum = self.z()
            x1 = self.sim_params.x1
            y1 = self.sim_params.y1
            x2 = self.sim_params.x2
            y2 = self.sim_params.y2
            x_out, y_out = np.meshgrid(np.arange(x1, x2, dx),
                                       np.arange(y1, y2, dy))
            # Interpolate using griddata
            dem = griddata(dum[:, :2], dum[:, 2], (x_out, y_out), method='nearest')
            return x_out.T, y_out.T, dem.T  # Transpose to match grid
        else:
            raise ValueError('Unsupported data format. Supported formats are "xyz" and "celeris".')

    def load_and_process_data(self):
        """
        Load and process the bathymetry data.
        """
        self.x_out, self.y_out, self.dem = self.topofield()

    def handle_boundary(self):
        """
        Handle the boundaries of the depth data by correcting edges.
        """
        bathy2d = self.dem.copy()

        # Top boundary
        if bathy2d.shape[0] >= 4:
            bathy2d[:2, :] = bathy2d[3:1:-1, :]
        elif bathy2d.shape[0] >= 2:
            bathy2d[:2, :] = bathy2d[:2, :]

        # Bottom boundary
        if bathy2d.shape[0] >= 4:
            bathy2d[-2:, :] = bathy2d[-3:-1, :][::-1, :]
        elif bathy2d.shape[0] >= 2:
            bathy2d[-2:, :] = bathy2d[-2:, :]

        # Left boundary
        if bathy2d.shape[1] >= 4:
            bathy2d[:, :2] = bathy2d[:, 3:1:-1]
        elif bathy2d.shape[1] >= 2:
            bathy2d[:, :2] = bathy2d[:, :2]

        # Right boundary
        if bathy2d.shape[1] >= 4:
            bathy2d[:, -2:] = bathy2d[:, -3:-1][:, ::-1]
        elif bathy2d.shape[1] >= 2:
            bathy2d[:, -2:] = bathy2d[:, -2:]

        self.dem = bathy2d

    def create_bottom_field(self):
        """
        Populate the existing Bottom Warp array
        """
        WIDTH = self.sim_params.WIDTH
        HEIGHT = self.sim_params.HEIGHT
        device = self.sim_params.device

        # Create a host-side numpy array to initialize z and w components
        bottom_np = np.zeros((WIDTH, HEIGHT, 4), dtype=np.float32)

        # Set Bed elevation (z component) to -1 * dem
        bottom_np[:, :, 2] = -1.0 * self.dem

        # Initialize near_dry flag to 99.0
        bottom_np[:, :, 3] = 99.0

        # Initialize the Bottom Warp array from NumPy
        self.array_manager.Bottom = wp.from_numpy(bottom_np, dtype=wp.vec4f, device=device)

        # Launch the Warp kernel to compute BN, BE, and update the near_dry flag
        wp.launch(
            kernel=Fill_Bottom,
            dim=(WIDTH, HEIGHT),
            inputs=[
                self.array_manager.Bottom,
                wp.int32(WIDTH),
                wp.int32(HEIGHT),
                wp.int32(3)
            ],
            device=device
        )

        # Ensure kernel completion
        wp.synchronize_device(device)

    def get_bottom(self):
        """
        Retrieve the Bottom field as a NumPy array.
        """
        return self.array_manager.Bottom.numpy()

    def get_dem(self):
        """
        Retrieve the depth grid.

        """
        return self.dem

    def get_grid(self):
        """
        Retrieve the grid coordinates.

        """
        return self.x_out, self.y_out

    def get_maxdepth(self):
        """
        Get the maximum depth.

        """
        return np.max(self.dem) if self.sim_params.base_depth is None else self.sim_params.base_depth

    def get_maxtopo(self):
        """
        Get the minimum topo value.

        """
        return np.min(self.dem)