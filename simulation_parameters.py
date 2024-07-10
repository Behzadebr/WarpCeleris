import numpy as np

class SimulationParameters:
    def __init__(self):
        # Parameters from JSON
        self.WIDTH: int = 800  # Width of the computational domain.
        self.HEIGHT: int = 600  # Height of the computational domain.
        self.dx: float = 1.0  # Cell size in the x direction.
        self.dy: float = 1.0  # Cell size in the y direction.
        self.Courant_num: float = 0.15  # Target Courant number. ~0.25 for P-C, ~0.05 for explicit methods.
        self.NLSW_or_Bous: int = 0  # Choose 0 for Non-linear Shallow Water (NLSW) or 1 for Boussinesq.
        self.base_depth: float = 1.0  # characteristic_depth (m), used to estimate time step, use depth in area of wave generation, or expected largest depth in domain.
        self.g: float = 9.80665  # Gravitational constant.
        self.Theta: float = 1.3  # Midmod limiter parameter. 1.0 most dissipative(upwind) to 2.0 least dissipative(centered).
        self.friction: float = 0.001  # Dimensionless friction coefficient, or Mannings 'n', depending on isManning choice.
        self.isManning: int = 1  # A boolean friction model value, if==1 'friction' is a Mannnigs n, otherwise it is a dimensionless friction factor (Moody).
        self.dissipation_threshold: float = 1.0  # For visualization purposes, represents a characterestic slope at breaking. 0.577=30 degrees, will need to be lower for coarse grid simulations that can not resolve this slope; 0.364=20 deg; 0.2679=15 deg.
        self.whiteWaterDecayRate: float = 1.0  # Decay rate of white water (foam).
        self.timeScheme: int = 2  # Time integration choices: 0: Euler, 1: 3rd-order A-B predictor, 2: A-B 4th-order predictor+corrector.
        self.seaLevel: float = 0.0  # Water level shift from given datum.
        self.Bcoef: float = 1.0 / 15.0  # Dispersion parameter, 1/15 is optimum value for this set of equations.
        self.tridiag_solve: int = 1  # Method to solve the tridiagonal Boussinesq system: 0: Thomas (extremely slow, only for small domains (nx,ny <500) due to thread memory req), 1: Gauss-Sid (very slow), or 2: parallel cyclic reduction (best).
        self.west_boundary_type: int = 0  # Type of boundary condition at the west boundary. 0: solid wall, 1 :sponge layer, 2: waves loaded from file, created by spectrum_2D
        self.east_boundary_type: int = 0  # Type of boundary condition at the east boundary. 0: solid wall, 1 :sponge layer, 2: waves loaded from file, created by spectrum_2D
        self.south_boundary_type: int = 0  # Type of boundary condition at the south boundary. 0: solid wall, 1 :sponge layer, 2: waves loaded from file, created by spectrum_2D
        self.north_boundary_type: int = 0  # Type of boundary condition at the north boundary. 0: solid wall, 1 :sponge layer, 2: waves loaded from file, created by spectrum_2D
        self.start_write_time: float = 60.0  # Start time for writing output.
        self.end_write_time: float = 120.0  # End time for writing output.
        self.write_dt: float = 1.0  # Time interval for writing output.
        self.n_disp_interval: float = 100.0  # Display interval.
        self.significant_wave_height: float = 0.0  # Significant wave height.

        # Other parameters that are not provided in the JSON
        self.amplitude: float = 0.0
        self.period: float = 10.0
        self.direction: float = 0.0
        self.rand_phase: float = 0.0
        self.dt: float = 0.0
        self.TWO_THETA: float = 0.0
        self.half_g: float = 0.0
        self.Bcoef_g: float = 0.0
        self.g_over_dx: float = 0.0
        self.g_over_dy: float = 0.0
        self.one_over_dx: float = 0.0
        self.one_over_dy: float = 0.0
        self.one_over_d2x: float = 0.0
        self.one_over_d3x: float = 0.0
        self.one_over_d2y: float = 0.0
        self.one_over_d3y: float = 0.0
        self.one_over_dxdy: float = 0.0
        self.epsilon: float = 0.0
        self.boundary_epsilon: float = 0.0
        self.boundary_nx: int = 0
        self.boundary_ny: int = 0
        self.reflect_x: int = 0
        self.reflect_y: int = 0
        self.BoundaryWidth: float = 0.0
        self.boundary_g: float = 0.0
        self.pred_or_corrector: int = 1
        self.numberOfWaves: int = 0
        self.Px: int = 0
        self.Py: int = 0
        self.n_writes: int = 0
        self.n_write_interval: int = 0
        self.waveVectors: np.ndarray = np.array([])
        self.waves: np.ndarray = np.array([])

        # For vessel motion
        self.numberOfShips: int = 0
        self.activeShipIdx: int = 0  # Index of ship being controlled [0 - (numberOfShips-1)]
        self.shipBoundaryPolygon: np.ndarray = np.array([])
        self.shipProperties: list = []

