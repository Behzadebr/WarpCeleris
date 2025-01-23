import numpy as np

class SimulationParameters:
    def __init__(self):

        self.WIDTH: int = 1000  # Width of the computational domain.
        self.HEIGHT: int = 1000  # Height of the computational domain.
        self.x1: float = 0.0  # Lower bound in the x-direction.
        self.x2: float = 1000.0  # Upper bound in the x-direction.
        self.y1: float = 0.0  # Lower bound in the y-direction.
        self.y2: float = 1000.0  # Upper bound in the y-direction.
        self.dx: float = 1.0  # Cell size in the x direction.
        self.dy: float = 1.0  # Cell size in the y direction.

        self.Courant_num: float = 0.2  # Target Courant number (~0.25 for P-C, ~0.05 for explicit methods).
        self.NLSW_or_Bous: int = 0  # Choose 0 for Non-linear Shallow Water (NLSW) or 1 for Boussinesq.
        self.base_depth: float = 1.0  # Characteristic depth (m), used to estimate time step.
        self.g: float = 9.80665  # Gravitational constant (m/s²).
        self.Theta: float = 2.0  # Midmod limiter parameter (1.0 most dissipative upwind to 2.0 least dissipative centered).
        self.timeScheme: int = 2  # Time integration choice.
        self.pred_or_corrector: int = 1  # Predictor or corrector flag.
        self.friction: float = 0.001  # Friction coefficient (Manning's n if isManning==1, else dimensionless).
        self.isManning: int = 0  # Boolean flag for friction model: 1 for Manning's n, 0 for dimensionless friction factor.
        self.Bcoef: float = 1.0 / 15.0  # Dispersion coefficient.

        self.west_boundary_type: int = 0  # 0: solid wall, 1: sponge layer, 2: waves loaded from file.
        self.east_boundary_type: int = 0  # Same options as west_boundary_type.
        self.south_boundary_type: int = 0  # Same options as west_boundary_type.
        self.north_boundary_type: int = 0  # Same options as west_boundary_type.
        self.incident_wave_type: int = -1
        self.amplitude: float = 0.5

        self.numberOfWaves: int = 0  # Number of waves in the simulation.
        self.waveVectors: np.ndarray = np.array([])

        self.useSedTransModel: bool = False  # Flag to use sediment transport model.
        self.sedC1_d50: float = 0.004  # Median grain size (m).
        self.sedC1_n: float = 0.4  # Porosity.
        self.sedC1_psi: float = 0.0005  # Sediment-related parameter.
        self.sedC1_criticalshields: float = 0.045  # Critical Shields parameter.
        self.sedC1_denrat: float = 2.65  # Density ratio.
        self.sedC1_shields: float = 0.0  # Shields parameter.
        self.sedC1_erosion: float = 0.0  # Erosion coefficient.
        self.sedC1_fallvel: float = 0.0  # Settling velocity coefficient.
        self.sed_C_rho_sat: float = 0.0  # Saturated density coefficient.

        self.dt: float = 0.0  # Time step.
        self.TWO_THETA: float = 0.0  # Twice the Theta parameter.
        self.half_g: float = 0.0  # Half of gravitational constant.
        self.Bcoef_g: float = 0.0  # Bcoef multiplied by gravitational constant.
        self.g_over_dx: float = 0.0  # g divided by dx.
        self.g_over_dy: float = 0.0  # g divided by dy.
        self.one_over_dx: float = 0.0  # 1 divided by dx.
        self.one_over_dy: float = 0.0  # 1 divided by dy.
        self.double_dx = 2.0 * self.dx
        self.double_dy = 2.0 * self.dy
        self.one_over_d2x: float = 0.0  # (1/dx)^2.
        self.one_over_d3x: float = 0.0  # (1/dx)^3.
        self.one_over_d2y: float = 0.0  # (1/dy)^2.
        self.one_over_d3y: float = 0.0  # (1/dy)^3.
        self.one_over_dxdy: float = 0.0  # (1/dx) * (1/dy).
        self.epsilon: float = 0.0  # Epsilon parameter for boundary conditions.
        self.boundary_epsilon: float = 0.0  # Boundary epsilon parameter.
        self.reflect_x: int = 0  # Reflection parameter in x-direction.
        self.reflect_y: int = 0  # Reflection parameter in y-direction.
        self.BoundaryWidth: int = 20  # Width of the boundary layer.
        self.boundary_g: float = 0.0  # Gravitational constant at the boundary.

        self.Px: int = 0
        self.Py: int = 0

        # Additional Simulation Parameters
        self.infiltrationRate: float = 0.001  # Infiltration rate.
        self.showBreaking: bool = False  # Flag to show breaking.

        # Reflection and Boundary Parameters
        self.boundary_shift: int = 4  # Boundary condition shift.
        self.boundary_ny: int = 0
        self.boundary_nx: int = 0

        # Grid and Topography Parameters
        self.nSL: float = 0.0  # North sea level.
        self.sSL: float = 0.0  # South sea level.
        self.eSL: float = 0.0  # East sea level.
        self.wSL: float = 0.0  # West sea level.
        self.seaLevel: float = 0.0
        self.maxtopo: float = 0.0
        self.init_eta: float = 5.0

        self.useBreakingModel: bool = False  # Flag to use breaking model.
        self.dissipation_threshold: float = 1.0  # Dissipation threshold.
        self.whiteWaterDecayRate: float = 1.0  # White water decay rate.
        self.whiteWaterDispersion: float = 0.0  # White water dispersion.
        self.delta_breaking: float = 2.0
        self.delta: float = 0.005
        self.show_window: bool = True
        self.maxsteps: int = 1000  # Maximum number of simulation steps.

        self.outdir: str = "./output"  # Output directory.

        self.T_star_coef: float = 5.0  # T_star coefficient.
        self.dzdt_F_coef: float = 0.15  # dzdt_F coefficient.
        self.dzdt_I_coef: float = 0.50  # dzdt_I coefficient.

        self.device: str = "cuda"