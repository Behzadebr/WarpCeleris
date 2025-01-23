import math
from utils.simulation_parameters import SimulationParameters

class ParameterCalculator:
    def calculate_parameters(self, sim_params: SimulationParameters):
        """
        Calculate derived parameters based on the initialized simulation parameters.

        """

        # Time Step Calculation
        try:
            sim_params.dt = sim_params.Courant_num * sim_params.dx / math.sqrt(sim_params.g * sim_params.base_depth)
        except ZeroDivisionError:
            sim_params.dt = 0.0

        # Derived Physical Parameters
        sim_params.TWO_THETA = 2.0 * sim_params.Theta
        sim_params.half_g = 0.5 * sim_params.g
        sim_params.Bcoef_g = sim_params.Bcoef * sim_params.g
        sim_params.g_over_dx = sim_params.g / sim_params.dx if sim_params.dx != 0 else 0.0
        sim_params.g_over_dy = sim_params.g / sim_params.dy if sim_params.dy != 0 else 0.0
        sim_params.one_over_dx = 1.0 / sim_params.dx if sim_params.dx != 0 else 0.0
        sim_params.one_over_dy = 1.0 / sim_params.dy if sim_params.dy != 0 else 0.0
        sim_params.one_over_d2x = sim_params.one_over_dx ** 2
        sim_params.one_over_d3x = sim_params.one_over_d2x * sim_params.one_over_dx
        sim_params.one_over_d2y = sim_params.one_over_dy ** 2
        sim_params.one_over_d3y = sim_params.one_over_d2y * sim_params.one_over_dy
        sim_params.one_over_dxdy = sim_params.one_over_dx * sim_params.one_over_dy
        sim_params.epsilon = (sim_params.base_depth / 1000.0) ** 2
        sim_params.boundary_epsilon = sim_params.epsilon
        sim_params.boundary_nx = sim_params.WIDTH - 1
        sim_params.boundary_ny = sim_params.HEIGHT - 1
        sim_params.reflect_x = 2 * (sim_params.WIDTH - 3)
        sim_params.reflect_y = 2 * (sim_params.HEIGHT - 3)
        sim_params.boundary_g = sim_params.g

        sim_params.Px = math.ceil(math.log2(sim_params.WIDTH)) if sim_params.WIDTH > 0 else 0
        sim_params.Py = math.ceil(math.log2(sim_params.HEIGHT)) if sim_params.HEIGHT > 0 else 0


        sim_params.sedC1_shields = 1.0 / ((sim_params.sedC1_denrat - 1.0) * sim_params.g * sim_params.sedC1_d50 / 1000.0)
        sim_params.sedC1_erosion = sim_params.sedC1_psi * (sim_params.sedC1_d50 / 1000.0) ** -0.2
        sim_params.sedC1_fallvel = math.sqrt(
            (4.0 / 3.0) * sim_params.g * sim_params.sedC1_d50 / 1000.0 / 0.2 * (sim_params.sedC1_denrat - 1.0)
        )
        sim_params.sed_C_rho_sat = (sim_params.sedC1_n + sim_params.sedC1_denrat * (1 - sim_params.sedC1_n)) / 1000.0