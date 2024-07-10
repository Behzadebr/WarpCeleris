import math
from simulation_parameters import SimulationParameters
from ship_properties import ShipProperties
import numpy as np

class ConstantsCalculator:
    def calculate_constants(self, sim_params: SimulationParameters):
        """
        This function calculates the parameters that are not provided in the JSON file.
        
        Args:
            sim_params (SimulationParameters): The object containing the simulation parameters.
        """
        # Time step is calculated based on Courant number, dx, gravity and base depth.
        sim_params.dt = sim_params.Courant_num * sim_params.dx / math.sqrt(sim_params.g * sim_params.base_depth)
        sim_params.TWO_THETA = sim_params.Theta * 2.0
        sim_params.half_g = sim_params.g / 2.0
        sim_params.Bcoef_g = sim_params.Bcoef * sim_params.g
        sim_params.g_over_dx = sim_params.g / sim_params.dx
        sim_params.g_over_dy = sim_params.g / sim_params.dy
        sim_params.one_over_dx = 1.0 / sim_params.dx
        sim_params.one_over_dy = 1.0 / sim_params.dy
        sim_params.one_over_d2x = sim_params.one_over_dx * sim_params.one_over_dx
        sim_params.one_over_d3x = sim_params.one_over_d2x * sim_params.one_over_dx
        sim_params.one_over_d2y = sim_params.one_over_dy * sim_params.one_over_dy
        sim_params.one_over_d3y = sim_params.one_over_d2y * sim_params.one_over_dy
        sim_params.one_over_dxdy = sim_params.one_over_dx * sim_params.one_over_dy
        sim_params.epsilon = (sim_params.base_depth / 1000.0) ** 2
        sim_params.boundary_epsilon = sim_params.epsilon
        sim_params.boundary_nx = sim_params.WIDTH - 1
        sim_params.boundary_ny = sim_params.HEIGHT - 1
        sim_params.reflect_x = 2 * (sim_params.WIDTH - 3)
        sim_params.reflect_y = 2 * (sim_params.HEIGHT - 3)
        sim_params.BoundaryWidth = 25.0
        sim_params.boundary_g = sim_params.g
        sim_params.Px = math.ceil(math.log2(sim_params.WIDTH))
        sim_params.Py = math.ceil(math.log2(sim_params.HEIGHT))
        sim_params.n_write_interval = math.ceil(sim_params.write_dt / sim_params.dt)
        sim_params.write_dt = sim_params.n_write_interval * sim_params.dt
        sim_params.n_writes = int((sim_params.end_write_time - sim_params.start_write_time) / sim_params.write_dt) + 1

        # Configure thread and dispatch size
        sim_params.ThreadX = 16
        sim_params.ThreadY = 16
        sim_params.DispatchX = math.ceil(sim_params.WIDTH / sim_params.ThreadX)
        sim_params.DispatchY = math.ceil(sim_params.HEIGHT / sim_params.ThreadY)

        # Initialize ship properties
        sim_params.numberOfShips = 10  # should eventually be set by JSON or interface
        sim_params.activeShipIdx = 0
        sim_params.shipProperties = [ShipProperties() for _ in range(sim_params.numberOfShips)]
        
        for i, ship in enumerate(sim_params.shipProperties):
            ship.ship_posx += i * 50  # hack to separate ships for initial demo

            # Pre-calculate various repeatedly used values
            ship.ship_c1a = 1.0 / (2.0 * (ship.ship_length / np.pi) ** 2)
            ship.ship_c1b = 1.0 / (2.0 * (ship.ship_width / np.pi) ** 2)
            ship.ship_c2 = -1.0 / (4.0 * (ship.ship_length / np.pi) ** 2) + 1.0 / (4.0 * (ship.ship_width / np.pi) ** 2)
            ship.ship_c3a = 1.0 / (2.0 * (ship.ship_length / np.pi) ** 2)
            ship.ship_c3b = 1.0 / (2.0 * (ship.ship_width / np.pi) ** 2)

        # Define the ship boundary polygon
        sim_params.shipBoundaryPolygon = np.array([
            [550, 50],
            [1250, 650],
            [1600, 1300],
            [2000, 1550],
            [2200, 3000],
            [3000, 3000],
            [3000, 50]
        ])
