import warp as wp
from datetime import datetime
from simulation_parameters import SimulationParameters
from ship_properties import ShipProperties
from utils.parameters_loader import ParametersLoader
from utils.constants_calculator import ConstantsCalculator
from utils.depth_surface_generator import DepthSurfaceGenerator
from utils.initial_ocean_elevation_generator import InitialOceanElevationGenerator
from utils.wave_loader import WaveLoader
from utils.tridiag_coef_generator import TridiagCoefGenerator
from utils.warp_array_manager import WarpArrayManager
from kernels.kernel_launchers import (
    launch_boundary_pass, launch_pass1, launch_pass2, 
    launch_tridiag_pcrx, launch_tridiag_pcry, 
    launch_pass3_nlsw, launch_pass3_bous
)

# Initialize Warp
wp.init()

# Initialize simulation parameters and ship properties
sim_params = SimulationParameters()
ship_properties = ShipProperties()

# Load parameters from JSON
params_loader = ParametersLoader()
params_loader.load_parameters(sim_params, 'data/config.json')

# Calculate constants
const_calculator = ConstantsCalculator()
const_calculator.calculate_constants(sim_params)

# Generate depth data
depth_surface_generator = DepthSurfaceGenerator()
depth_data = depth_surface_generator.generate_depth_surface(sim_params, 'data/bathy.txt')

# Generate initial ocean elevation data
initial_ocean_elevation_generator = InitialOceanElevationGenerator()
initial_ocean_elevation_data = initial_ocean_elevation_generator.generate_initial_ocean_elevation(sim_params, depth_data)

# Load wave data
wave_loader = WaveLoader()
wave_loader.load_irr_waves(sim_params, 'data/waves.txt', 0)

# Generate tridiagonal matrices
tridiag_coef_generator = TridiagCoefGenerator()
tridiag_coef_x_data = tridiag_coef_generator.generate_tridiag_coef_x(sim_params, depth_data)
tridiag_coef_y_data = tridiag_coef_generator.generate_tridiag_coef_y(sim_params, depth_data)

# Initialize Warp arrays
wp_arr = WarpArrayManager()
wp_arr.initialize_wp_arrays(sim_params, depth_data, initial_ocean_elevation_data, tridiag_coef_x_data, tridiag_coef_y_data)

total_time: float = 0.0
frame_count: int = 0
start_time = None

# Main simulation loop
def run_simulation(wp_arr, sim_params, total_steps):
    global frame_count, total_time, start_time

    start_time = datetime.now()  # Start wall clock timer

    for step in range(total_steps):
        frame_count += 1  # Frame or time step counter
        total_time = frame_count * sim_params.dt  # Simulation time

        # Call boundary pass first
        launch_boundary_pass(wp_arr, sim_params, total_time)

        # !!PREDICTOR!!

        # Execute Pass1
        launch_pass1(wp_arr, sim_params)

        # Execute Pass2
        launch_pass2(wp_arr, sim_params)

        # Pass3's
        sim_params.pred_or_corrector = 1  # This is used inside PassX to determine the proper State update equation to use
        if sim_params.NLSW_or_Bous == 0:  # NLSW
            launch_pass3_nlsw(wp_arr, sim_params)
        elif sim_params.NLSW_or_Bous == 1:  # Bous
            launch_pass3_bous(wp_arr, sim_params)

        # Execute BoundaryPass
        launch_boundary_pass(wp_arr, sim_params, total_time)

        # Execute TriDiag_PCRx
        launch_tridiag_pcrx(wp_arr, sim_params)

        # Execute TriDiag_PCRy
        launch_tridiag_pcry(wp_arr, sim_params)

        # !!END PREDICTOR!!

        # Step back values of F* and G*
        wp.copy(src=wp_arr.F_G_star_oldGradients, dest=wp_arr.F_G_star_oldOldGradients)
        wp.copy(src=wp_arr.F_G_star, dest=wp_arr.F_G_star_oldGradients)

        # !!CORRECTOR!!
        if sim_params.timeScheme == 2:  # Only called when using Predictor+Corrector method
            sim_params.pred_or_corrector = 2

            # Copy txState into txState_pred for the corrector equation
            wp.copy(src=wp_arr.txstateUVstar, dest=wp_arr.txStateUVstar_pred)
            wp.copy(src=wp_arr.txNewState, dest=wp_arr.txState)

            # Execute Pass1
            launch_pass1(wp_arr, sim_params)

            # Execute Pass2
            launch_pass2(wp_arr, sim_params)

            if sim_params.NLSW_or_Bous == 0:  # NLSW
                launch_pass3_nlsw(wp_arr, sim_params)
            elif sim_params.NLSW_or_Bous == 1:  # Bous
                launch_pass3_bous(wp_arr, sim_params)

            # Execute BoundaryPass
            launch_boundary_pass(wp_arr, sim_params, total_time)

            # Execute TriDiag_PCRx
            launch_tridiag_pcrx(wp_arr, sim_params)

            # Execute TriDiag_PCRy
            launch_tridiag_pcry(wp_arr, sim_params)
        # !!END CORRECTOR!!

        # Shift gradient textures
        wp.copy(src=wp_arr.oldGradients, dest=wp_arr.oldOldGradients)
        wp.copy(src=wp_arr.predictedGradients, dest=wp_arr.oldGradients)

        # Copy future_ocean_texture back to ocean_texture
        wp.copy(src=wp_arr.txNewState, dest=wp_arr.txState)
        wp.copy(src=wp_arr.current_stateUVstar, dest=wp_arr.txstateUVstar)

        # Calculate and log the simulation speed
        elapsed_real_time = (datetime.now() - start_time).total_seconds()
        sim_speed = total_time / elapsed_real_time if elapsed_real_time > 0 else 0
        # print(f"Iteration {frame_count}: Simulated Time = {total_time:.2f} s, Real Time = {elapsed_real_time:.2f} s, Speed = {sim_speed:.2f}x real time")

        # # For debugging, print the shape and first element of txState Warp array
        # print(frame_count)
        print(wp_arr.txState.numpy()[1000, 500])

# Define the total number of steps for the simulation
total_steps = 1000  # Example: 1000 time steps

# Run the simulation
run_simulation(wp_arr, sim_params, total_steps)


# print(wp_arr.coefMatx.numpy()[0, 500])
# print(wp_arr.coefMaty.numpy()[0, 500])
