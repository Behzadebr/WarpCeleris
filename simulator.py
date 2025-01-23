import os
import warp as wp
import numpy as np
import time

from my_renderer import OpenGLRenderer  # Custom renderer from Warp opengl renderer

from utils.simulation_parameters import SimulationParameters
from utils.parameters_loader import ParametersLoader
from utils.parameter_calculator import ParameterCalculator
from utils.wave_loader import WaveLoader
from utils.topo_process import TopoProcess
from utils.warp_array_manager import WarpArrayManager
from utils.tridiag_coef_generator import TridiagCoefGenerator
from kernels.kernel_launchers import (
    launch_boundary_pass,
    launch_pass1,
    launch_pass1_sedtrans,
    launch_pass2,
    launch_Tridiag,
    launch_pass3_nlsw,
    launch_pass3_bous,
    launch_pass3_sedtrans,
    launch_pass_breaking,
    launch_Update_Bottom
)

# Initialize Warp
wp.init()

class Simulator:
    def __init__(self, example_name='Crescent_City', device='cuda'):
        """
        Initializes the Simulator by loading parameters, generating data,
        and initializing Warp arrays.
        """
        self.device = device
        self.example_name = example_name
        self.example_path = os.path.join('examples', self.example_name)

        # Verify example directory
        if not os.path.isdir(self.example_path):
            raise FileNotFoundError(f"Example directory '{self.example_path}' does not exist.")

        # Initialize simulation parameters
        self.sim_params = SimulationParameters()
        self.params_loader = ParametersLoader()

        # Path to config.json
        config_path = os.path.join(self.example_path, 'config.json')
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
        self.params_loader.load_parameters(self.sim_params, config_path)

        # Calculate derived parameters
        self.calculator = ParameterCalculator()
        self.calculator.calculate_parameters(self.sim_params)

        # Load wave data
        self.wave_loader = WaveLoader()
        wave_file_path = os.path.join(self.example_path, 'waves.txt')
        if not os.path.isfile(wave_file_path):
            raise FileNotFoundError(f"Waves file '{wave_file_path}' not found.")
        self.wave_loader.load_waves(self.sim_params, wave_file_path)

        # Initialize Warp arrays
        self.warp_array_manager = WarpArrayManager(self.sim_params)

        # Initialize TopoProcess to fill the Bottom array
        bathy_file_path = os.path.join(self.example_path, 'bathy.txt')
        if not os.path.isfile(bathy_file_path):
            raise FileNotFoundError(f"Bathymetry file '{bathy_file_path}' not found.")
        self.topo = TopoProcess(
            filename='bathy.txt',
            datatype='celeris',
            path=self.example_path,
            sim_params=self.sim_params,
            array_manager=self.warp_array_manager
        )

        # Generate tridiagonal coefficients
        self.tridiag_coef_generator = TridiagCoefGenerator(self.sim_params, self.warp_array_manager)
        self.tridiag_coef_generator.generate_coefMatx()
        self.tridiag_coef_generator.generate_coefMaty()

        # Create a renderer instance
        self.renderer = OpenGLRenderer(
            sim_params=self.sim_params,
            title="Celeris Wave Simulation",
            screen_width=1280,
            screen_height=720,
            enable_mouse_interaction=True,
            enable_keyboard_interaction=True,
            device=device
        )

        # Render a plane for the waves
        self.renderer.render_plane(
            name="wave_plane",
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            width=self.sim_params.WIDTH / 100.0,
            length=self.sim_params.HEIGHT / 100.0,
            color=(1.0, 1.0, 1.0),
            color2=None,
            is_template=False
        )

    def evolve_step(self, step):
        """
        Performs a single simulation step
        """
        # Pass1
        launch_pass1(self.warp_array_manager, self.sim_params, device=self.device)

        # Pass1_sedtrans if enabled
        if self.sim_params.useSedTransModel:
            launch_pass1_sedtrans(self.warp_array_manager, self.sim_params, device=self.device)

        # Pass2
        launch_pass2(self.warp_array_manager, self.sim_params, device=self.device)

        # Breaking if enabled
        if self.sim_params.useBreakingModel:
            launch_pass_breaking(
                self.warp_array_manager,
                self.sim_params,
                time=self.sim_params.dt * step - self.sim_params.dt,
                device=self.device
            )

        # Pass3
        if self.sim_params.NLSW_or_Bous == 0:
            launch_pass3_nlsw(self.warp_array_manager, self.sim_params, pred_or_corrector=1, device=self.device)
        else:
            launch_pass3_bous(self.warp_array_manager, self.sim_params, pred_or_corrector=1, device=self.device)

        wp.copy(src=self.warp_array_manager.dU_by_dt, dest=self.warp_array_manager.predictedGradients)

        # SedTrans pass3 if used
        if self.sim_params.useSedTransModel:
            launch_pass3_sedtrans(self.warp_array_manager, self.sim_params, pred_or_corrector=1, device=self.device)
            wp.copy(src=self.warp_array_manager.dU_by_dt_Sed, dest=self.warp_array_manager.predictedGradients_Sed)

        # Boundary pass
        launch_boundary_pass(
            self.warp_array_manager,
            self.sim_params,
            time=self.sim_params.dt * step,
            txState=self.warp_array_manager.current_stateUVstar,
            device=self.device
        )

        # Tridiag
        launch_Tridiag(self.warp_array_manager, self.sim_params, device=self.device)

        # Additional boundary pass if not NLSW
        if self.sim_params.NLSW_or_Bous != 0:
            launch_boundary_pass(
                self.warp_array_manager,
                self.sim_params,
                time=self.sim_params.dt * step,
                txState=self.warp_array_manager.NewState,
                device=self.device
            )

        # If Boussinesq
        if self.sim_params.NLSW_or_Bous == 1:
            wp.copy(src=self.warp_array_manager.F_G_star_oldGradients,    dest=self.warp_array_manager.F_G_star_oldOldGradients)
            wp.copy(src=self.warp_array_manager.predictedF_G_star,        dest=self.warp_array_manager.F_G_star_oldGradients)

        # If timeScheme=2, do predictor/corrector
        if self.sim_params.timeScheme == 2:
            wp.copy(src=self.warp_array_manager.NewState, dest=self.warp_array_manager.State)

            if self.sim_params.useSedTransModel:
                wp.copy(src=self.warp_array_manager.NewState_Sed, dest=self.warp_array_manager.State_Sed)

            # Pass1
            launch_pass1(self.warp_array_manager, self.sim_params, device=self.device)
            if self.sim_params.useSedTransModel:
                launch_pass1_sedtrans(self.warp_array_manager, self.sim_params, device=self.device)

            # Pass2
            launch_pass2(self.warp_array_manager, self.sim_params, device=self.device)

            # Breaking if needed
            if self.sim_params.useBreakingModel:
                launch_pass_breaking(
                    self.warp_array_manager,
                    self.sim_params,
                    time=self.sim_params.dt * step,
                    device=self.device
                )

            # Pass3
            if self.sim_params.NLSW_or_Bous == 0:
                launch_pass3_nlsw(self.warp_array_manager, self.sim_params, pred_or_corrector=2, device=self.device)
            else:
                launch_pass3_bous(self.warp_array_manager, self.sim_params, pred_or_corrector=2, device=self.device)

            if self.sim_params.useSedTransModel:
                launch_pass3_sedtrans(self.warp_array_manager, self.sim_params, pred_or_corrector=2, device=self.device)

            # Boundary pass
            launch_boundary_pass(
                self.warp_array_manager,
                self.sim_params,
                time=self.sim_params.dt * step,
                txState=self.warp_array_manager.current_stateUVstar,
                device=self.device
            )

            # Tridiag
            launch_Tridiag(self.warp_array_manager, self.sim_params, device=self.device)

            if self.sim_params.NLSW_or_Bous != 0:
                launch_boundary_pass(
                    self.warp_array_manager,
                    self.sim_params,
                    time=self.sim_params.dt * step,
                    txState=self.warp_array_manager.NewState,
                    device=self.device
                )

            if self.sim_params.useSedTransModel:
                launch_Update_Bottom(self.warp_array_manager, self.sim_params, device=self.device)

                if self.sim_params.NLSW_or_Bous == 1:
                    self.topo.fill_bottom_field()
                    self.tridiag_coef_generator.generate_coefMatx()
                    self.tridiag_coef_generator.generate_coefMaty()

        wp.copy(src=self.warp_array_manager.oldGradients,    dest=self.warp_array_manager.oldOldGradients)
        wp.copy(src=self.warp_array_manager.predictedGradients, dest=self.warp_array_manager.oldGradients)

        wp.copy(src=self.warp_array_manager.NewState, dest=self.warp_array_manager.State)
        wp.copy(src=self.warp_array_manager.current_stateUVstar, dest=self.warp_array_manager.stateUVstar)

        if self.sim_params.useSedTransModel:
            wp.copy(src=self.warp_array_manager.oldGradients_Sed,    dest=self.warp_array_manager.oldOldGradients_Sed)
            wp.copy(src=self.warp_array_manager.predictedGradients_Sed, dest=self.warp_array_manager.oldGradients_Sed)
            wp.copy(src=self.warp_array_manager.NewState_Sed,       dest=self.warp_array_manager.State_Sed)


    def run_simulation(self, total_steps=1000, log_interval=100):
        """
        Runs the simulation for number of steps
        """
        start_time = time.time()

        for step in range(total_steps):
            if not self.renderer.is_running():
                # Stop simulation if closed the window
                break
                
            self.evolve_step(step)

            # After each evolve_step, update textures and render
            self.renderer.update_wave_texture(
                wave_array=self.warp_array_manager.State,
                width=self.sim_params.WIDTH,
                height=self.sim_params.HEIGHT
            )

            # Time uniform for the fragment shader
            current_time = self.sim_params.dt * step
            self.renderer.set_time(current_time)

            # Draw one frame
            self.renderer.begin_frame(current_time)
            self.renderer.end_frame()
            self.renderer.update()

            if step == 0 or ((step + 1) % log_interval) == 0:
                elapsed = time.time() - start_time
                print(f"Step {step + 1}/{total_steps} completed in {elapsed:.2f} seconds.")
                # Save state if outdir is specified
                if self.sim_params.outdir:
                    state = self.warp_array_manager.State.numpy()
                    os.makedirs(self.sim_params.outdir, exist_ok=True)
                    np.save(f"{self.sim_params.outdir}/state_{step}.npy", state)

        self.renderer.close()