# utils/tridiag_coef_generator.py

import warp as wp
from kernels.tridiag_coeffs_X import tridiag_coeffs_X
from kernels.tridiag_coeffs_Y import tridiag_coeffs_Y

class TridiagCoefGenerator:
    def __init__(self, sim_params, warp_array_manager):
        """
        Initialize the TridiagCoefGenerator with simulation parameters and Warp arrays.
        """
        self.sim_params = sim_params
        self.warp_array_manager = warp_array_manager

    def generate_coefMatx(self):
        """
        Launch the tridiag_coeffs_X kernel to fill coefMatx.
        """
        wp.launch(
            kernel=tridiag_coeffs_X,
            dim=(self.sim_params.WIDTH, self.sim_params.HEIGHT),
            inputs=[
                self.warp_array_manager.Bottom,
                self.warp_array_manager.coefMatx,
                self.sim_params.dx,
                self.sim_params.Bcoef,
                self.sim_params.WIDTH,
                self.sim_params.HEIGHT
            ],
            device=self.sim_params.device
        )
        # Ensure kernel completion
        wp.synchronize_device(self.sim_params.device)

    def generate_coefMaty(self):
        """
        Launch the tridiag_coeffs_Y kernel to fill coefMaty.
        """
        wp.launch(
            kernel=tridiag_coeffs_Y,
            dim=(self.sim_params.WIDTH, self.sim_params.HEIGHT),
            inputs=[
                self.warp_array_manager.Bottom,
                self.warp_array_manager.coefMaty,
                self.sim_params.dy,
                self.sim_params.Bcoef,
                self.sim_params.WIDTH,
                self.sim_params.HEIGHT
            ],
            device=self.sim_params.device
        )
        # Ensure kernel completion
        wp.synchronize_device(self.sim_params.device)