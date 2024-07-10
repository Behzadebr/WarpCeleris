import numpy as np
import warp as wp

class WarpArrayManager:
    def __init__(self):
        pass

    def create_warp_array(self, data, dtype):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        return wp.array2d(data, dtype=dtype, device="cuda")

    def initialize_wp_arrays(self, sim_params, depth_data, initial_ocean_elevation_data, tridiag_coef_x_data, tridiag_coef_y_data):
        self.txBottom = self.create_warp_array(depth_data, wp.vec4)
        self.txState = self.create_warp_array(initial_ocean_elevation_data, wp.vec4)
        self.txNewState = self.create_warp_array(initial_ocean_elevation_data, wp.vec4)
        self.txstateUVstar = self.create_warp_array(initial_ocean_elevation_data, wp.vec4)
        self.txstateFGstar = self.create_warp_array(initial_ocean_elevation_data, wp.vec4)
        self.current_stateUVstar = self.create_warp_array(initial_ocean_elevation_data, wp.vec4)
        self.current_stateFGstar = self.create_warp_array(initial_ocean_elevation_data, wp.vec4)
        self.txStateUVstar_pred = self.create_warp_array(initial_ocean_elevation_data, wp.vec4)
        self.txWaves = self.create_warp_array(sim_params.waveVectors, wp.vec4)

        zeros_shape = depth_data.shape
        self.txH = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.txU = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.txV = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.txW = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.txC = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.txXFlux = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.txYFlux = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.oldGradients = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.oldOldGradients = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.predictedGradients = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.F_G_star_oldGradients = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.F_G_star_oldOldGradients = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.F_G_star_predictedGradients = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.txNormal = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.txAuxiliary2 = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.txAuxiliary2Out = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.txtemp = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.txtemp2 = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.dU_by_dt = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.F_G_star = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.txShipPressure = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
        self.coefMatx = self.create_warp_array(tridiag_coef_x_data, wp.vec4)
        self.coefMaty = self.create_warp_array(tridiag_coef_y_data, wp.vec4)
        self.newcoef = self.create_warp_array(np.zeros(zeros_shape, dtype=np.float32), wp.vec4)
