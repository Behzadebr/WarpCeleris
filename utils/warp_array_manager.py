import warp as wp

class WarpArrayManager:
    def __init__(self, sim_params):
        """
        Initialize all Warp arrays.
        
        """
        # Initialize Warp
        wp.init()

        device = sim_params.device

        wp.set_device(device)

        self.Bottom = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.State = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.NewState = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.BottomFriction = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.stateUVstar = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.current_stateUVstar = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.stateFGstar = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.current_stateFGstar = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.Waves = wp.array2d(sim_params.waveVectors, dtype=wp.vec4, device=device)
        
        self.Hnear = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.H = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.U = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.V = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.C = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        
        self.XFlux = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.YFlux = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.oldGradients = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.oldOldGradients = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.predictedGradients = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.dU_by_dt = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        
        self.predictedF_G_star = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.F_G_star_oldGradients = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.F_G_star_oldOldGradients = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        
        self.coefMatx = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.coefMaty = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.temp_PCRx1 = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.temp_PCRy1 = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.temp_PCRx2 = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.temp_PCRy2 = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.temp2_PCRx = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.temp2_PCRy = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.current_buffer = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.next_buffer = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        
        self.State_Sed = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.float32, device=device)
        self.Sed_C = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.XFlux_Sed = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.float32, device=device)
        self.YFlux_Sed = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.float32, device=device)
        self.NewState_Sed = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.float32, device=device)
        self.oldGradients_Sed = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.float32, device=device)
        self.oldOldGradients_Sed = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.float32, device=device)
        self.predictedGradients_Sed = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.float32, device=device)
        self.dU_by_dt_Sed = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.float32, device=device)
        self.erosion_Sed = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.float32, device=device)
        self.deposition_Sed = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.float32, device=device)
        
        self.Auxiliary = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.ShipPressure = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.DissipationFlux = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)
        self.ContSource = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.float32, device=device)
        self.Breaking = wp.zeros(shape=(sim_params.WIDTH, sim_params.HEIGHT), dtype=wp.vec4, device=device)