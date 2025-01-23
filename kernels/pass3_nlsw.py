import warp as wp
from kernels.kernel_utils import FrictionCalc

@wp.kernel
def Pass3_NLSW(
    NewState: wp.array2d(dtype=wp.vec4),
    State: wp.array2d(dtype=wp.vec4),
    stateUVstar: wp.array2d(dtype=wp.vec4),
    Bottom: wp.array2d(dtype=wp.vec4),
    BottomFriction: wp.array2d(dtype=wp.vec4),
    ShipPressure: wp.array2d(dtype=wp.vec4),
    ContSource: wp.array2d(dtype=wp.float32),
    Breaking: wp.array2d(dtype=wp.vec4),
    XFlux: wp.array2d(dtype=wp.vec4),
    YFlux: wp.array2d(dtype=wp.vec4),
    oldGradients: wp.array2d(dtype=wp.vec4),
    oldOldGradients: wp.array2d(dtype=wp.vec4),
    predictedGradients: wp.array2d(dtype=wp.vec4),
    dU_by_dt: wp.array2d(dtype=wp.vec4),
    predictedF_G_star: wp.array2d(dtype=wp.vec4),
    current_stateUVstar: wp.array2d(dtype=wp.vec4),
    timeScheme: wp.int32,
    pred_or_corrector: wp.int32,
    useBreakingModel: wp.bool,
    showBreaking: wp.int32,
    g: wp.float32,
    dt: wp.float32,
    one_over_dx: wp.float32,
    one_over_dy: wp.float32,
    one_over_d2x: wp.float32,
    one_over_d2y: wp.float32,
    one_over_dxdy: wp.float32,
    g_over_dx: wp.float32,
    g_over_dy: wp.float32,
    delta: wp.float32,
    isManning: wp.int32,
    friction: wp.float32,
    base_depth: wp.float32,
    whiteWaterDispersion: wp.float32,
    whiteWaterDecayRate: wp.float32,
    infiltrationRate: wp.float32,
    width: wp.int32,
    height: wp.int32
):

    ix, iy = wp.tid()

    if ix >= width or iy >= height:
        return

    zero = wp.vec4(0.0, 0.0, 0.0, 0.0)

    # Boundary checks to prevent accessing out-of-bounds indices
    if ix >= (width - 2) or iy >= (height - 2) or ix <= 1 or iy <= 1:
        NewState[ix, iy] = zero
        dU_by_dt[ix, iy] = zero
        predictedF_G_star[ix, iy] = zero
        current_stateUVstar[ix, iy] = zero
        return

    in_state_here = State[ix, iy]
    in_state_here_UV = stateUVstar[ix, iy]

    B_here = Bottom[ix, iy].z
    B_south = Bottom[ix, iy - 1].z
    B_north = Bottom[ix, iy + 1].z
    B_west = Bottom[ix - 1, iy].z
    B_east = Bottom[ix + 1, iy].z

    eta_here = in_state_here.x
    eta_west = State[ix - 1, iy].x
    eta_east = State[ix + 1, iy].x
    eta_south = State[ix, iy - 1].x
    eta_north = State[ix, iy + 1].x

    h_here = eta_here - B_here
    h_west = eta_west - B_west
    h_east = eta_east - B_east
    h_north = eta_north - B_north
    h_south = eta_south - B_south

    h_cut = delta

    # Check if current and all neighbors are dry
    if h_here <= h_cut and h_north <= h_cut and h_east <= h_cut and h_south <= h_cut and h_west <= h_cut:
        NewState[ix, iy] = zero
        dU_by_dt[ix, iy] = zero
        predictedF_G_star[ix, iy] = zero
        current_stateUVstar[ix, iy] = zero
        return

    h_min = wp.vec4(
        wp.min(h_here, h_north),
        wp.min(h_here, h_east),
        wp.min(h_here, h_south),
        wp.min(h_here, h_west)
    )

    detadx = 0.5 * (eta_east - eta_west) * one_over_dx
    detady = 0.5 * (eta_north - eta_south) * one_over_dy

    xflux_here = XFlux[ix, iy]         # Flux at i+1/2, j (x-component)
    xflux_west = XFlux[ix - 1, iy]    # Flux at i-1/2, j (x-component)
    yflux_here = YFlux[ix, iy]         # Flux at i, j+1/2 (y-component)
    yflux_south = YFlux[ix, iy - 1]    # Flux at i, j-1/2 (y-component)

    # Compute friction
    friction_here = wp.max(friction, BottomFriction[ix, iy].x)
    friction_ = FrictionCalc(
        in_state_here.y,     # hu
        in_state_here.z,     # hv
        h_here, 
        base_depth, 
        delta, 
        isManning, 
        g, 
        friction_here
    )

    P_here = ShipPressure[ix, iy].x
    P_left = ShipPressure[ix - 1, iy].x
    P_right = ShipPressure[ix + 1, iy].x
    P_down = ShipPressure[ix, iy - 1].x
    P_up = ShipPressure[ix, iy + 1].x

    press_x = -0.5 * h_here * g_over_dx * (P_right - P_left)
    press_y = -0.5 * h_here * g_over_dy * (P_up - P_down)

    C_state_here = State[ix, iy].w
    C_state_right = State[ix + 1, iy].w
    C_state_left = State[ix - 1, iy].w
    C_state_up = State[ix, iy + 1].w
    C_state_down = State[ix, iy - 1].w
    C_state_up_left = State[ix - 1, iy + 1].w
    C_state_up_right = State[ix + 1, iy + 1].w
    C_state_down_left = State[ix - 1, iy - 1].w
    C_state_down_right = State[ix + 1, iy - 1].w

    Dxx = whiteWaterDispersion
    Dxy = whiteWaterDispersion
    Dyy = whiteWaterDispersion

    hc_by_dx_dx = Dxx * one_over_d2x * (C_state_right - 2.0 * in_state_here.w + C_state_left)
    hc_by_dy_dy = Dyy * one_over_d2y * (C_state_up - 2.0 * in_state_here.w + C_state_down)
    hc_by_dx_dy = 0.25 * Dxy * one_over_dxdy * (C_state_up_right - C_state_up_left - C_state_down_right + C_state_down_left)

    c_dissipation = -whiteWaterDecayRate * C_state_here

    breaking_B = 0.0
    if useBreakingModel:
        breaking_B = Breaking[ix, iy].z  # Breaking front parameter, range [0 - 1]

    # Fix slope near shoreline to avoid numerical instabilities
    if (h_min.x <= h_cut and h_min.z <= h_cut):
        detady = 0.0
    elif h_min.x <= h_cut:
        detady = 1.0 * (eta_here - eta_south) * one_over_dy
    elif h_min.z <= h_cut:
        detady = 1.0 * (eta_north - eta_here) * one_over_dy

    if (h_min.y <= h_cut and h_min.w <= h_cut):
        detadx = 0.0
    elif h_min.y <= h_cut:
        detadx = 1.0 * (eta_here - eta_west) * one_over_dx
    elif h_min.w <= h_cut:
        detadx = 1.0 * (eta_east - eta_here) * one_over_dx

    overflow_dry = 0.0
    if B_here > 0.0:
        overflow_dry = -infiltrationRate  # Hydraulic conductivity of coarse, unsaturated sand

    sx = -g * h_here * detadx - in_state_here.y * friction_ + press_x
    sy = -g * h_here * detady - in_state_here.z * friction_ + press_y

    sc = hc_by_dx_dx + hc_by_dy_dy + 2.0 * hc_by_dx_dy + c_dissipation

    source_term = wp.vec4(
        overflow_dry,
        sx,
        sy,
        sc
    )

    d_by_dt = (xflux_west - xflux_here) * one_over_dx + (yflux_south - yflux_here) * one_over_dy + source_term

    oldies = oldGradients[ix, iy]
    oldOldies = oldOldGradients[ix, iy]

    newState = zero

    # Update state based on time integration scheme
    if timeScheme == 0:
        # Euler method
        newState = in_state_here_UV + dt * d_by_dt
    elif pred_or_corrector == 1:
        # Predictor
        newState = in_state_here_UV + (dt / 12.0) * (23.0 * d_by_dt - 16.0 * oldies + 5.0 * oldOldies)
    elif pred_or_corrector == 2:
        # Corrector
        predicted = predictedGradients[ix, iy]
        newState = in_state_here_UV + (dt / 24.0) * (9.0 * d_by_dt + 19.0 * predicted - 5.0 * oldies + oldOldies)

    # Handle breaking and contaminant sources
    if showBreaking == 1:
        # Add breaking source
        newState.w = wp.max(newState.w, breaking_B)  # Use the breaking value from Breaking array
    elif showBreaking == 2:
        # Add contaminant source
        contaminant_source = ContSource[ix, iy]
        newState.w = wp.min(1.0, newState.w + contaminant_source)

    NewState[ix, iy] = newState
    dU_by_dt[ix, iy] = d_by_dt
    predictedF_G_star[ix, iy] = wp.vec4(0.0, 0.0, 0.0, 1.0)
    current_stateUVstar[ix, iy] = newState
