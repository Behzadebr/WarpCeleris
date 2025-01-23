import warp as wp
from kernels.kernel_utils import FrictionCalc

@wp.kernel
def Pass3_Bous(
    NewState: wp.array2d(dtype=wp.vec4),
    dU_by_dt: wp.array2d(dtype=wp.vec4),
    predictedF_G_star: wp.array2d(dtype=wp.vec4),
    current_stateUVstar: wp.array2d(dtype=wp.vec4),
    State: wp.array2d(dtype=wp.vec4),
    stateUVstar: wp.array2d(dtype=wp.vec4),
    Bottom: wp.array2d(dtype=wp.vec4),
    BottomFriction: wp.array2d(dtype=wp.vec4),
    XFlux: wp.array2d(dtype=wp.vec4),
    YFlux: wp.array2d(dtype=wp.vec4),
    F_G_star_oldOldGradients: wp.array2d(dtype=wp.vec4),
    oldGradients: wp.array2d(dtype=wp.vec4),
    oldOldGradients: wp.array2d(dtype=wp.vec4),
    predictedGradients: wp.array2d(dtype=wp.vec4),
    ShipPressure: wp.array2d(dtype=wp.vec4),
    ContSource: wp.array2d(dtype=wp.float32),
    Breaking: wp.array2d(dtype=wp.vec4),
    DissipationFlux: wp.array2d(dtype=wp.vec4),
    timeScheme: wp.int32,
    pred_or_corrector: wp.int32,
    width: wp.int32,
    height: wp.int32,
    dt: wp.float32,
    one_over_dx: wp.float32,
    one_over_dy: wp.float32,
    g_over_dx: wp.float32,
    g_over_dy: wp.float32,
    one_over_d2x: wp.float32,
    one_over_d3x: wp.float32,
    one_over_d2y: wp.float32,
    one_over_d3y: wp.float32,
    one_over_dxdy: wp.float32,
    Bcoef: wp.float32,
    Bcoef_g: wp.float32,
    delta: wp.float32,
    base_depth: wp.float32,
    whiteWaterDispersion: wp.float32,
    whiteWaterDecayRate: wp.float32,
    useBreakingModel: wp.bool,
    showBreaking: wp.int32,
    g: wp.float32,
    isManning: wp.int32,
    friction: wp.float32,
    infiltrationRate: wp.float32
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

    B_here = Bottom[ix, iy].z

    in_state_here = State[ix, iy]
    in_state_here_UV = stateUVstar[ix, iy]
    h_here = in_state_here.x - B_here

    B_south = Bottom[ix, iy - 1].z
    B_north = Bottom[ix, iy + 1].z
    B_west = Bottom[ix - 1, iy].z
    B_east = Bottom[ix + 1, iy].z

    eta_here = in_state_here.x
    eta_west = State[ix - 1, iy].x
    eta_east = State[ix + 1, iy].x
    eta_south = State[ix, iy - 1].x
    eta_north = State[ix, iy + 1].x

    h_west = eta_west - B_west
    h_east = eta_east - B_east
    h_north = eta_north - B_north
    h_south = eta_south - B_south

    if h_here <= delta:
        if h_north <= delta and h_east <= delta and h_south <= delta and h_west <= delta:
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

    xflux_here = XFlux[ix, iy]         # Flux at i+1/2, j (x-component)
    xflux_west = XFlux[ix - 1, iy]    # Flux at i-1/2, j (x-component)
    yflux_here = YFlux[ix, iy]         # Flux at j+1/2 (y-component)
    yflux_south = YFlux[ix, iy - 1]    # Flux at j-1/2 (y-component)

    detadx = 0.5 * (eta_east - eta_west) * one_over_dx
    detady = 0.5 * (eta_north - eta_south) * one_over_dy

    F_star = 0.0
    G_star = 0.0
    Psi1x = 0.0
    Psi2x = 0.0
    Psi1y = 0.0
    Psi2y = 0.0

    d_here = -B_here
    near_dry = Bottom[ix, iy].w

    # Proceed only if not near an initially dry cell
    if near_dry > 0.0:
        d2_here = d_here * d_here
        d3_here = d2_here * d_here

        in_state_left = State[ix - 1, iy]
        in_state_right = State[ix + 1, iy]
        in_state_up = State[ix, iy + 1]
        in_state_down = State[ix, iy - 1]
        in_state_up_left = State[ix - 1, iy + 1]
        in_state_up_right = State[ix + 1, iy + 1]
        in_state_down_left = State[ix - 1, iy - 1]
        in_state_down_right = State[ix + 1, iy - 1]

        F_G_star_oldOldies = F_G_star_oldOldGradients[ix, iy]

        d_left = -B_west
        d_right = -B_east
        d_down = -B_south
        d_up = -B_north

        d_left_left = wp.max(0.0 , -Bottom[ix - 2, iy].z)
        d_right_right = wp.max(0.0, -Bottom[ix + 2, iy].z)
        d_down_down = wp.max(0.0, -Bottom[ix, iy - 2].z)
        d_up_up = wp.max(0.0, -Bottom[ix, iy + 2].z)

        # Load eta values for higher-order derivatives
        eta_here = in_state_here.x
        eta_left = in_state_left.x
        eta_right = in_state_right.x
        eta_down = in_state_down.x
        eta_up = in_state_up.x

        eta_left_left = State[ix - 2, iy].x
        eta_right_right = State[ix + 2, iy].x
        eta_down_down = State[ix, iy - 2].x
        eta_up_up = State[ix, iy + 2].x

        eta_up_left = in_state_up_left.x
        eta_up_right = in_state_up_right.x
        eta_down_left = in_state_down_left.x
        eta_down_right = in_state_down_right.x

        detadx = (-eta_right_right + 8.0 * eta_right - 8.0 * eta_left + eta_left_left) * one_over_dx / 12.0
        detady = (-eta_up_up + 8.0 * eta_up - 8.0 * eta_down + eta_down_down) * one_over_dy / 12.0

        u_up = in_state_up.y
        u_down = in_state_down.y
        u_right = in_state_right.y
        u_left = in_state_left.y
        u_up_right = in_state_up_right.y
        u_down_right = in_state_down_right.y
        u_up_left = in_state_up_left.y
        u_down_left = in_state_down_left.y

        v_up = in_state_up.z
        v_down = in_state_down.z
        v_right = in_state_right.z
        v_left = in_state_left.z
        v_up_right = in_state_up_right.z
        v_down_right = in_state_down_right.z
        v_up_left = in_state_up_left.z
        v_down_left = in_state_down_left.z

        dd_by_dx = (-d_right_right + 8.0 * d_right - 8.0 * d_left + d_left_left) * one_over_dx / 12.0
        dd_by_dy = (-d_up_up + 8.0 * d_up - 8.0 * d_down + d_down_down) * one_over_dy / 12.0

        eta_by_dx_dy = 0.25 * one_over_dxdy * (eta_up_right - eta_down_right - eta_up_left + eta_down_left)
        eta_by_dx_dx = one_over_d2x * (eta_right - 2.0 * eta_here + eta_left)
        eta_by_dy_dy = one_over_d2y * (eta_up - 2.0 * eta_here + eta_down)

        # Calculate F_star and G_star using higher-order derivatives
        F_star = (1.0 / 6.0) * d_here * (
            dd_by_dx * (0.5 * one_over_dy) * (v_up - v_down) +
            dd_by_dy * (0.5 * one_over_dx) * (v_right - v_left)
        ) + (Bcoef + 1.0 / 3.0) * d2_here * (one_over_dxdy * 0.25) * (
            v_up_right - v_down_right - v_up_left + v_down_left
        )

        G_star = (1.0 / 6.0) * d_here * (
            dd_by_dx * (0.5 * one_over_dy) * (u_up - u_down) +
            dd_by_dy * (0.5 * one_over_dx) * (u_right - u_left)
        ) + (Bcoef + 1.0 / 3.0) * d2_here * (one_over_dxdy * 0.25) * (
            u_up_right - u_down_right - u_up_left + u_down_left
        )

        Psi1x = Bcoef_g * d3_here * (
            (eta_right_right - 2.0 * eta_right + 2.0 * eta_left - eta_left_left) * (0.5 * one_over_d3x) +
            (eta_up_right - eta_up_left - 2.0 * eta_right + 2.0 * eta_left + eta_down_right - eta_down_left) * (0.5 * one_over_dx * one_over_d2y)
        )

        Psi2x = Bcoef_g * d2_here * (
            dd_by_dx * (2.0 * eta_by_dx_dx + eta_by_dy_dy) +
            dd_by_dy * eta_by_dx_dy
        ) + (F_star - F_G_star_oldOldies.y) / dt / 2.0

        Psi1y = Bcoef_g * d3_here * (
            (eta_up_up - 2.0 * eta_up + 2.0 * eta_down - eta_down_down) * (0.5 * one_over_d3y) +
            (eta_up_right + eta_up_left - 2.0 * eta_up + 2.0 * eta_down - eta_down_right - eta_down_left) * (0.5 * one_over_dx * one_over_d2x)
        )

        Psi2y = Bcoef_g * d2_here * (
            dd_by_dy * (2.0 * eta_by_dy_dy + eta_by_dx_dx) +
            dd_by_dx * eta_by_dx_dy
        ) + (G_star - F_G_star_oldOldies.z) / dt / 2.0

    # Compute friction using the FrictionCalc utility
    friction_here = wp.max(friction, BottomFriction[ix, iy].x)
    friction_ = FrictionCalc(
        in_state_here_UV.y,
        in_state_here_UV.z,
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
    C_state_left = State[ix - 1, iy].w
    C_state_right = State[ix + 1, iy].w
    C_state_up = State[ix, iy + 1].w
    C_state_down = State[ix, iy - 1].w
    C_state_up_left = State[ix - 1, iy + 1].w
    C_state_up_right = State[ix + 1, iy + 1].w
    C_state_down_left = State[ix - 1, iy - 1].w
    C_state_down_right = State[ix + 1, iy - 1].w

    Dxx = whiteWaterDispersion
    Dxy = whiteWaterDispersion
    Dyy = whiteWaterDispersion

    hc_by_dx_dx = Dxx * one_over_d2x * (C_state_right - 2.0 * C_state_here + C_state_left)
    hc_by_dy_dy = Dyy * one_over_d2y * (C_state_up - 2.0 * C_state_here + C_state_down)
    hc_by_dx_dy = 0.25 * Dxy * one_over_dxdy * (C_state_up_right - C_state_up_left - C_state_down_right + C_state_down_left)

    c_dissipation = -whiteWaterDecayRate * C_state_here

    breaking_x = 0.0
    breaking_y = 0.0
    breaking_B = 0.0

    if useBreakingModel:
        breaking_B = Breaking[ix, iy].z  # Breaking front parameter, range [0 - 1]
        nu_flux_here = DissipationFlux[ix, iy]
        nu_flux_right = DissipationFlux[ix + 1, iy]
        nu_flux_left = DissipationFlux[ix - 1, iy]
        nu_flux_up = DissipationFlux[ix, iy + 1]
        nu_flux_down = DissipationFlux[ix, iy - 1]

        dPdxx = 0.5 * (nu_flux_right.x - nu_flux_left.x) * one_over_dx
        dPdyx = 0.5 * (nu_flux_right.y - nu_flux_left.y) * one_over_dx
        dPdyy = 0.5 * (nu_flux_up.y - nu_flux_down.y) * one_over_dy

        dQdxx = 0.5 * (nu_flux_right.z - nu_flux_left.z) * one_over_dx
        dQdxy = 0.5 * (nu_flux_up.z - nu_flux_down.z) * one_over_dy
        dQdyy = 0.5 * (nu_flux_up.w - nu_flux_down.w) * one_over_dy

        if near_dry > 0.0:
            breaking_x = dPdxx + 0.5 * dPdyy + 0.5 * dQdxy
            breaking_y = dQdyy + 0.5 * dPdyx + 0.5 * dQdxx

    # Fix slope near shoreline
    if (h_min.x <= delta and h_min.z <= delta):
        detady = 0.0
    elif h_min.x <= delta:
        detady = 1.0 * (eta_here - eta_south) * one_over_dy
    elif h_min.z <= delta:
        detady = 1.0 * (eta_north - eta_here) * one_over_dy

    if (h_min.y <= delta and h_min.w <= delta):
        detadx = 0.0
    elif h_min.y <= delta:
        detadx = 1.0 * (eta_here - eta_west) * one_over_dx
    elif h_min.w <= delta:
        detadx = 1.0 * (eta_east - eta_here) * one_over_dx

    overflow_dry = 0.0
    if B_here > 0.0:
        overflow_dry = -infiltrationRate  # Hydraulic conductivity of coarse, unsaturated sand

    # Compute source terms for momentum and sediment concentration
    sx = -g * h_here * detadx - in_state_here.y * friction_ + breaking_x + (Psi1x + Psi2x) + press_x
    sy = -g * h_here * detady - in_state_here.z * friction_ + breaking_y + (Psi1y + Psi2y) + press_y

    source_term = wp.vec4(
        overflow_dry,
        sx,
        sy,
        hc_by_dx_dx + hc_by_dy_dy + 2.0 * hc_by_dx_dy + c_dissipation
    )

    d_by_dt = (xflux_west - xflux_here) * one_over_dx + (yflux_south - yflux_here) * one_over_dy + source_term

    oldies = oldGradients[ix, iy]
    oldOldies = oldOldGradients[ix, iy]

    newState = zero
    F_G_here = wp.vec4(0.0, F_star, G_star, 0.0)

    # Update state based on time integration scheme
    if timeScheme == 0:
        # Euler method
        newState = in_state_here_UV + dt * d_by_dt
    elif pred_or_corrector == 1:
        # Predictor step
        newState = in_state_here_UV + (dt / 12.0) * (23.0 * d_by_dt - 16.0 * oldies + 5.0 * oldOldies)
    elif pred_or_corrector == 2:
        # Corrector step
        predicted = predictedGradients[ix, iy]
        newState = in_state_here_UV + (dt / 24.0) * (9.0 * d_by_dt + 19.0 * predicted - 5.0 * oldies + oldOldies)

    # Handle breaking and contaminant sources
    if showBreaking == 1:
        newState.w = wp.max(newState.w, breaking_B)  # Use the breaking value from Breaking array
    elif showBreaking == 2:
        # Add contaminant source
        contaminant_source = ContSource[ix, iy]
        newState.w = wp.min(1.0, newState.w + contaminant_source)

    NewState[ix, iy] = newState
    dU_by_dt[ix, iy] = d_by_dt
    predictedF_G_star[ix, iy] = F_G_here
    current_stateUVstar[ix, iy] = newState