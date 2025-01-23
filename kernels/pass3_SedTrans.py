import warp as wp

@wp.kernel
def Pass3_SedTrans(
    NewState_Sed: wp.array2d(dtype=wp.float32),
    dU_by_dt_Sed: wp.array2d(dtype=wp.float32),
    erosion_Sed: wp.array2d(dtype=wp.float32),
    deposition_Sed: wp.array2d(dtype=wp.float32),
    State_Sed: wp.array2d(dtype=wp.float32),
    State: wp.array2d(dtype=wp.vec4),
    Bottom: wp.array2d(dtype=wp.vec4),
    XFlux_Sed: wp.array2d(dtype=wp.float32),
    YFlux_Sed: wp.array2d(dtype=wp.float32),
    oldGradients_Sed: wp.array2d(dtype=wp.float32),
    oldOldGradients_Sed: wp.array2d(dtype=wp.float32),
    predictedGradients_Sed: wp.array2d(dtype=wp.float32),
    timeScheme: wp.int32,
    g: wp.float32,
    dt: wp.float32,
    one_over_dx: wp.float32,
    one_over_dy: wp.float32,
    one_over_d2x: wp.float32,
    one_over_d2y: wp.float32,
    one_over_dxdy: wp.float32,
    epsilon: wp.float32,
    isManning: wp.int32,
    friction: wp.float32,
    pred_or_corrector: wp.int32,
    sedC1_shields: wp.float32,
    sedC1_criticalshields: wp.float32,
    sedC1_erosion: wp.float32,
    sedC1_fallvel: wp.float32,
    sedC1_n: wp.float32,
    width: wp.int32,
    height: wp.int32
):

    ix, iy = wp.tid()

    if ix >= width or iy >= height:
        return

    # Boundary checks to prevent accessing out-of-bounds indices
    if ix >= (width - 2) or iy >= (height - 2) or ix <= 1 or iy <= 1:
        NewState_Sed[ix, iy] = 0.0
        dU_by_dt_Sed[ix, iy] = 0.0
        return

    xflux_here = XFlux_Sed[ix, iy]         # Flux at i+1/2, j
    xflux_west = XFlux_Sed[ix - 1, iy]     # Flux at i-1/2, j
    yflux_here = YFlux_Sed[ix, iy]         # Flux at i, j+1/2
    yflux_south = YFlux_Sed[ix, iy - 1]    # Flux at i, j-1/2

    C_state_here = State_Sed[ix, iy]
    C_state_left = State_Sed[ix - 1, iy]
    C_state_right = State_Sed[ix + 1, iy]
    C_state_up = State_Sed[ix, iy + 1]
    C_state_down = State_Sed[ix, iy - 1]
    C_state_up_left = State_Sed[ix - 1, iy + 1]
    C_state_up_right = State_Sed[ix + 1, iy + 1]
    C_state_down_left = State_Sed[ix - 1, iy - 1]
    C_state_down_right = State_Sed[ix + 1, iy - 1]

    Dxx = 1.0
    Dxy = 1.0
    Dyy = 1.0

    # Compute dispersion terms
    hc_by_dx_dx = Dxx * one_over_d2x * (C_state_right - 2.0 * C_state_here + C_state_left)
    hc_by_dy_dy = Dyy * one_over_d2y * (C_state_up - 2.0 * C_state_here + C_state_down)
    hc_by_dx_dy = Dxy * one_over_dxdy * (C_state_up_right - C_state_up_left - C_state_down_right + C_state_down_left) / 4.0

    B = Bottom[ix, iy].z
    in_state_here = State[ix, iy]
    eta = in_state_here.x
    hu = in_state_here.y
    hv = in_state_here.z
    h = eta - B

    divide_by_h = 2.0 * h / (h * h + wp.max(h * h, epsilon))


    f = friction / 2.0
    if isManning == 1:
        f = g * wp.pow(friction, 2.0) * wp.pow(wp.abs(divide_by_h), 1.0 / 3.0)

    u = hu * divide_by_h
    v = hv * divide_by_h

    local_speed = wp.sqrt(u * u + v * v)

    shear_velocity = wp.sqrt(f) * local_speed

    shields = shear_velocity * shear_velocity * sedC1_shields
    erosion = 0.0

    if shields >= sedC1_criticalshields:
        erosion = sedC1_erosion * (shields - sedC1_criticalshields) * local_speed * divide_by_h

    Cmin = wp.max(1.0e-6, C_state_here)
    deposition = wp.min(2.0, (1.0 - sedC1_n) / Cmin) * C_state_here * sedC1_fallvel

    source_term = hc_by_dx_dx + hc_by_dy_dy + 2.0 * hc_by_dx_dy + erosion - deposition

    d_by_dt = (xflux_west - xflux_here) * one_over_dx + (yflux_south - yflux_here) * one_over_dy + source_term

    oldies = oldGradients_Sed[ix, iy]
    oldOldies = oldOldGradients_Sed[ix, iy]

    Out = 0.0

    # Update sediment concentration based on time integration scheme
    if timeScheme == 0:
        # Euler method
        Out = C_state_here + dt * d_by_dt
    elif pred_or_corrector == 1:
        # Predictor
        Out = C_state_here + (dt / 12.0) * (23.0 * d_by_dt - 16.0 * oldies + 5.0 * oldOldies)
    elif pred_or_corrector == 2:
        # Corrector
        predicted = predictedGradients_Sed[ix, iy]
        Out = C_state_here + (dt / 24.0) * (9.0 * d_by_dt + 19.0 * predicted - 5.0 * oldies + oldOldies)

    NewState_Sed[ix, iy] = Out
    dU_by_dt_Sed[ix, iy] = d_by_dt
    erosion_Sed[ix, iy] = erosion
    deposition_Sed[ix, iy] = deposition
