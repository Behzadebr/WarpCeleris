import warp as wp
import math

# A function to calculate the friction term based on the momentum and depth.
@wp.func
def FrictionCalc(hu: wp.float32, hv: wp.float32, h: wp.float32, epsilon: wp.float32, g: wp.float32, friction: wp.float32, isManning: wp.int32) -> wp.float32:
    h2 = h * h
    divide_by_h = 2.0 * h / wp.sqrt(h2 + wp.max(h2, epsilon))

    if isManning == 1:
        f = g * wp.pow(friction, 2.0) * wp.pow(wp.abs(divide_by_h), 1.0 / 3.0)
    else:
        f = friction / 2.0

    f = f * wp.sqrt(hu * hu + hv * hv) * divide_by_h * divide_by_h
    return f

# The main kernel
@wp.kernel
def pass3_bous_kernel(txState: wp.array2d(dtype=wp.vec4), txBottom: wp.array2d(dtype=wp.vec4), txH: wp.array2d(dtype=wp.vec4), txXFlux: wp.array2d(dtype=wp.vec4), 
                      txYFlux: wp.array2d(dtype=wp.vec4), oldGradients: wp.array2d(dtype=wp.vec4), oldOldGradients: wp.array2d(dtype=wp.vec4), 
                      predictedGradients: wp.array2d(dtype=wp.vec4), F_G_star_oldOldGradients: wp.array2d(dtype=wp.vec4), txstateUVstar: wp.array2d(dtype=wp.vec4), 
                      txShipPressure: wp.array2d(dtype=wp.vec4), txNewState: wp.array2d(dtype=wp.vec4), dU_by_dt: wp.array2d(dtype=wp.vec4), 
                      F_G_star: wp.array2d(dtype=wp.vec4), current_stateUVstar: wp.array2d(dtype=wp.vec4), width: wp.int32, height: wp.int32, dt: wp.float32, 
                      dx: wp.float32, dy: wp.float32, one_over_dx: wp.float32, one_over_dy: wp.float32, g_over_dx: wp.float32, g_over_dy: wp.float32, timeScheme: wp.int32, 
                      epsilon: wp.float32, isManning: wp.int32, g: wp.float32, friction: wp.float32, pred_or_corrector: wp.int32, Bcoef: wp.float32, 
                      Bcoef_g: wp.float32, one_over_d2x: wp.float32, one_over_d3x: wp.float32, one_over_d2y: wp.float32, one_over_d3y: wp.float32, one_over_dxdy: wp.float32, 
                      seaLevel: wp.float32):
    ix, iy = wp.tid()
    idx = wp.vec2i(ix, iy)

    zero = wp.vec4(0.0, 0.0, 0.0, 0.0)
    txNewState[ix, iy] = zero
    dU_by_dt[ix, iy] = zero
    F_G_star[ix, iy] = zero
    current_stateUVstar[ix, iy] = zero

    if ix >= width - 2 or iy >= height - 2 or ix <= 1 or iy <= 1:
        return

    leftIdx = wp.vec2i(ix - 1, iy)
    rightIdx = wp.vec2i(ix + 1, iy)
    downIdx = wp.vec2i(ix, iy + 1)
    upIdx = wp.vec2i(ix, iy - 1)
    upleftIdx = wp.vec2i(ix - 1, iy - 1)
    uprightIdx = wp.vec2i(ix + 1, iy - 1)
    downleftIdx = wp.vec2i(ix - 1, iy + 1)
    downrightIdx = wp.vec2i(ix + 1, iy + 1)

    B_here = txBottom[ix, iy][2]
    near_dry = txBottom[ix, iy][3]

    in_state_here = txState[ix, iy]
    h_vec = txH[ix, iy]
    h_here = in_state_here[0] - B_here

    xflux_here = txXFlux[ix, iy]
    xflux_west = txXFlux[leftIdx[0], leftIdx[1]]
    yflux_here = txYFlux[ix, iy]
    yflux_south = txYFlux[downIdx[0], downIdx[1]]

    B_south = txBottom[downIdx[0], downIdx[1]][2]
    B_north = txBottom[upIdx[0], upIdx[1]][2]
    B_west = txBottom[leftIdx[0], leftIdx[1]][2]
    B_east = txBottom[rightIdx[0], rightIdx[1]][2]
    
    eta_west = txState[leftIdx[0], leftIdx[1]][0]
    eta_east = txState[rightIdx[0], rightIdx[1]][0]
    eta_south = txState[downIdx[0], downIdx[1]][0]
    eta_north = txState[upIdx[0], upIdx[1]][0]

    detadx = 0.5 * (eta_east - eta_west) * one_over_dx
    detady = 0.5 * (eta_north - eta_south) * one_over_dy
    
    minH = wp.min(h_vec)
    dB = wp.max(wp.max(wp.max(wp.abs(B_south - B_here), wp.abs(B_north - B_here)), wp.abs(B_west - B_here)), wp.abs(B_east - B_here))
    u_here = in_state_here[1]
    v_here = in_state_here[2]
    speed2_here = u_here * u_here + v_here * v_here

    if minH * minH < 2.0 * dx * dB and speed2_here < 0.00001 * dB * g:
        detady = 0.0
        detadx = 0.0

    oldies = oldGradients[ix, iy]
    oldOldies = oldOldGradients[ix, iy]

    F_star = 0.0
    G_star = 0.0
    Psi1x = 0.0
    Psi2x = 0.0
    Psi1y = 0.0
    Psi2y = 0.0
    d_here = seaLevel - B_here

    if near_dry > 0:
        d2_here = d_here * d_here
        d3_here = d2_here * d_here

        leftleftIdx = wp.vec2i(ix - 2, iy)
        rightrightIdx = wp.vec2i(ix + 2, iy)
        downdownIdx = wp.vec2i(ix, iy + 2)
        upupIdx = wp.vec2i(ix, iy - 2)

        in_state_right = txState[rightIdx[0], rightIdx[1]]
        in_state_left = txState[leftIdx[0], leftIdx[1]]
        in_state_up = txState[upIdx[0], upIdx[1]]
        in_state_down = txState[downIdx[0], downIdx[1]]
        in_state_up_left = txState[upleftIdx[0], upleftIdx[1]]
        in_state_up_right = txState[uprightIdx[0], uprightIdx[1]]
        in_state_down_left = txState[downleftIdx[0], downleftIdx[1]]
        in_state_down_right = txState[downrightIdx[0], downrightIdx[1]]

        F_G_star_oldOldies = F_G_star_oldOldGradients[ix, iy]

        d_left = seaLevel - B_west
        d_right = seaLevel - B_east
        d_down = seaLevel - B_south
        d_up = seaLevel - B_north
        d_left_left = wp.max(0.0, seaLevel - txBottom[leftleftIdx[0], leftleftIdx[1]][2])
        d_right_right = wp.max(0.0, seaLevel - txBottom[rightrightIdx[0], rightrightIdx[1]][2])
        d_down_down = wp.max(0.0, seaLevel - txBottom[downdownIdx[0], downdownIdx[1]][2])
        d_up_up = wp.max(0.0, seaLevel - txBottom[upupIdx[0], upupIdx[1]][2])

        eta_here = in_state_here[0] - seaLevel
        eta_left = in_state_left[0] - seaLevel
        eta_right = in_state_right[0] - seaLevel
        eta_down = in_state_down[0] - seaLevel
        eta_up = in_state_up[0] - seaLevel
        eta_left_left = txState[leftleftIdx[0], leftleftIdx[1]][0] - seaLevel
        eta_right_right = txState[rightrightIdx[0], rightrightIdx[1]][0] - seaLevel
        eta_down_down = txState[downdownIdx[0], downdownIdx[1]][0] - seaLevel
        eta_up_up = txState[upupIdx[0], upupIdx[1]][0] - seaLevel
        eta_up_left = in_state_up_left[0] - seaLevel
        eta_up_right = in_state_up_right[0] - seaLevel
        eta_down_left = in_state_down_left[0] - seaLevel
        eta_down_right = in_state_down_right[0] - seaLevel

        v_up = in_state_up[2]
        v_down = in_state_down[2]
        v_right = in_state_right[2]
        v_left = in_state_left[2]
        v_up_right = in_state_up_right[2]
        v_down_right = in_state_down_right[2]
        v_up_left = in_state_up_left[2]
        v_down_left = in_state_down_left[2]

        u_up = in_state_up[1]
        u_down = in_state_down[1]
        u_right = in_state_right[1]
        u_left = in_state_left[1]
        u_up_right = in_state_up_right[1]
        u_down_right = in_state_down_right[1]
        u_up_left = in_state_up_left[1]
        u_down_left = in_state_down_left[1]

        dd_by_dx = (-d_right_right + 8.0 * d_right - 8.0 * d_left + d_left_left) * one_over_dx / 12.0
        dd_by_dy = (-d_up_up + 8.0 * d_up - 8.0 * d_down + d_down_down) * one_over_dy / 12.0
        eta_by_dx_dy = 0.25 * one_over_dx * one_over_dy * (eta_up_right - eta_down_right - eta_up_left + eta_down_left)
        eta_by_dx_dx = one_over_d2x * (eta_right - 2.0 * eta_here + eta_left)
        eta_by_dy_dy = one_over_d2y * (eta_up - 2.0 * eta_here + eta_down)

        F_star = (1.0 / 6.0) * d_here * (dd_by_dx * (0.5 * one_over_dy) * (v_up - v_down) + dd_by_dy * (0.5 * one_over_dx) * (v_right - v_left)) + (Bcoef + 1.0 / 3.0) * d2_here * (one_over_dxdy / 4.0) * (v_up_right - v_down_right - v_up_left + v_down_left)
        G_star = (1.0 / 6.0) * d_here * (dd_by_dx * (0.5 * one_over_dy) * (u_up - u_down) + dd_by_dy * (0.5 * one_over_dx) * (u_right - u_left)) + (Bcoef + 1.0 / 3.0) * d2_here * (one_over_dxdy / 4.0) * (u_up_right - u_down_right - u_up_left + u_down_left)
        
        Psi1x = Bcoef_g * d3_here * ((eta_right_right - 2.0 * eta_right + 2.0 * eta_left - eta_left_left) * (0.5 * one_over_d3x) + (eta_up_right - eta_up_left - 2.0 * eta_right + 2.0 * eta_left + eta_down_right - eta_down_left) * (0.5 * one_over_dx * one_over_d2y))
        Psi2x = Bcoef_g * d2_here * (dd_by_dx * (2.0 * eta_by_dx_dx + eta_by_dy_dy) + dd_by_dy * eta_by_dx_dy) + (F_star - F_G_star_oldOldies[0]) / dt / 2.0
        
        Psi1y = Bcoef_g * d3_here * ((eta_up_up - 2.0 * eta_up + 2.0 * eta_down - eta_down_down) * (0.5 * one_over_d3y) + (eta_up_right + eta_up_left - 2.0 * eta_up + 2.0 * eta_down - eta_down_right - eta_down_left) * (0.5 * one_over_dx * one_over_d2x))
        Psi2y = Bcoef_g * d2_here * (dd_by_dy * (2.0 * eta_by_dy_dy + eta_by_dx_dx) + dd_by_dx * eta_by_dx_dy) + (G_star - F_G_star_oldOldies[1]) / dt / 2.0

    friction_ = FrictionCalc(in_state_here[0], in_state_here[1], h_here, epsilon, g, friction, isManning)
    P_here = txShipPressure[ix, iy]
    P_left = txShipPressure[leftIdx[0], leftIdx[1]][0]
    P_right = txShipPressure[rightIdx[0], rightIdx[1]][0]
    P_down = txShipPressure[downIdx[0], downIdx[1]][0]
    P_up = txShipPressure[upIdx[0], upIdx[1]][0]

    press_x = -0.5 * h_here * g_over_dx * (P_right - P_left)
    press_y = -0.5 * h_here * g_over_dy * (P_up - P_down)

    C_state_here = txState[ix, iy][3]
    C_state_right = txState[rightIdx[0], rightIdx[1]][3]
    C_state_left = txState[leftIdx[0], leftIdx[1]][3]
    C_state_up = txState[upIdx[0], upIdx[1]][3]
    C_state_down = txState[downIdx[0], downIdx[1]][3]
    C_state_up_left = txState[upleftIdx[0], upleftIdx[1]][3]
    C_state_up_right = txState[uprightIdx[0], uprightIdx[1]][3]
    C_state_down_left = txState[downleftIdx[0], downleftIdx[1]][3]
    C_state_down_right = txState[downrightIdx[0], downrightIdx[1]][3]

    Dxx = 1.0
    Dxy = 1.0
    Dyy = 1.0
    
    hc_by_dx_dx = Dxx * one_over_d2x * (C_state_right - 2.0 * in_state_here[3] + C_state_left)
    hc_by_dy_dy = Dyy * one_over_d2y * (C_state_up - 2.0 * in_state_here[3] + C_state_down)
    hc_by_dx_dy = 0.25 * Dxy * one_over_dxdy * (C_state_up_right - C_state_up_left - C_state_down_right + C_state_down_left)
    c_dissipation = -0.1 * C_state_here
    source_term = wp.vec4(0.0, -g * h_here * detadx - in_state_here[1] * friction_ + (Psi1x + Psi2x) + press_x, -g * h_here * detady - in_state_here[2] * friction_ + (Psi1y + Psi2y) + press_y, hc_by_dx_dx + hc_by_dy_dy + 2.0 * hc_by_dx_dy + c_dissipation)
    d_by_dt = (xflux_west - xflux_here) * one_over_dx + (yflux_south - yflux_here) * one_over_dy + source_term
    newState = wp.vec4(0.0, 0.0, 0.0, 0.0)
    in_state_here_UV = txstateUVstar[ix, iy]

    if timeScheme == 0:
        newState = in_state_here_UV + dt * d_by_dt
    elif pred_or_corrector == 1:
        newState = in_state_here_UV + dt / 12.0 * (23.0 * d_by_dt - 16.0 * oldies + 5.0 * oldOldies)
    elif pred_or_corrector == 2:
        predicted = predictedGradients[ix, iy]
        newState = in_state_here_UV + dt / 24.0 * (9.0 * d_by_dt + 19.0 * predicted - 5.0 * oldies + oldOldies)

    if wp.abs(P_here[0]) > 1.0:
        newState[3] = 1.0 * wp.min(1.0, P_here[3] / 5.0)

    txNewState[ix, iy] = newState
    dU_by_dt[ix, iy] = d_by_dt
    F_G_star[ix, iy] = wp.vec4(F_star, G_star, 0.0, 1.0)
    current_stateUVstar[ix, iy] = newState
