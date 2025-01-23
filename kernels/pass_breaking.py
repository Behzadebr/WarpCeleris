import warp as wp

@wp.kernel
def Pass_Breaking(
    State: wp.array2d(dtype=wp.vec4),
    XFlux: wp.array2d(dtype=wp.vec4),
    YFlux: wp.array2d(dtype=wp.vec4),
    Breaking: wp.array2d(dtype=wp.vec4),
    DissipationFlux: wp.array2d(dtype=wp.vec4),
    Bottom: wp.array2d(dtype=wp.vec4),
    dU_by_dt: wp.array2d(dtype=wp.vec4),
    time: wp.float32,
    width: wp.int32,
    height: wp.int32,
    dt: wp.float32,
    dx: wp.float32,
    dy: wp.float32,
    one_over_dx: wp.float32,
    one_over_dy: wp.float32,
    T_star_coef: wp.float32,
    dzdt_I_coef: wp.float32,
    dzdt_F_coef: wp.float32,
    delta_breaking: wp.float32,
    g: wp.float32,
    epsilon: wp.float32
):
    """
    Warp kernel to perform Pass_Breaking computations: Calculate dissipation flux and update breaking parameters.
    
    """
    ix, iy = wp.tid()

    # Ensure thread is within bounds
    if ix >= width or iy >= height:
        return

    # Compute neighbor indices
    rightIdx = wp.min(ix + 1, width - 1)
    upIdx = wp.min(iy + 1, height - 1)
    leftIdx = wp.max(ix - 1, 0)
    downIdx = wp.max(iy - 1, 0)

    # Load fluxes
    xflux_here = XFlux[ix, iy].x
    xflux_west = XFlux[leftIdx, iy].x

    yflux_here = YFlux[ix, iy].x
    yflux_south = YFlux[ix, downIdx].x

    P_south = State[ix, downIdx].y
    P_here = State[ix, iy].y
    P_north = State[ix, upIdx].y

    Q_west = State[leftIdx, iy].z
    Q_here = State[ix, iy].z
    Q_east = State[rightIdx, iy].z

    detadt = dU_by_dt[ix, iy].x

    # Determine dominant direction of flow
    t_here = Breaking[ix, iy].x
    t1 = 0.0
    t2 = 0.0
    t3 = 0.0

    if wp.abs(P_here) > wp.abs(Q_here):
        if P_here > 0.0:
            t1 = Breaking[leftIdx, iy].x
            t2 = Breaking[leftIdx, upIdx].x
            t3 = Breaking[leftIdx, downIdx].x
        else:
            t1 = Breaking[rightIdx, iy].x
            t2 = Breaking[rightIdx, upIdx].x
            t3 = Breaking[rightIdx, downIdx].x
    else:
        if Q_here > 0.0:
            t1 = Breaking[ix, downIdx].x
            t2 = Breaking[upIdx, downIdx].x
            t3 = Breaking[leftIdx, downIdx].x
        else:
            t1 = Breaking[ix, upIdx].x
            t2 = Breaking[rightIdx, upIdx].x
            t3 = Breaking[leftIdx, upIdx].x

    t_here = wp.max(t_here, wp.max(t1, wp.max(t2, t3)))

    dPdx = (xflux_here - xflux_west) * one_over_dx
    dPdy = 0.5 * (P_north - P_south) * one_over_dy

    dQdx = 0.5 * (Q_east - Q_west) * one_over_dx
    dQdy = (yflux_here - yflux_south) * one_over_dy

    B_here = Bottom[ix, iy].z
    eta_here = State[ix, iy].x
    h_here = eta_here - B_here
    c_here = wp.sqrt(g * h_here)

    h2 = h_here * h_here
    divide_by_h = 2.0 * h_here / (h2 + wp.max(h2, epsilon))

    # Kennedy et al breaking model calculations
    T_star = T_star_coef * wp.sqrt(h_here / g)
    dzdt_I = dzdt_I_coef * c_here
    dzdt_F = dzdt_F_coef * c_here

    dzdt_star = 0.0
    if t_here <= dt:
        dzdt_star = dzdt_I
    elif (time - t_here) <= T_star:
        dzdt_star = dzdt_I + (time - t_here) / T_star * (dzdt_F - dzdt_I)
    else:
        dzdt_star = dzdt_F

    # Compute B_Breaking and update t_here based on detadt and dzdt_star
    B_Breaking = 0.0
    if detadt < dzdt_star:
        t_here = 0.0
    elif detadt > 2.0 * dzdt_star:
        B_Breaking = 1.0
        if t_here <= dt:
            t_here = time
    else:
        B_Breaking = detadt / dzdt_star - 1.0
        if t_here <= dt:
            t_here = time

    nu_breaking = wp.min(1.0 * dx * dy / dt, B_Breaking * delta_breaking * h_here * detadt)

    # Compute Smagorinsky subgrid eddy viscosity
    Smag_cm = 0.04
    strain_rate_squared = 2.0 * dPdx * dPdx + 2.0 * dQdy * dQdy + (dPdy + dQdx) * (dPdy + dQdx)
    nu_Smag = Smag_cm * dx * dy * wp.sqrt(strain_rate_squared) * divide_by_h

    nu_total = nu_breaking + nu_Smag

    nu_dPdx = nu_total * dPdx
    nu_dPdy = nu_total * dPdy

    nu_dQdx = nu_total * dQdx
    nu_dQdy = nu_total * dQdy

    nu_flux = wp.vec4(nu_dPdx, nu_dPdy, nu_dQdx, nu_dQdy)

    Bvalues = wp.vec4(t_here, nu_breaking, B_Breaking, nu_Smag)

    DissipationFlux[ix, iy] = nu_flux
    Breaking[ix, iy] = Bvalues