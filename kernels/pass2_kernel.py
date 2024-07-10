import warp as wp

# Custom square root function for vec2
@wp.func
def vec2_sqrt(v: wp.vec2):
    return wp.vec2(wp.sqrt(v[0]), wp.sqrt(v[1]))

# NumericalFlux function
@wp.func
def numerical_flux(aplus: wp.float32, aminus: wp.float32, Fplus: wp.float32, Fminus: wp.float32, Udifference: wp.float32):
    if aplus - aminus != 0.0:
        return (aplus * Fminus - aminus * Fplus + aplus * aminus * Udifference) / (aplus - aminus)
    else:
        return 0.0

# ScalarAntiDissipation function
@wp.func
def scalar_anti_dissipation(uplus: wp.float32, uminus: wp.float32, aplus: wp.float32, aminus: wp.float32, epsilon: wp.float32):
    Fr = wp.float32(0.0)
    if aplus != 0.0 and aminus != 0.0:
        if wp.abs(uplus) >= wp.abs(uminus):
            Fr = wp.abs(uplus) / aplus
        else:
            Fr = wp.abs(uminus) / aminus
        phi = (Fr + epsilon) / (Fr + 1.0)
    elif aplus == 0.0 or aminus == 0.0:
        phi = epsilon
    return phi

# Pass2 Kernel
@wp.kernel
def pass2_kernel(txH: wp.array2d(dtype=wp.vec4), txU: wp.array2d(dtype=wp.vec4), txV: wp.array2d(dtype=wp.vec4), txBottom: wp.array2d(dtype=wp.vec4), txC: wp.array2d(dtype=wp.vec4),
                 txXFlux: wp.array2d(dtype=wp.vec4), txYFlux: wp.array2d(dtype=wp.vec4),
                 width: int, height: int, g: wp.float32, half_g: wp.float32, dx: wp.float32, dy: wp.float32):

    ix, iy = wp.tid()  # 2D thread index

    leftIdx_x = wp.max(ix - 1, 0)
    leftIdx_y = wp.max(iy, 0)
    rightIdx_x = wp.min(ix + 1, width - 1)
    rightIdx_y = wp.min(iy, height - 1)
    upIdx_x = wp.max(ix, 0)
    upIdx_y = wp.max(iy - 1, 0)
    downIdx_x = wp.min(ix, width - 1)
    downIdx_y = wp.min(iy + 1, height - 1)

    h_vec = txH[ix, iy]
    h_here = wp.vec2(h_vec[0], h_vec[1])

    hW_east = txH[rightIdx_x, rightIdx_y][3]
    hS_north = txH[upIdx_x, upIdx_y][2]

    u_here = wp.vec2(txU[ix, iy][0], txU[ix, iy][1])
    uW_east = txU[rightIdx_x, rightIdx_y][3]
    uS_north = txU[upIdx_x, upIdx_y][2]

    v_here = wp.vec2(txV[ix, iy][0], txV[ix, iy][1])
    vW_east = txV[rightIdx_x, rightIdx_y][3]
    vS_north = txV[upIdx_x, upIdx_y][2]

    cNE = vec2_sqrt(wp.vec2(g * h_here[0], g * h_here[1]))
    cW = wp.sqrt(g * hW_east)
    cS = wp.sqrt(g * hS_north)

    aplus = wp.max(wp.max(u_here[1] + cNE[1], uW_east + cW), 0.0)
    aminus = wp.min(wp.min(u_here[1] - cNE[1], uW_east - cW), 0.0)
    bplus = wp.max(wp.max(v_here[0] + cNE[0], vS_north + cS), 0.0)
    bminus = wp.min(wp.min(v_here[0] - cNE[0], vS_north - cS), 0.0)

    B_here = txBottom[ix, iy][2]
    B_south = txBottom[downIdx_x, downIdx_y][2]
    B_north = txBottom[upIdx_x, upIdx_y][2]
    B_west = txBottom[leftIdx_x, leftIdx_y][2]
    B_east = txBottom[rightIdx_x, rightIdx_y][2]
    dB = wp.max(wp.abs(B_south - B_here), wp.max(wp.abs(B_north - B_here), wp.max(wp.abs(B_west - B_here), wp.abs(B_east - B_here))))

    near_dry = txBottom[ix, iy][3]

    c_here = wp.vec2(txC[ix, iy][0], txC[ix, iy][1])
    cW_east = txC[rightIdx_x, rightIdx_y][3]
    cS_north = txC[upIdx_x, upIdx_y][2]

    phix = wp.float32(0.5)
    phiy = wp.float32(0.5)

    minH = wp.min(h_vec)
    mass_diff_x = (hW_east - h_here[1])
    mass_diff_y = (hS_north - h_here[0])

    if (minH * minH) <= 3.0 * dx * dB:
        mass_diff_x = 0.0
        mass_diff_y = 0.0
        phix = 1.0
        phiy = 1.0

    xflux = wp.vec4(
        numerical_flux(aplus, aminus, hW_east * uW_east, h_here[1] * u_here[1], mass_diff_x),
        numerical_flux(aplus, aminus, hW_east * uW_east * uW_east, h_here[1] * u_here[1] * u_here[1], hW_east * uW_east - h_here[1] * u_here[1]),
        numerical_flux(aplus, aminus, hW_east * uW_east * vW_east, h_here[1] * u_here[1] * v_here[1], hW_east * vW_east - h_here[1] * v_here[1]),
        numerical_flux(aplus, aminus, hW_east * uW_east * cW_east, h_here[1] * u_here[1] * c_here[1], phix * (hW_east * cW_east - h_here[1] * c_here[1]))
    )

    yflux = wp.vec4(
        numerical_flux(bplus, bminus, hS_north * vS_north, h_here[0] * v_here[0], mass_diff_y),
        numerical_flux(bplus, bminus, hS_north * uS_north * vS_north, h_here[0] * u_here[0] * v_here[0], hS_north * uS_north - h_here[0] * u_here[0]),
        numerical_flux(bplus, bminus, hS_north * vS_north * vS_north, h_here[0] * v_here[0] * v_here[0], hS_north * vS_north - h_here[0] * v_here[0]),
        numerical_flux(bplus, bminus, hS_north * cS_north * vS_north, h_here[0] * c_here[0] * v_here[0], phiy * (hS_north * cS_north - h_here[0] * c_here[0]))
    )

    txXFlux[ix, iy] = xflux
    txYFlux[ix, iy] = yflux

