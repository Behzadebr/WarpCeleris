import warp as wp

# TriDiag_PCRy Kernel
@wp.kernel
def tridiag_pcry_kernel(coefMaty: wp.array2d(dtype=wp.vec4), current_state: wp.array2d(dtype=wp.vec4), current_stateUVstar: wp.array2d(dtype=wp.vec4),
                        newcoefy: wp.array2d(dtype=wp.vec4), txNewState: wp.array2d(dtype=wp.vec4), width: wp.int32, height: wp.int32, p: wp.int32, s: wp.int32):
    
    ix, iy = wp.tid()  # 2D thread index
    idx = wp.vec2i(ix, iy)

    CurrentState = current_state[ix, iy]
    
    # Initialize state
    newcoefy[ix, iy] = wp.vec4(0.0, 0.0, 0.0, 0.0)
    txNewState[ix, iy] = CurrentState
    
    # Return if in imaginary grid points
    if iy >= width-2 or ix >= height-2 or iy <= 1 or ix <= 1:
        return

    idx_left = wp.vec2i(ix, (iy + s + height) % height)
    idx_right = wp.vec2i(ix, (iy - s + height) % height)

    bIn, bInLeft, bInRight = 0.0, 0.0, 0.0
    aIn, aInLeft, aInRight = 0.0, 0.0, 0.0
    cIn, cInLeft, cInRight = 0.0, 0.0, 0.0
    dIn, dInLeft, dInRight = 0.0, 0.0, 0.0

    if p == 0:
        bIn = coefMaty[ix, iy][1]
        bInLeft = coefMaty[idx_left[0], idx_left[1]][1]
        bInRight = coefMaty[idx_right[0], idx_right[1]][1]

        aIn = coefMaty[ix, iy][0] / bIn
        aInLeft = coefMaty[idx_left[0], idx_left[1]][0] / bInLeft
        aInRight = coefMaty[idx_right[0], idx_right[1]][0] / bInRight

        cIn = coefMaty[ix, iy][2] / bIn
        cInLeft = coefMaty[idx_left[0], idx_left[1]][2] / bInLeft
        cInRight = coefMaty[idx_right[0], idx_right[1]][2] / bInRight

        dIn = current_stateUVstar[ix, iy][2] / bIn
        dInLeft = current_stateUVstar[idx_left[0], idx_left[1]][2] / bInLeft
        dInRight = current_stateUVstar[idx_right[0], idx_right[1]][2] / bInRight
    else:
        aIn = coefMaty[ix, iy][0]
        aInLeft = coefMaty[idx_left[0], idx_left[1]][0]
        aInRight = coefMaty[idx_right[0], idx_right[1]][0]

        cIn = coefMaty[ix, iy][2]
        cInLeft = coefMaty[idx_left[0], idx_left[1]][2]
        cInRight = coefMaty[idx_right[0], idx_right[1]][2]

        dIn = coefMaty[ix, iy][3]
        dInLeft = coefMaty[idx_left[0], idx_left[1]][3]
        dInRight = coefMaty[idx_right[0], idx_right[1]][3]

    r = 1.0 / (1.0 - aIn * cInLeft - cIn * aInRight)
    aOut = -r * aIn * aInLeft
    cOut = -r * cIn * cInRight
    dOut = r * (dIn - aIn * dInLeft - cIn * dInRight)

    newcoefy[ix, iy] = wp.vec4(aOut, 1.0, cOut, dOut)
    txNewState[ix, iy] = wp.vec4(CurrentState[0], CurrentState[1], dOut, CurrentState[3])
