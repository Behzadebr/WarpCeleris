import warp as wp

# TriDiag_PCRx Kernel
@wp.kernel
def tridiag_pcrx_kernel(coefMatx: wp.array2d(dtype=wp.vec4), current_state: wp.array2d(dtype=wp.vec4), current_stateUVstar: wp.array2d(dtype=wp.vec4),
                        newcoefx: wp.array2d(dtype=wp.vec4), txNewState: wp.array2d(dtype=wp.vec4), width: wp.int32, height: wp.int32, p: wp.int32, s: wp.int32):
    
    ix, iy = wp.tid()  # 2D thread index
    idx = wp.vec2i(ix, iy)

    CurrentState = current_state[ix, iy]
    
    # Initialize state
    newcoefx[ix, iy] = wp.vec4(0.0, 0.0, 0.0, 0.0)
    txNewState[ix, iy] = CurrentState
    
    # Return if in imaginary grid points
    if ix >= width-2 or iy >= height-2 or ix <= 1 or iy <= 1:
        return

    idx_left = wp.vec2i((ix - s + width) % width, iy)
    idx_right = wp.vec2i((ix + s + width) % width, iy)

    bIn, bInLeft, bInRight = 0.0, 0.0, 0.0
    aIn, aInLeft, aInRight = 0.0, 0.0, 0.0
    cIn, cInLeft, cInRight = 0.0, 0.0, 0.0
    dIn, dInLeft, dInRight = 0.0, 0.0, 0.0

    if p == 0:
        bIn = coefMatx[ix, iy][1]
        bInLeft = coefMatx[idx_left[0], idx_left[1]][1]
        bInRight = coefMatx[idx_right[0], idx_right[1]][1]

        aIn = coefMatx[ix, iy][0] / bIn
        aInLeft = coefMatx[idx_left[0], idx_left[1]][0] / bInLeft
        aInRight = coefMatx[idx_right[0], idx_right[1]][0] / bInRight

        cIn = coefMatx[ix, iy][2] / bIn
        cInLeft = coefMatx[idx_left[0], idx_left[1]][2] / bInLeft
        cInRight = coefMatx[idx_right[0], idx_right[1]][2] / bInRight

        dIn = current_stateUVstar[ix, iy][1] / bIn
        dInLeft = current_stateUVstar[idx_left[0], idx_left[1]][1] / bInLeft
        dInRight = current_stateUVstar[idx_right[0], idx_right[1]][1] / bInRight
    else:
        aIn = coefMatx[ix, iy][0]
        aInLeft = coefMatx[idx_left[0], idx_left[1]][0]
        aInRight = coefMatx[idx_right[0], idx_right[1]][0]

        cIn = coefMatx[ix, iy][2]
        cInLeft = coefMatx[idx_left[0], idx_left[1]][2]
        cInRight = coefMatx[idx_right[0], idx_right[1]][2]

        dIn = coefMatx[ix, iy][3]
        dInLeft = coefMatx[idx_left[0], idx_left[1]][3]
        dInRight = coefMatx[idx_right[0], idx_right[1]][3]

    r = 1.0 / (1.0 - aIn * cInLeft - cIn * aInRight)
    aOut = -r * aIn * aInLeft
    cOut = -r * cIn * cInRight
    dOut = r * (dIn - aIn * dInLeft - cIn * dInRight)

    newcoefx[ix, iy] = wp.vec4(aOut, 1.0, cOut, dOut)
    txNewState[ix, iy] = wp.vec4(CurrentState[0], dOut, CurrentState[2], CurrentState[3])
