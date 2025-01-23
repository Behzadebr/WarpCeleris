import warp as wp

@wp.kernel
def TriDiag_PCRy(
    coefMaty: wp.array2d(dtype=wp.vec4),
    current_stateUVstar: wp.array2d(dtype=wp.vec4),
    current_buffer: wp.array2d(dtype=wp.vec4),
    next_buffer: wp.array2d(dtype=wp.vec4),
    temp2_PCRy: wp.array2d(dtype=wp.vec4),
    NewState: wp.array2d(dtype=wp.vec4),
    p: wp.int32,
    s: wp.int32,
    width: wp.int32,
    height: wp.int32,
):
    """
    Warp kernel to perform Tri-Diagonal PCRy computations.
    
    """
    ix, iy = wp.tid()

    # Ensure thread is within bounds
    if ix >= width or iy >= height:
        return
    
    # Retrieve the current state
    CurrentState = NewState[ix, iy]

    # Compute periodic neighbor indices with wrapping in y-direction
    idx_down = wp.mod(iy - s + height, height)
    idx_up = wp.mod(iy + s + height, height)

    aIn, bIn, cIn, dIn = 0.0, 0.0, 0.0, 0.0
    aInDown, bInDown, cInDown, dInDown = 0.0, 0.0, 0.0, 0.0
    aInUp, bInUp, cInUp, dInUp = 0.0, 0.0, 0.0, 0.0

    if p == 0:
        bIn = coefMaty[ix, iy].y
        bInDown = coefMaty[ix, idx_down].y
        bInUp = coefMaty[ix, idx_up].y

        aIn = coefMaty[ix, iy].x / bIn
        aInDown = coefMaty[ix, idx_down].x / bInDown
        aInUp = coefMaty[ix, idx_up].x / bInUp

        cIn = coefMaty[ix, iy].z / bIn
        cInDown = coefMaty[ix, idx_down].z / bInDown
        cInUp = coefMaty[ix, idx_up].z / bInUp

        dIn = current_stateUVstar[ix, iy].z / bIn
        dInDown = current_stateUVstar[ix, idx_down].z / bInDown
        dInUp = current_stateUVstar[ix, idx_up].z / bInUp
    else:
        aIn = current_buffer[ix, iy].x
        aInDown = current_buffer[ix, idx_down].x
        aInUp = current_buffer[ix, idx_up].x

        cIn = current_buffer[ix, iy].z
        cInDown = current_buffer[ix, idx_down].z
        cInUp = current_buffer[ix, idx_up].z

        dIn = current_buffer[ix, iy].w
        dInDown = current_buffer[ix, idx_down].w
        dInUp = current_buffer[ix, idx_up].w

    r = 1.0 / (1.0 - aIn * cInDown - cIn * aInUp)

    aOut = -r * aIn * aInDown
    cOut = -r * cIn * cInUp
    dOut = r * (dIn - aIn * dInDown - cIn * dInUp)

    next_buffer[ix, iy] = wp.vec4(aOut, 1.0, cOut, dOut)
    temp2_PCRy[ix, iy] = wp.vec4(CurrentState.x,CurrentState.y, dOut, CurrentState.w)
