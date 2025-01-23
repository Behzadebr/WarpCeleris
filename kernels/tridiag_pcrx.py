import warp as wp

@wp.kernel
def TriDiag_PCRx(
    coefMatx: wp.array2d(dtype=wp.vec4),
    current_stateUVstar: wp.array2d(dtype=wp.vec4),
    current_buffer: wp.array2d(dtype=wp.vec4),
    next_buffer: wp.array2d(dtype=wp.vec4),
    temp2_PCRx: wp.array2d(dtype=wp.vec4),
    NewState: wp.array2d(dtype=wp.vec4),
    p: wp.int32,
    s: wp.int32,
    width: wp.int32,
    height: wp.int32
):
    """
    Warp kernel to perform Tri-Diagonal PCRx computations.
    
    """
    ix, iy = wp.tid()

    # Ensure thread is within bounds
    if ix >= width or iy >= height:
        return

    # Retrieve the current state
    CurrentState = NewState[ix, iy]

    # Compute periodic neighbor indices with wrapping
    idx_left = wp.mod(ix - s + width, width)
    idx_right = wp.mod(ix + s + width, width)

    # Initialize coefficients
    aIn, bIn, cIn, dIn = 0.0, 0.0, 0.0, 0.0
    aInLeft, bInLeft, cInLeft, dInLeft = 0.0, 0.0, 0.0, 0.0
    aInRight, bInRight, cInRight, dInRight = 0.0, 0.0, 0.0, 0.0

    if p == 0:
        bIn = coefMatx[ix, iy].y
        bInLeft = coefMatx[idx_left, iy].y
        bInRight = coefMatx[idx_right, iy].y

        aIn = coefMatx[ix, iy].x / bIn
        aInLeft = coefMatx[idx_left, iy].x / bInLeft
        aInRight = coefMatx[idx_right, iy].x / bInRight

        cIn = coefMatx[ix, iy].z / bIn
        cInLeft = coefMatx[idx_left, iy].z / bInLeft
        cInRight = coefMatx[idx_right, iy].z / bInRight

        dIn = current_stateUVstar[ix, iy].y / bIn
        dInLeft = current_stateUVstar[idx_left, iy].y / bInLeft
        dInRight = current_stateUVstar[idx_right, iy].y / bInRight
    else:
        aIn = current_buffer[ix, iy].x
        aInLeft = current_buffer[idx_left, iy].x
        aInRight = current_buffer[idx_right, iy].x

        cIn = current_buffer[ix, iy].z
        cInLeft = current_buffer[idx_left, iy].z
        cInRight = current_buffer[idx_right, iy].z

        dIn = current_buffer[ix, iy].w
        dInLeft = current_buffer[idx_left, iy].w
        dInRight = current_buffer[idx_right, iy].w

    r = 1.0 / (1.0 - aIn * cInLeft - cIn * aInRight)

    aOut = -r * aIn * aInLeft
    cOut = -r * cIn * cInRight
    dOut = r * (dIn - aIn * dInLeft - cIn * dInRight)

    next_buffer[ix, iy] = wp.vec4(aOut, 1.0, cOut, dOut)
    temp2_PCRx[ix, iy] = wp.vec4(CurrentState.x, dOut, CurrentState.z, CurrentState.w)
