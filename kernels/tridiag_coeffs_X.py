import warp as wp

@wp.kernel
def tridiag_coeffs_X(
    Bottom: wp.array2d(dtype=wp.vec4),
    coefMatx: wp.array2d(dtype=wp.vec4),
    dx: wp.float32,
    Bcoef: wp.float32,
    width: wp.int32,
    height: wp.int32,
):

    ix, iy = wp.tid()

    # Ensure thread indices are within bounds
    if ix >= width or iy >= height:
        return

    neardry = Bottom[ix, iy].w

    if ix <= 2 or ix >= (width - 3) or neardry < 0.0:
        a = 0.0
        b = 1.0
        c = 0.0
    else:
        depth_here = -Bottom[ix, iy].z
        depth_plus = -Bottom[ix + 1, iy].z
        depth_minus = -Bottom[ix - 1, iy].z

        d_dx = (depth_plus - depth_minus) / (2.0 * dx)

        a = (depth_here * d_dx) / (6.0 * dx) - (Bcoef + (1.0 / 3.0)) * (wp.pow(depth_here, 2.0)) / (wp.pow(dx, 2.0))
        b = 1.0 + 2.0 * (Bcoef + (1.0 / 3.0)) * (wp.pow(depth_here, 2.0)) / (wp.pow(dx, 2.0))
        c = -(depth_here * d_dx) / (6.0 * dx) - (Bcoef + (1.0 / 3.0)) * (wp.pow(depth_here, 2.0)) / (wp.pow(dx, 2.0))

    coefMatx[ix, iy] = wp.vec4(a, b, c, 0.0)