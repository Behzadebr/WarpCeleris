import warp as wp

@wp.kernel
def tridiag_coeffs_Y(
    Bottom: wp.array2d(dtype=wp.vec4),
    coefMaty: wp.array2d(dtype=wp.vec4),
    dy: wp.float32,
    Bcoef: wp.float32,
    width: wp.int32,
    height: wp.int32,
):

    ix, iy = wp.tid()

    # Ensure thread indices are within bounds
    if ix >= width or iy >= height:
        return

    neardry = Bottom[ix, iy].w

    if iy <= 2 or iy >= (height - 3) or neardry < 0.0:
        a = 0.0
        b = 1.0
        c = 0.0
    else:
        depth_here = -Bottom[ix, iy].z
        depth_plus = -Bottom[ix, iy + 1].z
        depth_minus = -Bottom[ix, iy - 1].z

        d_dy = (depth_plus - depth_minus) / (2.0 * dy)

        a = (depth_here * d_dy) / (6.0 * dy) - (Bcoef + (1.0 / 3.0)) * (wp.pow(depth_here, 2.0)) / (wp.pow(dy, 2.0))
        b = 1.0 + 2.0 * (Bcoef + (1.0 / 3.0)) * (wp.pow(depth_here, 2.0)) / (wp.pow(dy, 2.0))
        c = -(depth_here * d_dy) / (6.0 * dy) - (Bcoef + (1.0 / 3.0)) * (wp.pow(depth_here, 2.0)) / (wp.pow(dy, 2.0))

    coefMaty[ix, iy] = wp.vec4(a, b, c, 0.0)