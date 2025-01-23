import warp as wp

@wp.kernel
def Update_Bottom(
    Bottom: wp.array2d(dtype=wp.vec4),
    erosion_Sed: wp.array2d(dtype=wp.float32),
    deposition_Sed: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    sedC1_n: wp.float32,
    width: wp.int32,
    height: wp.int32
) -> None:

    ix, iy = wp.tid()

    # Boundary check to prevent out-of-bounds access
    if ix >= width or iy >= height:
        return

    B = Bottom[ix, iy].z

    e = erosion_Sed[ix, iy]
    d = deposition_Sed[ix, iy]

    delta_B = dt * (e - d) / (1.0 - sedC1_n)

    Bottom[ix, iy].z = B + delta_B