import warp as wp

@wp.kernel
def Fill_Bottom(
    Bottom: wp.array2d(dtype=wp.vec4),
    width: wp.int32,
    height: wp.int32,
    lengthCheck: wp.int32
):
    """
    Warp kernel to fill the Bottom array.
    """
    i, j = wp.tid()

    if i >= width or j >= height:
        return

    bn = 0.5 * Bottom[i, j].z
    if j + 1 < height:
        bn += 0.5 * Bottom[i, j + 1].z
    else:
        bn += 0.5 * Bottom[i, j].z
    Bottom[i, j].x = bn

    be = 0.5 * Bottom[i, j].z
    if i + 1 < width:
        be += 0.5 * Bottom[i + 1, j].z
    else:
        be += 0.5 * Bottom[i, j].z
    Bottom[i, j].y = be

    Bottom[i, j].w = 99.0

    # Check within the window for bathy >= 0 to set near_dry flag
    for dy in range(-lengthCheck, lengthCheck + 1):
        for dx in range(-lengthCheck, lengthCheck + 1):
            xC = wp.max(0, wp.min(width - 1, i + dx))
            yC = wp.max(0, wp.min(height - 1, j + dy))
            if Bottom[xC, yC].z >= 0.0:
                Bottom[i, j].w = -99.0
                # Early exit
                break
        if Bottom[i, j].w == -99.0:
            break
