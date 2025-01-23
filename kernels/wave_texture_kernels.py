import warp as wp

@wp.kernel
def populate_wave_texture(
    width: int,
    height: int,
    wave_data: wp.array(dtype=wp.vec4, ndim=2),
    out_tex:  wp.array(dtype=wp.vec3, ndim=2)
):
    """
    Convert wave_data (wp.vec4) into an RGB float texture for visualization.
    Use wave_data[x,y].x into grayscale for now for debug.
    """

    x, y = wp.tid()

    float_min = -1.0
    float_max =  1.0

    wave4 = wave_data[x, y]
    wave_h = wave4[0]  # 'r' channel from wave_data[x,y]

    val = (wave_h - float_min) / (float_max - float_min)
    val = wp.clamp(val, 0.0, 1.0)

    out_tex[y, x] = wp.vec3(val, val, val)