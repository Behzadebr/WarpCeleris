import warp as wp

# Function to Compute the wavenumber using Eckart's approximation
@wp.func
def calc_wavenumber_approx(omega: wp.float32, d: wp.float32, boundary_g: wp.float32):
    # Compute the wavenumber using Eckart's approximation
    return omega * omega / (boundary_g * wp.sqrt(wp.tanh(omega * omega * d / boundary_g)))

# Function to compute sine wave
@wp.func
def sine_wave(x: wp.float32, y: wp.float32, t: wp.float32, d: wp.float32, amplitude: wp.float32, period: wp.float32, theta: wp.float32, phase: wp.float32, boundary_g: wp.float32, PI: wp.float32):
    omega = 2.0 * PI / period
    k = calc_wavenumber_approx(omega, d, boundary_g)
    c = omega / k
    kx = wp.cos(theta) * x * k
    ky = wp.sin(theta) * y * k
    eta = amplitude * wp.sin(omega * t - kx - ky + phase) * wp.min(1.0, t / period)
    speed = boundary_g * eta / (c * k) * wp.tanh(k * d)
    hu = speed * wp.cos(theta)
    hv = speed * wp.sin(theta)
    return wp.vec3(eta, hu, hv)

# Functions for boundary conditions
@wp.func
def west_boundary_solid(idx: wp.vec2i, reflect_x: wp.int32, txState: wp.array2d(dtype=wp.vec4)):
    shift = 4
    real_idx = wp.vec2i(shift - idx[0], idx[1])
    in_state_real = txState[real_idx[0], real_idx[1]]
    return wp.vec4(in_state_real[0], -in_state_real[1], in_state_real[2], in_state_real[3])

@wp.func
def east_boundary_solid(idx: wp.vec2i, reflect_x: wp.int32, txState: wp.array2d(dtype=wp.vec4)):
    real_idx = wp.vec2i(reflect_x - idx[0], idx[1])
    in_state_real = txState[real_idx[0], real_idx[1]]
    return wp.vec4(in_state_real[0], -in_state_real[1], in_state_real[2], in_state_real[3])


@wp.func
def south_boundary_solid(idx: wp.vec2i, reflect_y: wp.int32, txState: wp.array2d(dtype=wp.vec4)):
    real_idx = wp.vec2i(idx[0], reflect_y - idx[1])
    in_state_real = txState[real_idx[0], real_idx[1]]
    return wp.vec4(in_state_real[0], in_state_real[1], -in_state_real[2], in_state_real[3])

@wp.func
def north_boundary_solid(idx: wp.vec2i, reflect_y: wp.int32, txState: wp.array2d(dtype=wp.vec4)):
    shift = 4
    real_idx = wp.vec2i(idx[0], shift - idx[1])
    in_state_real = txState[real_idx[0], real_idx[1]]
    return wp.vec4(in_state_real[0], in_state_real[1], -in_state_real[2], in_state_real[3])

@wp.func
def west_boundary_sponge(idx: wp.vec2i, PI: wp.float32, BoundaryWidth: wp.float32, txState: wp.array2d(dtype=wp.vec4)):
    gamma = wp.pow(0.5 * (0.5 + 0.5 * wp.cos(PI * (BoundaryWidth - float(idx[0]) + 2.0) / (BoundaryWidth - 1.0))), 0.01)
    new_state = txState[idx[0], idx[1]]
    return wp.vec4(gamma * new_state[0], gamma * new_state[1], gamma * new_state[2], gamma * new_state[3])

@wp.func
def east_boundary_sponge(idx: wp.vec2i, PI: wp.float32, BoundaryWidth: wp.float32, boundary_nx: wp.int32, txState: wp.array2d(dtype=wp.vec4)):
    gamma = wp.pow(0.5 * (0.5 + 0.5 * wp.cos(PI * (BoundaryWidth - float(boundary_nx - idx[0])) / (BoundaryWidth - 1.0))), 0.01)
    new_state = txState[idx[0], idx[1]]
    return wp.vec4(gamma * new_state[0], gamma * new_state[1], gamma * new_state[2], gamma * new_state[3])

@wp.func
def south_boundary_sponge(idx: wp.vec2i, PI: wp.float32, BoundaryWidth: wp.float32, boundary_ny: wp.int32, txState: wp.array2d(dtype=wp.vec4)):
    gamma = wp.pow(0.5 * (0.5 + 0.5 * wp.cos(PI * (BoundaryWidth - float(boundary_ny - idx[1])) / (BoundaryWidth - 1.0))), 0.01)
    new_state = txState[idx[0], idx[1]]
    return wp.vec4(new_state[0], gamma * new_state[1], gamma * new_state[2], gamma * new_state[3])

@wp.func
def north_boundary_sponge(idx: wp.vec2i, PI: wp.float32, BoundaryWidth: wp.float32, txState: wp.array2d(dtype=wp.vec4)):
    gamma = wp.pow(0.5 * (0.5 + 0.5 * wp.cos(PI * (BoundaryWidth - float(idx[1]) + 2.0) / (BoundaryWidth - 1.0))), 0.01)
    new_state = txState[idx[0], idx[1]]
    return wp.vec4(new_state[0], gamma * new_state[1], gamma * new_state[2], gamma * new_state[3])

@wp.func
def boundary_sine_wave(idx: wp.vec2i, txBottom: wp.array2d(dtype=wp.vec4), txWaves: wp.array2d(dtype=wp.vec4), dx: wp.float32, dy: wp.float32, seaLevel: wp.float32, total_time: wp.float32, numberOfWaves: wp.int32, boundary_g: wp.float32, PI: wp.float32):
    B_here = txBottom[idx[0], idx[1]][2]
    d_here = wp.max(0.0, seaLevel - B_here)
    x = float(idx[0]) * dx
    y = float(idx[1]) * dy

    result = wp.vec3(0.0, 0.0, 0.0)
    if d_here > 0.0001:
        for iw in range(numberOfWaves):
            wave = txWaves[iw, 0]
            result += sine_wave(x, y, total_time, d_here, wave[0], wave[1], wave[2], wave[3], boundary_g, PI)

    return wp.vec4(result[0] + seaLevel, result[1], result[2], 0.0)

# BoundaryPass Kernel
@wp.kernel
def boundary_pass_kernel(txState: wp.array2d(dtype=wp.vec4), txBottom: wp.array2d(dtype=wp.vec4), txWaves: wp.array2d(dtype=wp.vec4), txNewState: wp.array2d(dtype=wp.vec4),
                         width: wp.int32, height: wp.int32, dt: wp.float32, dx: wp.float32, dy: wp.float32, total_time: wp.float32, reflect_x: wp.int32, reflect_y: wp.int32,
                         PI: wp.float32, BoundaryWidth: wp.float32, seaLevel: wp.float32, boundary_nx: wp.int32, boundary_ny: wp.int32, numberOfWaves: wp.int32,
                         west_boundary_type: wp.int32, east_boundary_type: wp.int32, south_boundary_type: wp.int32, north_boundary_type: wp.int32, boundary_g: wp.float32):
    ix, iy = wp.tid()  # 2D thread index
    idx = wp.vec2i(ix, iy)
    BCState = txState[ix, iy]

    # Sponge Layers
    if west_boundary_type == 1 and ix <= 2.0 + BoundaryWidth:
        BCState = west_boundary_sponge(idx, PI, BoundaryWidth, txState)
    if east_boundary_type == 1 and ix >= float(width) - BoundaryWidth - 1.0:
        BCState = east_boundary_sponge(idx, PI, BoundaryWidth, boundary_nx, txState)
    if north_boundary_type == 1 and iy <= 2.0 + BoundaryWidth:
        BCState = north_boundary_sponge(idx, PI, BoundaryWidth, txState)
    if south_boundary_type == 1 and iy >= float(height) - BoundaryWidth - 1.0:
        BCState = south_boundary_sponge(idx, PI, BoundaryWidth, boundary_ny, txState)

    # Solid Walls
    if west_boundary_type <= 1 and ix <= 1:
        BCState = west_boundary_solid(idx, reflect_x, txState)
    elif west_boundary_type <= 1 and ix == 2:
        BCState[1] = 0.0
    if east_boundary_type <= 1 and ix >= width - 2:
        BCState = east_boundary_solid(idx, reflect_x, txState)
    elif east_boundary_type <= 1 and ix == width - 3:
        BCState[1] = 0.0
    if north_boundary_type <= 1 and iy <= 1:
        BCState = north_boundary_solid(idx, reflect_y, txState)
    elif north_boundary_type <= 1 and iy == 2:
        BCState[2] = 0.0
    if south_boundary_type <= 1 and iy >= height - 2:
        BCState = south_boundary_solid(idx, reflect_y, txState)
    elif south_boundary_type <= 1 and iy == height - 3:
        BCState[2] = 0.0

    # Sine Waves
    if west_boundary_type == 2 and ix <= 2:
        BCState = boundary_sine_wave(idx, txBottom, txWaves, dx, dy, seaLevel, total_time, numberOfWaves, boundary_g, PI)
    if east_boundary_type == 2 and ix >= width - 3:
        BCState = boundary_sine_wave(idx, txBottom, txWaves, dx, dy, seaLevel, total_time, numberOfWaves, boundary_g, PI)
    if north_boundary_type == 2 and iy <= 2:
        BCState = boundary_sine_wave(idx, txBottom, txWaves, dx, dy, seaLevel, total_time, numberOfWaves, boundary_g, PI)
    if south_boundary_type == 2 and iy >= height - 3:
        BCState = boundary_sine_wave(idx, txBottom, txWaves, dx, dy, seaLevel, total_time, numberOfWaves, boundary_g, PI)

    # Check for negative depths
    bottom = txBottom[ix, iy][2]
    elev = BCState[0]
    if elev <= bottom:
        BCState = wp.vec4(bottom, 0.0, 0.0, 0.0)

    txNewState[ix, iy] = BCState
