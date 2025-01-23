import warp as wp
    
@wp.func
def minmod(a: wp.float32, b: wp.float32, c: wp.float32) -> wp.float32:
    """
    Compute the MinMod function:
    - Returns the minimum of the positive inputs if all are positive.
    - Returns the maximum of the negative inputs if all are negative.
    - Returns zero otherwise.
    """
    if a > 0.0 and b > 0.0 and c > 0.0:
        return wp.min(wp.min(a, b), c)
    elif a < 0.0 and b < 0.0 and c < 0.0:
        return wp.max(wp.max(a, b), c)
    else:
        return 0.0


@wp.func
def Reconstruct(
    west: wp.float32,
    here: wp.float32,
    east: wp.float32,
    TWO_THETAc: wp.float32
) -> wp.vec2:
    """
    Reconstructs the west and east values using the MinMod limiter.
    """
    z1 = TWO_THETAc * (here - west)
    z2 = east - west
    z3 = TWO_THETAc * (east - here)
    
    min_value = minmod(z1, z2, z3)
    dx_grad_over_two = 0.25 * min_value
    
    return wp.vec2(here - dx_grad_over_two, here + dx_grad_over_two)

# Custom square root function for vec2
@wp.func
def vec2_sqrt(v: wp.vec2):
    return wp.vec2(wp.sqrt(v.x), wp.sqrt(v.y))

# Custom square root function for vec4
@wp.func
def vec4_sqrt(v: wp.vec4):
    return wp.vec4(wp.sqrt(v[0]), wp.sqrt(v[1]), wp.sqrt(v[2]), wp.sqrt(v[3]))

@wp.func
def NumericalFlux(
    aplus: wp.float32,
    aminus: wp.float32,
    Fplus: wp.float32,
    Fminus: wp.float32,
    Udifference: wp.float32
) -> wp.float32:
    """
    Calculate the numerical flux.
    """
    numerical_flux = 0.0
    flux_diff = aplus - aminus
    if flux_diff != 0.0:
        numerical_flux = (aplus * Fminus - aminus * Fplus + aplus * aminus * Udifference) / flux_diff
    return numerical_flux


@wp.func
def CalcUV(
    h: wp.vec4,
    hu: wp.vec4,
    hv: wp.vec4,
    hc: wp.vec4,
    epsilon: wp.float32,
    dB_max: wp.vec4
):
    """
    Calculate velocity components based on depth and momentum.
    """
    epsilon_c = wp.max(wp.vec4(epsilon, epsilon, epsilon, epsilon), dB_max)
    divide_by_h = 2.0 * wp.cw_div(h, (wp.cw_mul(h, h) + wp.max(wp.cw_mul(h, h), epsilon_c)))
    return wp.cw_mul(divide_by_h, hu), wp.cw_mul(divide_by_h, hv), wp.cw_mul(divide_by_h, hc)

@wp.func
def CalcUV_Sed(
    h: wp.float32,
    hc1: wp.float32,
    hc2: wp.float32,
    hc3: wp.float32,
    hc4: wp.float32,
    epsilon: wp.float32,
    dB_max: wp.float32
) -> wp.vec4:
    """
    Calculate sediment velocity components based on depth and momentum.
    """
    epsilon_c = wp.max(epsilon, dB_max)
    divide_by_h = wp.sqrt(2.0) * h / (h * h + wp.max(h * h, epsilon_c))
    c1 = divide_by_h * hc1
    c2 = divide_by_h * hc2
    c3 = divide_by_h * hc3
    c4 = divide_by_h * hc4
    return wp.vec4(c1, c2, c3, c4)


@wp.func
def ScalarAntiDissipation(
    uplus: wp.float32,
    uminus: wp.float32,
    aplus: wp.float32,
    aminus: wp.float32,
    epsilon: wp.float32
) -> wp.float32:
    """
    Calculate the scalar anti-dissipation coefficient.
    """
    R = 0.0
    if (aplus != 0.0) and (aminus != 0.0):
        Fr = wp.abs(uplus) / aplus if wp.abs(uplus) >= wp.abs(uminus) else wp.abs(uminus) / aminus
        R = (Fr + epsilon) / (Fr + 1.0)
    elif (aplus == 0.0) or (aminus == 0.0):
        R = epsilon
    return R


@wp.func
def FrictionCalc(
    hu: wp.float32,
    hv: wp.float32,
    h: wp.float32,
    base_depth: wp.float32,
    delta: wp.float32,
    isManning: wp.int32,
    g: wp.float32,
    friction: wp.float32
) -> wp.float32:
    """
    Calculate friction
    """
    h_scaled = h / base_depth
    h2 = h_scaled * h_scaled
    h4 = h2 * h2
    epsilon_val = wp.max(h4, 1.e-6)
    divide_by_h2 = (2.0 * h2) / (h4 + epsilon_val) / (base_depth * base_depth)
    divide_by_h = 1.0 / wp.max(h, delta)

    f = friction
    if isManning == 1:
        f = g * wp.pow(friction, 2.0) * wp.pow(wp.abs(divide_by_h), 1.0 / 3.0)
    f = wp.min(f, 0.5)
    f = f * wp.sqrt(hu * hu + hv * hv) * divide_by_h2
    return f


@wp.func
def SolitaryWave(
    x0: wp.float32,
    y0: wp.float32,
    theta: wp.float32,
    x: wp.float32,
    y: wp.float32,
    time: wp.float32,
    d_here: wp.float32,
    amplitude: wp.float32,
    g: wp.float32
) -> wp.vec3:
    """
    Compute the solitary wave parameters eta, hu, hv based on position and time.
    """
    amp = amplitude
    xloc = x - x0
    yloc = y - y0
    k = wp.sqrt(0.75 * wp.abs(amp) / wp.pow(d_here, 3.0))
    c = wp.sqrt(g * (amp + d_here))
    eta = amp / wp.pow(wp.cosh(k * (xloc * wp.cos(theta) + yloc * wp.sin(theta) - c * time)), 2.0)
    hu = wp.sqrt(1.0 + 0.5 * amp / d_here) * eta * c * wp.cos(theta)
    hv = wp.sqrt(1.0 + 0.5 * amp / d_here) * eta * c * wp.sin(theta)
    return wp.vec3(eta, hu, hv)


@wp.func
def sine_wave(
    x: wp.float32,
    y: wp.float32,
    time: wp.float32,
    d: wp.float32,
    amplitude: wp.float32,
    period: wp.float32,
    theta: wp.float32,
    phase: wp.float32,
    g: wp.float32,
    wave_type: wp.int32
) -> wp.vec3:
    """
    Generates a sine wave perturbation.
    """
    # Calculate angular frequency
    omega = 2.0 * wp.PI / period

    # Calculate wavenumber k
    sqrt_tanh = wp.sqrt(wp.tanh(wp.pow(omega, 2.0) * d / g))
    k = wp.pow(omega, 2.0) / (g * sqrt_tanh)

    # Calculate phase terms
    kx = wp.cos(theta) * x * k
    ky = wp.sin(theta) * y * k

    # Compute wave elevation eta with time scaling
    eta = amplitude * wp.sin(omega * time - kx - ky + phase) * wp.min(1.0, time / period)

    # Handle different wave types
    num_waves = 0.0
    if wave_type == 2:
        num_waves = 4.0

    if num_waves > 0.0:
        eta *= wp.max(0.0, wp.min(1.0, ((num_waves * period - time) / period)))

    # Calculate wave speed
    speed = g * eta / omega * wp.tanh(k * d)

    # Calculate momentum components
    hu = speed * wp.cos(theta)
    hv = speed * wp.sin(theta)

    return wp.vec3(eta, hu, hv)


@wp.func
def boundary_sine_wave(
    numberOfWaves: wp.int32,
    Waves: wp.array2d(dtype=wp.vec4),
    x: wp.float32,
    y: wp.float32,
    time: wp.float32,
    d_here: wp.float32,
    boundary_g: wp.float32,
    wave_type: wp.int32
) -> wp.vec3:
    
    result = wp.vec3(0.0, 0.0, 0.0)

    if d_here > 0.0001:
        for iw in range(numberOfWaves):
            wave = Waves[iw, 0]
            result += sine_wave(x, y, time, d_here, wave[0], wave[1], wave[2], wave[3], boundary_g, wave_type)

    return result
