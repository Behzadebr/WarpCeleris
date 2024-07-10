import warp as wp

# MinMod function
@wp.func
def minmod(a: wp.float32, b: wp.float32, c: wp.float32):
    if a > 0.0 and b > 0.0 and c > 0.0:
        return wp.min(wp.min(a, b), c)
    elif a < 0.0 and b < 0.0 and c < 0.0:
        return wp.max(wp.max(a, b), c)
    else:
        return 0.0

# Reconstruct_w function
@wp.func
def reconstruct_w(west: wp.float32, here: wp.float32, east: wp.float32, TWO_THETAc: wp.float32):
    z1 = TWO_THETAc * (here - west)
    z2 = east - west
    z3 = TWO_THETAc * (east - here)
    
    dx_grad_over_two = 0.25 * minmod(z1, z2, z3)
    mu = 0.5 * (z1 + z2 + z3) / 3.0
    standard_deviation = wp.sqrt(((z1 - mu) * (z1 - mu) + (z2 - mu) * (z2 - mu) + (z3 - mu) * (z3 - mu)) / 3.0)
    out_east = here + dx_grad_over_two
    out_west = here - dx_grad_over_two
    return out_west, out_east, standard_deviation

# Reconstruct function
@wp.func
def reconstruct(west: wp.float32, here: wp.float32, east: wp.float32, TWO_THETAc: wp.float32):
    z1 = TWO_THETAc * (here - west)
    z2 = east - west
    z3 = TWO_THETAc * (east - here)
    
    dx_grad_over_two = 0.25 * minmod(z1, z2, z3)
    mu = 0.5 * (z1 + z2 + z3) / 3.0
    standard_deviation = wp.sqrt(((z1 - mu) * (z1 - mu) + (z2 - mu) * (z2 - mu) + (z3 - mu) * (z3 - mu)) / 3.0)
    out_east = here + dx_grad_over_two
    out_west = here - dx_grad_over_two
    return out_west, out_east, standard_deviation

# Correct_Edges function
@wp.func
def correct_edges(B_west: wp.float32, B_east: wp.float32, w_west: wp.float32, w_east: wp.float32,
                  H_west: wp.float32, H_east: wp.float32, U_west: wp.float32, U_east: wp.float32,
                  V_west: wp.float32, V_east: wp.float32, h_here: wp.float32, h_west: wp.float32, h_east: wp.float32):
    if h_here <= 0.0:
        w_east = w_west
        H_east = 0.0
        H_west = 0.0
        U_east = 0.0
        U_west = 0.0
        V_east = 0.0
        V_west = 0.0
    elif w_west <= B_east and h_east <= 0.0:
        w_east = w_west
    elif w_east <= B_west and h_west <= 0.0:
        w_west = w_east
    return w_west, w_east, H_west, H_east, U_west, U_east, V_west, V_east

# CorrectW function
@wp.func
def correct_w(B_west: wp.float32, B_east: wp.float32, w_bar: wp.float32, w_west: wp.float32, w_east: wp.float32):
    if w_east < B_east:
        w_east = B_east
        w_west = wp.max(B_west, 2.0 * w_bar - B_east)
    elif w_west < B_west:
        w_east = wp.max(B_east, 2.0 * w_bar - B_west)
        w_west = B_west
    return w_west, w_east

# Custom square root function for vec4
@wp.func
def vec4_sqrt(v: wp.vec4):
    return wp.vec4(wp.sqrt(v[0]), wp.sqrt(v[1]), wp.sqrt(v[2]), wp.sqrt(v[3]))

# CalcUVC function
@wp.func
def calc_uvc(h: wp.vec4, hu: wp.vec4, hv: wp.vec4, hc: wp.vec4, epsilon: wp.float32):
    h2 = wp.cw_mul(h, h)  # Component-wise multiplication
    divide_by_h = wp.cw_div(wp.cw_mul(h, wp.vec4(2.0, 2.0, 2.0, 2.0)), h2 + wp.max(h2, wp.vec4(epsilon, epsilon, epsilon, epsilon)))
    
    utemp = wp.cw_mul(divide_by_h, hu)
    vtemp = wp.cw_mul(divide_by_h, hv)
    ctemp = wp.cw_mul(divide_by_h, hc)
    
    speed = vec4_sqrt(wp.cw_mul(utemp, utemp) + wp.cw_mul(vtemp, vtemp))
    Fr = wp.cw_div(speed, vec4_sqrt(wp.cw_div(wp.vec4(9.81, 9.81, 9.81, 9.81), divide_by_h)))
    Frumax = wp.max(Fr)  # Maximum of all components of Fr
    Fr_maxallowed = 3.0

    if Frumax > Fr_maxallowed:
        Fr_red = Fr_maxallowed / Frumax
        utemp = wp.cw_mul(utemp, wp.vec4(Fr_red, Fr_red, Fr_red, Fr_red))
        vtemp = wp.cw_mul(vtemp, wp.vec4(Fr_red, Fr_red, Fr_red, Fr_red))

    return utemp, vtemp, ctemp

# Pass1 Kernel
@wp.kernel
def pass1_kernel(txState: wp.array2d(dtype=wp.vec4), txBottom: wp.array2d(dtype=wp.vec4), txAuxiliary2: wp.array2d(dtype=wp.vec4),
                 txH: wp.array2d(dtype=wp.vec4), txU: wp.array2d(dtype=wp.vec4), txV: wp.array2d(dtype=wp.vec4),
                 txNormal: wp.array2d(dtype=wp.vec4), txAuxiliary2Out: wp.array2d(dtype=wp.vec4), txW: wp.array2d(dtype=wp.vec4),
                 txC: wp.array2d(dtype=wp.vec4), width: int, height: int, one_over_dx: wp.float32, one_over_dy: wp.float32,
                 dissipation_threshold: wp.float32, TWO_THETA: wp.float32, epsilon: wp.float32, whiteWaterDecayRate: wp.float32, dt: wp.float32, base_depth: wp.float32):
    
    ix, iy = wp.tid()  # 2D thread index

    leftIdx_x = wp.max(ix - 1, 0)
    leftIdx_y = wp.max(iy, 0)
    rightIdx_x = wp.min(ix + 1, width - 1)
    rightIdx_y = wp.min(iy, height - 1)
    upIdx_x = wp.max(ix, 0)
    upIdx_y = wp.max(iy - 1, 0)
    downIdx_x = wp.min(ix, width - 1)
    downIdx_y = wp.min(iy + 1, height - 1)

    in_here = txState[ix, iy]
    in_south = txState[downIdx_x, downIdx_y]
    in_north = txState[upIdx_x, upIdx_y]
    in_west = txState[leftIdx_x, leftIdx_y]
    in_east = txState[rightIdx_x, rightIdx_y]

    B = wp.vec4(0.0, 0.0, 0.0, 0.0)
    B[0]= txBottom[ix, iy][0]
    B[1] = txBottom[ix, iy][1]
    B[2] = txBottom[downIdx_x, downIdx_y][0]
    B[3] = txBottom[leftIdx_x, leftIdx_y][1]

    B_here = txBottom[ix, iy][2]
    B_south = txBottom[downIdx_x, downIdx_y][2]
    B_north = txBottom[upIdx_x, upIdx_y][2]
    B_west = txBottom[leftIdx_x, leftIdx_y][2]
    B_east = txBottom[rightIdx_x, rightIdx_y][2]

    h_here = in_here[0] - B_here
    h_south = in_south[0] - B_south
    h_north = in_north[0] - B_north
    h_west = in_west[0] - B_west
    h_east = in_east[0] - B_east

    w = wp.vec4(0.0, 0.0, 0.0, 0.0)
    hu = wp.vec4(0.0, 0.0, 0.0, 0.0)
    hv = wp.vec4(0.0, 0.0, 0.0, 0.0)
    max_sd2 = 0.0
    temp_sd2 = 0.0

    wetdry = wp.min(B)
    rampcoef = wp.min(wp.max(0.0, wetdry / (0.02 * base_depth)), 1.0)
    TWO_THETAc = TWO_THETA * rampcoef + 2.0 * (1.0 - rampcoef)
    h = wp.vec4(0.0, 0.0, 0.0, 0.0)
    
    if wetdry >= 0:
        temp_west, temp_east, temp_sd2 = reconstruct(h_west, h_here, h_east, TWO_THETAc)
        h[3] = temp_west
        h[1] = temp_east
        max_sd2 = temp_sd2 * one_over_dx
        temp_west, temp_east, temp_sd2 = reconstruct(h_south, h_here, h_north, TWO_THETAc)
        h[2] = temp_west
        h[0] = temp_east
        max_sd2 = wp.max(max_sd2, temp_sd2 * one_over_dy)
        w = h + B
    else:
        temp_west, temp_east, temp_sd2 = reconstruct_w(in_west[0], in_here[0], in_east[0], TWO_THETAc)
        w[3] = temp_west
        w[1] = temp_east
        max_sd2 = temp_sd2 * one_over_dx
        temp_west, temp_east, temp_sd2 = reconstruct_w(in_south[0], in_here[0], in_north[0], TWO_THETAc)
        w[2] = temp_west
        w[0] = temp_east
        max_sd2 = wp.max(max_sd2, temp_sd2 * one_over_dy)
        h = w - B

    temp_west, temp_east, temp_sd2 = reconstruct(in_west[1], in_here[1], in_east[1], TWO_THETAc)
    hu[3] = temp_west
    hu[1] = temp_east
    temp_west, temp_east, temp_sd2 = reconstruct(in_south[1], in_here[1], in_north[1], TWO_THETAc)
    hu[2] = temp_west
    hu[0] = temp_east

    temp_west, temp_east, temp_sd2 = reconstruct(in_west[2], in_here[2], in_east[2], TWO_THETAc)
    hv[3] = temp_west
    hv[1] = temp_east
    temp_west, temp_east, temp_sd2 = reconstruct(in_south[2], in_here[2], in_north[2], TWO_THETAc)
    hv[2] = temp_west
    hv[0] = temp_east

    hc = wp.vec4(0.0, 0.0, 0.0, 0.0)
    temp_west, temp_east, temp_sd2 = reconstruct(in_west[3], in_here[3], in_east[3], TWO_THETAc)
    hc[3] = temp_west
    hc[1] = temp_east
    temp_west, temp_east, temp_sd2 = reconstruct(in_south[3], in_here[3], in_north[3], TWO_THETAc)
    hc[2] = temp_west
    hc[0] = temp_east

    u, v, c = calc_uvc(h, hu, hv, hc, epsilon)

    temp_west, temp_east, temp_h_west, temp_h_east, temp_u_west, temp_u_east, temp_v_west, temp_v_east = correct_edges(B[3], B[1], w[3], w[1], h[3], h[1], u[3], u[1], v[3], v[1], h_here, h_west, h_east)
    w[3] = temp_west
    w[1] = temp_east
    h[3] = temp_h_west
    h[1] = temp_h_east
    u[3] = temp_u_west
    u[1] = temp_u_east
    v[3] = temp_v_west
    v[1] = temp_v_east

    temp_west, temp_east, temp_h_west, temp_h_east, temp_u_west, temp_u_east, temp_v_west, temp_v_east = correct_edges(B[2], B[0], w[2], w[0], h[2], h[0], u[2], u[0], v[2], v[0], h_here, h_west, h_east)
    w[2] = temp_west
    w[0] = temp_east
    h[2] = temp_h_west
    h[0] = temp_h_east
    u[2] = temp_u_west
    u[0] = temp_u_east
    v[2] = temp_v_west
    v[0] = temp_v_east

    normal = wp.vec3(0.0, 0.0, 0.0)
    normal[0] = (in_west[0] - in_east[0]) * one_over_dx
    normal[1] = (in_south[0] - in_north[0]) * one_over_dy
    normal[2] = 2.0
    normal = wp.normalize(normal)

    maxInundatedDepth = wp.max((h[0] + h[1] + h[2] + h[3]) / 4.0, txAuxiliary2[ix, iy][0])

    n = wp.vec4(normal[0], normal[1], normal[2], 0.0)

    breaking_white = txAuxiliary2[ix, iy][3]
    if max_sd2 * wp.sign(normal[0] * in_here[1] + normal[1] * in_here[2]) > dissipation_threshold:
        breaking_white = 1.0

    breaking_white = breaking_white * wp.pow(wp.abs(whiteWaterDecayRate), dt)

    aux = wp.vec4(maxInundatedDepth, 0.0, max_sd2, breaking_white)

    txH[ix, iy] = h
    txU[ix, iy] = u
    txV[ix, iy] = v
    txNormal[ix, iy] = n
    txAuxiliary2Out[ix, iy] = aux
    txW[ix, iy] = w
    txC[ix, iy] = c
