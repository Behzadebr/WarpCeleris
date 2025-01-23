# pass1.py
import warp as wp
from kernels.kernel_utils import (Reconstruct, CalcUV, vec4_sqrt)

@wp.kernel
def Pass1(
    State: wp.array2d(dtype=wp.vec4),
    Bottom: wp.array2d(dtype=wp.vec4),
    Hnear: wp.array2d(dtype=wp.vec4),
    H: wp.array2d(dtype=wp.vec4),
    U: wp.array2d(dtype=wp.vec4),
    V: wp.array2d(dtype=wp.vec4),
    C: wp.array2d(dtype=wp.vec4),
    Auxiliary: wp.array2d(dtype=wp.vec4),
    TWO_THETA: wp.float32,
    epsilon: wp.float32,
    delta: wp.float32,
    double_dx: wp.float32,
    double_dy: wp.float32,
    base_depth: wp.float32,
    width: wp.int32,
    height: wp.int32
):

    ix, iy = wp.tid()
    
    if ix >= width or iy >= height:
        return
    
    zero = wp.vec4(0.0, 0.0, 0.0, 0.0)

    rightIdx = wp.min(ix + 1, width - 1)
    upIdx = wp.min(iy + 1, height - 1)
    leftIdx = wp.max(ix - 1, 0)
    downIdx = wp.max(iy - 1, 0)
    
    in_here = State[ix, iy]
    in_S = State[ix, downIdx]
    in_N = State[ix, upIdx]
    in_W = State[leftIdx, iy]
    in_E = State[rightIdx, iy]
    
    B_here = Bottom[ix, iy].z
    B_south = Bottom[ix, downIdx].z
    B_north = Bottom[ix, upIdx].z
    B_west = Bottom[leftIdx, iy].z
    B_east = Bottom[rightIdx, iy].z
    
    h_here  = in_here.x - B_here
    h_south = in_S.x - B_south
    h_north = in_N.x - B_north
    h_west  = in_W.x - B_west
    h_east  = in_E.x - B_east
    
    Hnear[ix, iy] = wp.vec4(h_north, h_east, h_south, h_west)
    
    if h_here <= delta:
        if h_north <= delta and h_east <= delta and h_south <= delta and h_west <= delta:
            H[ix, iy] = zero
            U[ix, iy] = zero
            V[ix, iy] = zero
            C[ix, iy] = zero
            return
    
    # Pass 1
    B = wp.vec4(
        Bottom[ix, iy].x,          # BN
        Bottom[ix, iy].y,          # BE
        Bottom[ix, downIdx].x,     # BS
        Bottom[leftIdx, iy].y      # BW
    )
    
    dB_max = wp.vec4(0.0, 0.0, 0.0, 0.0)
    
    dB_west = wp.abs(B_here - B_west)
    dB_east = wp.abs(B_here - B_east)
    dB_south = wp.abs(B_here - B_south)
    dB_north = wp.abs(B_here - B_north)
    
    h = wp.vec4(0.0, 0.0, 0.0, 0.0)
    w = wp.vec4(0.0, 0.0, 0.0, 0.0)
    hu = wp.vec4(0.0, 0.0, 0.0, 0.0)
    hv = wp.vec4(0.0, 0.0, 0.0, 0.0)
    hc = wp.vec4(0.0, 0.0, 0.0, 0.0)
    
    wetdry = wp.min(h_here, wp.min(h_south, wp.min(h_north, wp.min(h_west, h_east))))
    rampcoef = wp.min(wp.max(0.0, wetdry / (0.02 * base_depth)), 1.0)

    TWO_THETAc = TWO_THETA * rampcoef + 2.0 * (1.0 - rampcoef)
    
    if wetdry <= epsilon:
        dB_max = 0.5 * wp.vec4(dB_north, dB_east, dB_south, dB_west)
    
    # Reconstruction eta using the generalized minmod limiter
    wwy = Reconstruct(in_W.x, in_here.x, in_E.x, TWO_THETAc)
    wzx = Reconstruct(in_S.x, in_here.x, in_N.x, TWO_THETAc)
    w = wp.vec4(wzx.y, wwy.y, wzx.x, wwy.x)
    
    h = w - B
    h = wp.max(h, wp.vec4(0.0, 0.0, 0.0, 0.0))
    
    huwy = Reconstruct(in_W.y, in_here.y, in_E.y, TWO_THETAc)
    huzx = Reconstruct(in_S.y, in_here.y, in_N.y, TWO_THETAc)
    hu = wp.vec4(huzx.y, huwy.y, huzx.x, huwy.x)
    
    hvwy = Reconstruct(in_W.z, in_here.z, in_E.z, TWO_THETAc)
    hvzx = Reconstruct(in_S.z, in_here.z, in_N.z, TWO_THETAc)
    hv = wp.vec4(hvzx.y, hvwy.y, hvzx.x, hvwy.x)
    
    hcwy = Reconstruct(in_W.w, in_here.w, in_E.w, TWO_THETAc)
    hczx = Reconstruct(in_S.w, in_here.w, in_N.w, TWO_THETAc)
    hc = wp.vec4(hczx.y, hcwy.y, hczx.x, hcwy.x)
    
    output_u, output_v, output_c = CalcUV(h, hu, hv, hc, epsilon, dB_max)
    
    # Froude number limiter
    epsilon_c = wp.max(wp.vec4(epsilon, epsilon, epsilon, epsilon), dB_max)
    divide_by_h = 2.0 * wp.cw_div(h, (wp.cw_mul(h, h) + wp.max(wp.cw_mul(h, h), epsilon_c)))
    
    Fr = wp.cw_div(vec4_sqrt(wp.cw_mul(output_u, output_u) + wp.cw_mul(output_v, output_v)),
                  vec4_sqrt(wp.cw_div(wp.vec4(9.81, 9.81, 9.81, 9.81), divide_by_h)))
    Frumax = wp.max(Fr.x, wp.max(Fr.y, wp.max(Fr.z, Fr.w)))
    dBdx = wp.abs(B_east - B_west) / double_dx
    dBdy = wp.abs(B_north - B_south) / double_dy
    dBds_max = wp.max(dBdx, dBdy)
    Fr_maxallowed = 3.0 / wp.max(1.0, dBds_max)
    
    if Frumax > Fr_maxallowed:
        Fr_red = Fr_maxallowed / Frumax
        output_u = wp.cw_mul(output_u, wp.vec4(Fr_red, Fr_red, Fr_red, Fr_red))
        output_v = wp.cw_mul(output_v, wp.vec4(Fr_red, Fr_red, Fr_red, Fr_red))
    
    # Compute maximum inundated depth
    maxInundatedDepth = wp.max((h.x + h.y + h.z + h.w) / 4.0, Auxiliary[ix, iy].x)
    
    H[ix, iy] = h
    U[ix, iy] = output_u
    V[ix, iy] = output_v
    C[ix, iy] = output_c
    Auxiliary[ix, iy] = wp.vec4(maxInundatedDepth, 0.0, 0.0, 0.0)