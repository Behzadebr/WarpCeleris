import warp as wp
from kernels.kernel_utils import Reconstruct

@wp.kernel
def Pass1_SedTrans(
    State_Sed: wp.array2d(dtype=wp.float32),
    Bottom: wp.array2d(dtype=wp.vec4),
    H: wp.array2d(dtype=wp.vec4),
    Sed_C: wp.array2d(dtype=wp.vec4),
    TWO_THETA: wp.float32,
    epsilon: wp.float32,
    base_depth: wp.float32,
    width: wp.int32,
    height: wp.int32
):

    ix, iy = wp.tid()
    
    if ix >= width or iy >= height:
        return
    
    # Compute neighbor indices with boundary checks
    rightIdx = wp.min(ix + 1, width - 1)
    upIdx = wp.min(iy + 1, height - 1)
    leftIdx = wp.max(ix - 1, 0)
    downIdx = wp.max(iy - 1, 0)
    
    in_here = State_Sed[ix, iy]
    in_S = State_Sed[ix, downIdx]
    in_N = State_Sed[ix, upIdx]
    in_W = State_Sed[leftIdx, iy]
    in_E = State_Sed[rightIdx, iy]
    
    B_here = Bottom[ix, iy].z
    B_south = Bottom[ix, downIdx].z
    B_north = Bottom[ix, upIdx].z
    B_west = Bottom[leftIdx, iy].z
    B_east = Bottom[rightIdx, iy].z
    
    dB_west = wp.abs(B_here - B_west)
    dB_east = wp.abs(B_here - B_east)
    dB_south = wp.abs(B_here - B_south)
    dB_north = wp.abs(B_here - B_north)
    
    dB_max = 0.5 * wp.vec4(dB_north, dB_east, dB_south, dB_west)
    
    h_here  = in_here - B_here
    h_south = in_S - B_south
    h_north = in_N - B_north
    h_west  = in_W - B_west
    h_east  = in_E - B_east
    
    wetdry = wp.min(h_here, wp.min(h_south, wp.min(h_north, wp.min(h_west, h_east))))
    rampcoef = wp.min(wp.max(0.0, wetdry / (0.02 * base_depth)), 1.0)
    
    TWO_THETAc = TWO_THETA * rampcoef + 2.0 * (1.0 - rampcoef)
    
    hcwy = Reconstruct(in_W, in_here, in_E, TWO_THETAc)
    hczx = Reconstruct(in_S, in_here, in_N, TWO_THETAc)
    
    hc = wp.vec4(hczx.y, hcwy.y, hczx.x, hcwy.x)
    
    h = H[ix, iy]
    
    epsilon_c = wp.max(epsilon, dB_max)
    
    divide_by_h = 2.0 * wp.cw_div(h, (wp.cw_mul(h, h) + wp.max(wp.cw_mul(h, h), epsilon_c)))
    
    c_sed = wp.cw_mul(divide_by_h, hc)
    
    Sed_C[ix, iy] = c_sed