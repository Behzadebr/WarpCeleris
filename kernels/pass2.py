import warp as wp
from kernels.kernel_utils import NumericalFlux, vec2_sqrt

@wp.kernel
def Pass2(
    Hnear: wp.array2d(dtype=wp.vec4),
    H: wp.array2d(dtype=wp.vec4),
    U: wp.array2d(dtype=wp.vec4),
    V: wp.array2d(dtype=wp.vec4),
    C: wp.array2d(dtype=wp.vec4),
    Bottom: wp.array2d(dtype=wp.vec4),
    Sed_C: wp.array2d(dtype=wp.vec4),
    XFlux: wp.array2d(dtype=wp.vec4),
    YFlux: wp.array2d(dtype=wp.vec4),
    XFlux_Sed: wp.array2d(dtype=wp.float32),
    YFlux_Sed: wp.array2d(dtype=wp.float32),
    g: wp.float32,
    delta: wp.float32,
    width: wp.int32,
    height: wp.int32,
    useSedTransModel: wp.bool
):

    ix, iy = wp.tid()
    
    if ix >= width or iy >= height:
        return
    
    rightIdx = wp.min(ix + 1, width - 1)
    upIdx = wp.min(iy + 1, height - 1)
    leftIdx = wp.max(ix - 1, 0)
    downIdx = wp.max(iy - 1, 0)
    
    h_vec = Hnear[ix, iy]
    
    h_here = wp.vec2(H[ix, iy].x, H[ix, iy].y)

    hW_east = H[rightIdx, iy].w
    hS_north = H[ix, upIdx].z
    
    u_here = wp.vec2(U[ix, iy].x, U[ix, iy].y)
    
    uW_east = U[rightIdx, iy].w
    uS_north = U[ix, upIdx].z
    
    v_here = wp.vec2(V[ix, iy].x, V[ix, iy].y)
    
    vW_east = V[rightIdx, iy].w
    vS_north = V[ix, upIdx].z
    
    cNE = vec2_sqrt(wp.vec2(g * h_here.x, g * h_here.y))
    cW = wp.sqrt(g * hW_east)
    cS = wp.sqrt(g * hS_north)
    
    aplus = wp.max(wp.max(u_here.y + cNE.y, uW_east + cW), 0.0)
    aminus = wp.min(wp.min(u_here.y - cNE.y, uW_east - cW), 0.0)
    bplus = wp.max(wp.max(v_here.x + cNE.x, vS_north + cS), 0.0)
    bminus = wp.min(wp.min(v_here.x - cNE.x, vS_north - cS), 0.0)
    
    B_here = Bottom[ix, iy].z
    
    dB_south = Bottom[ix, downIdx].z - B_here
    dB_north = Bottom[ix, upIdx].z - B_here
    dB_west = Bottom[leftIdx, iy].z - B_here
    dB_east = Bottom[rightIdx, iy].z - B_here
    dB = wp.max(dB_south, wp.max(dB_north, wp.max(dB_west, dB_east)))
    
    c_here = wp.vec2(C[ix, iy].x, C[ix, iy].y)
    cW_east = C[rightIdx, iy].w
    cS_north = C[ix, upIdx].z
    
    phix = wp.float32(0.5)
    phiy = wp.float32(0.5)
    
    minH = wp.min(h_vec.w, wp.min(h_vec.z, wp.min(h_vec.y, h_vec.x)))
    
    mass_diff_x = hW_east - h_here.y
    mass_diff_y = hS_north - h_here.x
    
    P_diff_x = hW_east * uW_east - h_here.y * u_here.y
    P_diff_y = hS_north * uS_north - h_here.x * u_here.x
    
    Q_diff_x = hW_east * vW_east - h_here.y * v_here.y
    Q_diff_y = hS_north * vS_north - h_here.x * v_here.x
    
    if (minH <= delta):
        mass_diff_x = 0.0
        mass_diff_y = 0.0
        phix = 1.0
        phiy = 1.0
    
    xflux = wp.vec4(
        NumericalFlux(aplus, aminus, hW_east * uW_east, h_here.y * u_here.y, mass_diff_x),
        NumericalFlux(aplus, aminus, hW_east * uW_east * uW_east, h_here.y * u_here.y * u_here.y, P_diff_x),
        NumericalFlux(aplus, aminus, hW_east * uW_east * vW_east, h_here.y * u_here.y * v_here.y, Q_diff_x),
        NumericalFlux(aplus, aminus, hW_east * uW_east * cW_east, h_here.y * u_here.y * c_here.y, phix * (hW_east * cW_east - h_here.y * c_here.y))
    )
    
    yflux = wp.vec4(
        NumericalFlux(bplus, bminus, hS_north * vS_north, h_here.x * v_here.x, mass_diff_y),
        NumericalFlux(bplus, bminus, hS_north * uS_north * vS_north, h_here.x * u_here.x * v_here.x, P_diff_y),
        NumericalFlux(bplus, bminus, hS_north * vS_north * vS_north, h_here.x * v_here.x * v_here.x, Q_diff_y),
        NumericalFlux(bplus, bminus, hS_north * cS_north * vS_north, h_here.x * c_here.x * v_here.x, phiy * (hS_north * cS_north - h_here.x * c_here.x))
    )
    
    XFlux[ix, iy] = xflux
    YFlux[ix, iy] = yflux
    
    # Handle Sediment Transport Model
    if (useSedTransModel):
        c1_here = wp.vec2(Sed_C[ix, iy].x, Sed_C[ix, iy].y)
        c1W_east = Sed_C[rightIdx, iy].w
        c1S_north = Sed_C[ix, upIdx].z
        
        xflux_Sed = NumericalFlux(
            aplus, aminus, 
            hW_east * uW_east * c1W_east, 
            h_here.y * u_here.y * c1_here.y, 
            phix * (hW_east * c1W_east - h_here.y * c1_here.y)
        )
        yflux_Sed = NumericalFlux(
            bplus, bminus, 
            hS_north * c1S_north * vS_north, 
            h_here.x * c_here.x * v_here.x, 
            phiy * (hS_north * c1S_north - h_here.x * c_here.x)
        )
        
        XFlux_Sed[ix, iy] = xflux_Sed
        YFlux_Sed[ix, iy] = yflux_Sed