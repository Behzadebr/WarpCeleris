import warp as wp
from kernels.kernel_utils import (boundary_sine_wave, SolitaryWave)

@wp.kernel
def Boundary_Pass(
    txState: wp.array2d(dtype=wp.vec4),
    Bottom: wp.array2d(dtype=wp.vec4),
    Waves: wp.array2d(dtype=wp.vec4),
    State_Sed: wp.array2d(dtype=wp.float32),
    NewState_Sed: wp.array2d(dtype=wp.float32),
    width: wp.int32,
    height: wp.int32,
    dx: wp.float32,
    dy: wp.float32,
    time: wp.float32,
    amplitude: wp.float32,
    g: wp.float32,
    reflect_x: wp.int32,
    reflect_y: wp.int32,
    BoundaryWidth: wp.int32,
    base_depth: wp.float32,
    boundary_nx: wp.int32,
    boundary_ny: wp.int32,
    numberOfWaves: wp.int32,
    west_boundary_type: wp.int32,
    east_boundary_type: wp.int32,
    south_boundary_type: wp.int32,
    north_boundary_type: wp.int32,
    incident_wave_type: wp.int32,
    nSL: wp.float32,
    sSL: wp.float32,
    eSL: wp.float32,
    wSL: wp.float32,
    boundary_g: wp.float32,
    delta: wp.float32,
    boundary_shift: wp.int32
):
    """
    Warp kernel to handle boundary conditions, including sponge layers,
    solid walls, incoming waves, and wet/dry states.
    """

    ix, iy = wp.tid()

    # Ensure thread is within bounds
    if ix >= width or iy >= height:
        return

    BCState = txState[ix, iy]
    BCState_Sed = wp.max(State_Sed[ix, iy], 0.0)

    # SPONGE LAYERS

    # West boundary sponge layer
    if (west_boundary_type == 1 and ix <= 2 + BoundaryWidth):
        gamma = wp.pow(
            0.5 * (0.5 + 0.5 * wp.cos(wp.PI * (float(BoundaryWidth - ix) + 2.0) / float(BoundaryWidth - 1))),
            0.005
        )
        BCState = wp.cw_mul(txState[ix, iy], wp.vec4(gamma, gamma, gamma, gamma))
        BCState_Sed = 0.0

    # East boundary sponge layer
    if (east_boundary_type == 1 and ix >= width - BoundaryWidth - 1):
        gamma = wp.pow(
            0.5 * (0.5 + 0.5 * wp.cos(wp.PI * float(BoundaryWidth - boundary_nx - ix) / float(BoundaryWidth - 1))),
            0.005
        )
        BCState = wp.cw_mul(txState[ix, iy], wp.vec4(gamma, gamma, gamma, gamma))
        BCState_Sed = 0.0

    # South boundary sponge layer
    if (south_boundary_type == 1 and iy <= 2 + BoundaryWidth):
        gamma = wp.pow(
            0.5 * (0.5 + 0.5 * wp.cos(wp.PI * (float(BoundaryWidth - iy) + 2.0) / float(BoundaryWidth - 1))),
            0.005
        )
        BCState = wp.cw_mul(txState[ix, iy], wp.vec4(gamma, gamma, gamma, gamma))
        BCState_Sed = 0.0

    # North boundary sponge layer
    if (north_boundary_type == 1 and iy >= height - BoundaryWidth - 1):
        gamma = wp.pow(
            0.5 * (0.5 + 0.5 * wp.cos(wp.PI * float(BoundaryWidth - (boundary_ny - iy)) / float(BoundaryWidth - 1))),
            0.005
        )
        BCState = wp.cw_mul(txState[ix, iy], wp.vec4(gamma, gamma, gamma, gamma))
        BCState_Sed = 0.0

    # SOLID WALLS

    # West boundary solid walls
    if west_boundary_type <= 1:
        if ix <= 1:
            BCState.x = txState[boundary_shift - ix, iy].x
            BCState.y = -txState[boundary_shift - ix, iy].y
            BCState.z = txState[boundary_shift - ix, iy].z
            BCState.w = txState[boundary_shift - ix, iy].w
            BCState_Sed = 0.0
        elif ix == 2:
            BCState.y = 0.0
            BCState_Sed = 0.0

    # East boundary solid walls
    if east_boundary_type <= 1:
        if ix >= width - 2:
            BCState.x = txState[reflect_x - ix, iy].x
            BCState.y = -txState[reflect_x - ix, iy].y
            BCState.z = txState[reflect_x - ix, iy].z
            BCState.w = txState[reflect_x - ix, iy].w
            BCState_Sed = 0.0
        elif ix == width - 3:
            BCState.y = 0.0
            BCState_Sed = 0.0

    # South boundary solid walls
    if south_boundary_type <= 1:
        if iy <= 1:
            BCState.x = txState[ix, boundary_shift - iy].x
            BCState.y = txState[ix, boundary_shift - iy].y
            BCState.z = -txState[ix, boundary_shift - iy].z
            BCState.w = txState[ix, boundary_shift - iy].w
            BCState_Sed = 0.0
        elif iy == 2:
            BCState.z = 0.0
            BCState_Sed = 0.0

    # North boundary solid walls
    if north_boundary_type <= 1:
        if iy >= height - 2:
            BCState.x = txState[ix, reflect_y - iy].x
            BCState.y = txState[ix, reflect_y - iy].y
            BCState.z = -txState[ix, reflect_y - iy].z
            BCState.w = txState[ix, reflect_y - iy].w
            BCState_Sed = 0.0
        elif iy >= height - 3:
            BCState.z = 0.0
            BCState_Sed = 0.0

    # INCOMING WAVES

    # West boundary incoming waves
    if west_boundary_type == 2 and ix <= 2:
        if incident_wave_type <= 2:
            B_here = -base_depth
            d_here = wp.max(0.0, -B_here)
            x = float(ix) * dx
            y = float(iy) * dy
            bcwave = boundary_sine_wave(numberOfWaves, Waves, x, y, time, d_here, boundary_g, incident_wave_type) #check
            BCState = wp.vec4(bcwave.x + wSL, bcwave.y, bcwave.z, 0.0)
            BCState_Sed = 0.0
        elif incident_wave_type == 3:
            d_here = wp.max(0.0, wSL - Bottom[ix, iy].z)
            x0 = -10.0 * base_depth
            sol = SolitaryWave(x0, 0.0, 0.0, float(ix) * dx, float(iy) * dy, time, d_here, amplitude, g) #check
            eta = sol.x
            hu = sol.y
            hv = sol.z
            BCState = wp.vec4(eta + wSL, hu, hv, 0.0)
            BCState_Sed = 0.0

    # East boundary incoming waves
    if east_boundary_type == 2 and ix >= width - 3:
        if incident_wave_type <= 2:
            B_here = -base_depth
            d_here = wp.max(0.0, -B_here)
            x = float(ix) * dx
            y = float(iy) * dy
            bcwave = boundary_sine_wave(numberOfWaves, Waves, x, y, time, d_here, boundary_g, incident_wave_type)
            BCState = wp.vec4(bcwave.x + eSL, bcwave.y, bcwave.z, 0.0)
            BCState_Sed = 0.0
        elif incident_wave_type == 3:
            d_here = wp.max(0.0, eSL - Bottom[ix, iy].z)
            x0 = float(width) * dx + 10.0 * base_depth
            y0 = 0.0
            theta = -wp.PI
            sol = SolitaryWave(x0, y0, theta, float(ix) * dx, float(iy) * dy, time, d_here, amplitude, g)
            eta = sol.x
            hu = sol.y
            hv = sol.z
            BCState = wp.vec4(eta + eSL, hu, hv, 0.0)
            BCState_Sed = 0.0

    # South boundary incoming waves
    if south_boundary_type == 2 and iy <= 2:
        if incident_wave_type <= 2:
            B_here = -base_depth
            d_here = wp.max(0.0, -B_here)
            x = float(ix) * dx
            y = float(iy) * dy
            bcwave = boundary_sine_wave(numberOfWaves, Waves, x, y, time, d_here, boundary_g, incident_wave_type)
            BCState = wp.vec4(bcwave.x + sSL, bcwave.y, bcwave.z, 0.0)
            BCState_Sed = 0.0
        elif incident_wave_type == 3:
            d_here = wp.max(0.0, sSL - Bottom[ix, iy].z)
            x0 = 0.0
            y0 = -10.0 * base_depth
            theta = wp.PI / 2.0
            sol = SolitaryWave(x0, y0, theta, float(ix) * dx, float(iy) * dy, time, d_here, amplitude, g)
            eta = sol.x
            hu = sol.y
            hv = sol.z
            BCState = wp.vec4(eta + sSL, hu, hv, 0.0)
            BCState_Sed = 0.0

    # North boundary incoming waves
    if north_boundary_type == 2 and iy >= height - 3:
        if incident_wave_type <= 2:
            B_here = -base_depth
            d_here = wp.max(0.0, -B_here)
            x = float(ix) * dx
            y = float(iy) * dy
            bcwave = boundary_sine_wave(numberOfWaves, Waves, x, y, time, d_here, boundary_g, incident_wave_type)
            BCState = wp.vec4(bcwave.x + nSL, bcwave.y, bcwave.z, 0.0)
            BCState_Sed = 0.0
        elif incident_wave_type == 3:
            d_here = wp.max(0.0, nSL - Bottom[ix, iy].z)
            y0 = float(height) * dy + 10.0 * base_depth
            theta = -wp.PI / 2.0
            sol = SolitaryWave(0.0, y0, theta, float(ix) * dx, float(iy) * dy, time, d_here, amplitude, g)
            eta = sol.x
            hu = sol.y
            hv = sol.z
            BCState = wp.vec4(eta + nSL, hu, hv, 0.0)
            BCState_Sed = 0.0

    # WET/DRY STATE HANDLING

    # Neighbor indices
    rightIdx = wp.min(ix + 1, width - 1)
    upIdx = wp.min(iy + 1, height - 1)
    leftIdx = wp.max(ix - 1, 0)
    downIdx = wp.max(iy - 1, 0)

    # Bottom elevation and states of neighbors
    B_here = Bottom[ix, iy].z
    B_south = Bottom[ix, downIdx].z
    B_north = Bottom[ix, upIdx].z
    B_west = Bottom[leftIdx, iy].z
    B_east = Bottom[rightIdx, iy].z

    state_south = txState[ix, downIdx]
    state_north = txState[ix, upIdx]
    state_west = txState[leftIdx, iy]
    state_east = txState[rightIdx, iy]

    # eta for current and neighbors
    eta_here = BCState.x
    eta_west = state_west.x
    eta_east = state_east.x
    eta_south = state_south.x
    eta_north = state_north.x

    # water depth
    h_here  = eta_here - B_here
    h_south = eta_south - B_south
    h_north = eta_north - B_north
    h_west = eta_west - B_west
    h_east = eta_east - B_east

    # h_cut to determine dry conditions
    h_cut = wp.vec4(
        wp.max(delta, wp.abs(B_here - B_north)),
        wp.max(delta, wp.abs(B_here - B_east)),
        wp.max(delta, wp.abs(B_here - B_south)),
        wp.max(delta, wp.abs(B_here - B_west))
    )

    # Initialize dry flags
    dry_here = 1
    dry_west = 1
    dry_east = 1
    dry_south = 1
    dry_north = 1

    # Update dry flags based on water depth
    if h_here <= delta:
        dry_here = 0
    if h_west <= h_cut.w:
        dry_west = 0
    if h_east <= h_cut.y:
        dry_east = 0
    if h_south <= h_cut.z:
        dry_south = 0
    if h_north <= h_cut.x:
        dry_north = 0

    sum_dry = dry_west + dry_east + dry_south + dry_north

    # Compute minimum water depth between current and neighbors
    h_min = wp.vec4(
        wp.min(h_here, h_north),
        wp.min(h_here, h_east),
        wp.min(h_here, h_south),
        wp.min(h_here, h_west)
    )

    # Remove artificial islands by adjusting water surface elevation
    if dry_here == 1:
        if sum_dry == 0:
            if B_here <= 0.0:
                BCState = wp.vec4(wp.max(BCState.x, B_here), 0.0, 0.0, 0.0)
                BCState_Sed = 0.0
            else:
                BCState = wp.vec4(B_here, 0.0, 0.0, 0.0)
                BCState_Sed = 0.0
        elif sum_dry == 1:
            wet_eta = (float(dry_west) * eta_west +
                       float(dry_east) * eta_east +
                       float(dry_south) * eta_south +
                       float(dry_north) * eta_north) / float(sum_dry)
            BCState = wp.vec4(wet_eta, 0.0, 0.0, 0.0)
            BCState_Sed = 0.0

    # Check for negative depths and correct them
    h_here = BCState.x - B_here
    if h_here <= delta:
        if B_here <= 0.0:
            BCState = wp.vec4(wp.max(BCState.x, B_here), 0.0, 0.0, 0.0)
        else:
            BCState = wp.vec4(B_here, 0.0, 0.0, 0.0)
        BCState_Sed = 0.0

    # Update State Arrays
    txState[ix, iy] = BCState
    NewState_Sed[ix, iy] = BCState_Sed