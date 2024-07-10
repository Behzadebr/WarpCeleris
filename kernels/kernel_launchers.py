import warp as wp
import math
from kernels.pass1_kernel import pass1_kernel
from kernels.pass2_kernel import pass2_kernel
from kernels.boundary_pass_kernel import boundary_pass_kernel
from kernels.tridiag_pcrx_kernel import tridiag_pcrx_kernel
from kernels.tridiag_pcry_kernel import tridiag_pcry_kernel
from kernels.pass3_nlsw_kernel import pass3_nlsw_kernel
from kernels.pass3_bous_kernel import pass3_bous_kernel

# Helper function to launch boundary_pass kernel
def launch_boundary_pass(wp_arr, sim_params, total_time, device="cuda"):
    # Launch the BoundaryPass kernel
    wp.launch(
        kernel=boundary_pass_kernel,
        dim=wp_arr.txBottom.shape,
        inputs=[
            wp_arr.current_stateUVstar, wp_arr.txBottom, wp_arr.txWaves, wp_arr.txtemp,     #check the args(current_stateUVstar and )
            sim_params.WIDTH, sim_params.HEIGHT, sim_params.dt, sim_params.dx, sim_params.dy,
            total_time, sim_params.reflect_x, sim_params.reflect_y,
            math.pi, sim_params.BoundaryWidth, sim_params.seaLevel,
            sim_params.boundary_nx, sim_params.boundary_ny, sim_params.numberOfWaves,
            sim_params.west_boundary_type, sim_params.east_boundary_type,
            sim_params.south_boundary_type, sim_params.north_boundary_type,
            sim_params.boundary_g
        ],
        device=device
    )

    # Synchronize to ensure the BoundaryPass kernel execution is finished
    wp.synchronize()

    # Copy txNewState to txState
    wp.copy(src=wp_arr.txtemp, dest=wp_arr.current_stateUVstar)

    wp.synchronize()


# Helper function to launch pass1 kernel
def launch_pass1(wp_arr, sim_params, device="cuda"):
    # Launch the Pass1 kernel
    wp.launch(
        kernel=pass1_kernel,
        dim=wp_arr.txBottom.shape,
        inputs=[
            wp_arr.txState, wp_arr.txBottom, wp_arr.txAuxiliary2,
            wp_arr.txH, wp_arr.txU, wp_arr.txV, wp_arr.txNormal,
            wp_arr.txAuxiliary2Out, wp_arr.txW, wp_arr.txC,
            sim_params.WIDTH, sim_params.HEIGHT,
            sim_params.one_over_dx, sim_params.one_over_dy,
            sim_params.dissipation_threshold, sim_params.TWO_THETA,
            sim_params.epsilon, sim_params.whiteWaterDecayRate,
            sim_params.dt, sim_params.base_depth
        ],
        device=device
    )

    # Synchronize to ensure the Pass1 kernel execution is finished
    wp.synchronize()

    # Copy txAuxiliary2Out to txAuxiliary2
    wp.copy(src=wp_arr.txAuxiliary2Out, dest=wp_arr.txAuxiliary2)

    wp.synchronize()

# Helper function to launch pass2 kernel
def launch_pass2(wp_arr, sim_params, device="cuda"):
    # Launch the Pass2 kernel
    wp.launch(
        kernel=pass2_kernel,
        dim=wp_arr.txBottom.shape,
        inputs=[
            wp_arr.txH, wp_arr.txU, wp_arr.txV, wp_arr.txBottom, wp_arr.txC,
            wp_arr.txXFlux, wp_arr.txYFlux,
            sim_params.WIDTH, sim_params.HEIGHT,
            sim_params.g, sim_params.half_g, sim_params.dx, sim_params.dy
        ],
        device=device
    )

    # Synchronize to ensure the Pass2 kernel execution is finished
    wp.synchronize()


# Helper function to launch tridiag_pcrx kernel
def launch_tridiag_pcrx(wp_arr, sim_params, device="cuda"):
    if sim_params.NLSW_or_Bous == 0:
        wp.copy(src=wp_arr.current_stateUVstar, dest=wp_arr.txNewState)
    else:
        for p in range(sim_params.Px):
            s = 1 << p  # Bit shift to calculate s as 2^p

            # Launch the TriDiag_PCRx kernel
            wp.launch(
                kernel=tridiag_pcrx_kernel,
                dim=wp_arr.txBottom.shape,
                inputs=[
                    wp_arr.coefMatx, wp_arr.txNewState, wp_arr.current_stateUVstar,
                    wp_arr.txtemp, wp_arr.txtemp2,
                    sim_params.WIDTH, sim_params.HEIGHT, p, s
                ],
                device=device
            )

            # Synchronize to ensure the TriDiag_PCRx kernel execution is finished
            wp.synchronize()

            # Copy new textures to old ones only if the loop counter is less than Px - 1
            if p < sim_params.Px - 1:
                wp.copy(src=wp_arr.txtemp, dest=wp_arr.newcoef)
                wp_arr.coefMatx = wp_arr.newcoef

        # After all the iterations, copy the new state into the current state
        wp.copy(src=wp_arr.txtemp2, dest=wp_arr.txNewState)

    wp.synchronize()


# Helper function to launch tridiag_pcry kernel
def launch_tridiag_pcry(wp_arr, sim_params, device="cuda"):
    if sim_params.NLSW_or_Bous == 0:
        wp.copy(src=wp_arr.current_stateUVstar, dest=wp_arr.txNewState)
    else:
        for p in range(sim_params.Py):
            s = 1 << p  # Bit shift to calculate s as 2^p

            # Launch the TriDiag_PCRy kernel
            wp.launch(
                kernel=tridiag_pcry_kernel,
                dim=wp_arr.txBottom.shape,
                inputs=[
                    wp_arr.coefMaty, wp_arr.txNewState, wp_arr.current_stateUVstar,
                    wp_arr.txtemp, wp_arr.txtemp2,
                    sim_params.WIDTH, sim_params.HEIGHT, p, s
                ],
                device=device
            )

            # Synchronize to ensure the TriDiag_PCRy kernel execution is finished
            wp.synchronize()

            # Copy new textures to old ones only if the loop counter is less than Py - 1
            if p < sim_params.Py - 1:
                wp.copy(src=wp_arr.txtemp, dest=wp_arr.newcoef)
                wp_arr.coefMaty = wp_arr.newcoef

        # After all the iterations, copy the new state into the current state
        wp.copy(src=wp_arr.txtemp2, dest=wp_arr.txNewState)

    wp.synchronize()


# Helper function to launch pass3_nlsw kernel
def launch_pass3_nlsw(wp_arr, sim_params, device="cuda"):
    # Launch the Pass3_NLSW kernel
    wp.launch(
        kernel=pass3_nlsw_kernel,
        dim=wp_arr.txBottom.shape,
        inputs=[
            wp_arr.txState, wp_arr.txBottom, wp_arr.txH, wp_arr.txXFlux, wp_arr.txYFlux,
            wp_arr.oldGradients, wp_arr.oldOldGradients, wp_arr.predictedGradients, wp_arr.F_G_star_oldOldGradients,
            wp_arr.txstateUVstar, wp_arr.txShipPressure, wp_arr.txNewState, wp_arr.dU_by_dt, wp_arr.F_G_star, wp_arr.current_stateUVstar,
            sim_params.WIDTH, sim_params.HEIGHT, sim_params.dt, sim_params.dx, sim_params.dy, sim_params.one_over_dx, sim_params.one_over_dy,
            sim_params.g_over_dx, sim_params.g_over_dy, sim_params.timeScheme, sim_params.epsilon, sim_params.isManning, sim_params.g, sim_params.friction,
            sim_params.pred_or_corrector, sim_params.Bcoef, sim_params.Bcoef_g, sim_params.one_over_d2x, sim_params.one_over_d3x, sim_params.one_over_d2y,
            sim_params.one_over_d3y, sim_params.one_over_dxdy, sim_params.seaLevel
        ],
        device=device
    )
    wp.synchronize()

    # Copy dU_by_dt to predictedGradients after kernel execution
    wp.copy(src=wp_arr.dU_by_dt, dest=wp_arr.predictedGradients)

    wp.synchronize()


# Helper function to launch pass3_bous kernel
def launch_pass3_bous(wp_arr, sim_params, device="cuda"):
    # Launch the Pass3_Bous kernel
    wp.launch(
        kernel=pass3_bous_kernel,
        dim=wp_arr.txBottom.shape,
        inputs=[
            wp_arr.txState, wp_arr.txBottom, wp_arr.txH, wp_arr.txXFlux, wp_arr.txYFlux,
            wp_arr.oldGradients, wp_arr.oldOldGradients, wp_arr.predictedGradients, wp_arr.F_G_star_oldOldGradients,
            wp_arr.txstateUVstar, wp_arr.txShipPressure, wp_arr.txNewState, wp_arr.dU_by_dt, wp_arr.F_G_star, wp_arr.current_stateUVstar,
            sim_params.WIDTH, sim_params.HEIGHT, sim_params.dt, sim_params.dx, sim_params.dy, sim_params.one_over_dx, sim_params.one_over_dy,
            sim_params.g_over_dx, sim_params.g_over_dy, sim_params.timeScheme, sim_params.epsilon, sim_params.isManning, sim_params.g, sim_params.friction,
            sim_params.pred_or_corrector, sim_params.Bcoef, sim_params.Bcoef_g, sim_params.one_over_d2x, sim_params.one_over_d3x, sim_params.one_over_d2y,
            sim_params.one_over_d3y, sim_params.one_over_dxdy, sim_params.seaLevel
        ],
        device=device
    )
    wp.synchronize()

    # Copy dU_by_dt to predictedGradients after kernel execution
    wp.copy(src=wp_arr.dU_by_dt, dest=wp_arr.predictedGradients)

    wp.synchronize()