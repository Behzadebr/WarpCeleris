import warp as wp
import math
from kernels.boundary_pass import Boundary_Pass
from kernels.pass1 import Pass1
from kernels.pass1_SedTrans import Pass1_SedTrans
from kernels.pass2 import Pass2
from kernels.tridiag_pcrx import TriDiag_PCRx
from kernels.tridiag_pcry import TriDiag_PCRy
from kernels.pass3_nlsw import Pass3_NLSW
from kernels.pass3_SedTrans import Pass3_SedTrans
from kernels.pass3_bous import Pass3_Bous
from kernels.pass_breaking import Pass_Breaking
from kernels.update_bottom import Update_Bottom

def launch_boundary_pass(wp_arr, sim_params, time, txState, device="cuda"):
    """
    Launches the BoundaryPass kernel to handle boundary conditions.
    """
    wp.launch(
        kernel=Boundary_Pass,
        dim=(sim_params.WIDTH, sim_params.HEIGHT),
        inputs=[
            txState,
            wp_arr.Bottom,
            wp_arr.Waves,
            wp_arr.State_Sed,
            wp_arr.NewState_Sed,
            wp.int32(sim_params.WIDTH),
            wp.int32(sim_params.HEIGHT),
            wp.float32(sim_params.dx),
            wp.float32(sim_params.dy),
            wp.float32(time),
            wp.float32(sim_params.amplitude),
            wp.float32(sim_params.g),
            wp.int32(sim_params.reflect_x),
            wp.int32(sim_params.reflect_y),
            wp.int32(sim_params.BoundaryWidth),
            wp.float32(sim_params.base_depth),
            wp.int32(sim_params.boundary_nx),
            wp.int32(sim_params.boundary_ny),
            wp.int32(sim_params.numberOfWaves),
            wp.int32(sim_params.west_boundary_type),
            wp.int32(sim_params.east_boundary_type),
            wp.int32(sim_params.south_boundary_type),
            wp.int32(sim_params.north_boundary_type),
            wp.int32(sim_params.incident_wave_type),
            wp.float32(sim_params.nSL),
            wp.float32(sim_params.sSL),
            wp.float32(sim_params.eSL),
            wp.float32(sim_params.wSL),
            wp.float32(sim_params.boundary_g),
            wp.float32(sim_params.delta),
            wp.int32(sim_params.boundary_shift)
        ],
        device=device
    )

    wp.synchronize()
    

def launch_pass1(wp_arr, sim_params, device="cuda"):

    wp.launch(
        kernel=Pass1,
        dim=(sim_params.WIDTH, sim_params.HEIGHT),
        inputs=[
            wp_arr.State,
            wp_arr.Bottom,
            wp_arr.Hnear,
            wp_arr.H,
            wp_arr.U,
            wp_arr.V,
            wp_arr.C,
            wp_arr.Auxiliary,
            wp.float32(sim_params.TWO_THETA),
            wp.float32(sim_params.epsilon),
            wp.float32(sim_params.delta),
            wp.float32(sim_params.double_dx),
            wp.float32(sim_params.double_dy),
            wp.float32(sim_params.base_depth),
            wp.int32(sim_params.WIDTH),
            wp.int32(sim_params.HEIGHT)
        ],
        device=device
    )

    wp.synchronize()


def launch_pass1_sedtrans(wp_arr, sim_params, device="cuda"):

    wp.launch(
        kernel=Pass1_SedTrans,
        dim=(sim_params.WIDTH, sim_params.HEIGHT),
        inputs=[
            wp_arr.State_Sed,
            wp_arr.Bottom,
            wp_arr.H,
            wp_arr.Sed_C,
            wp.float32(sim_params.TWO_THETA),
            wp.float32(sim_params.epsilon),
            wp.float32(sim_params.base_depth),
            wp.int32(sim_params.WIDTH),
            wp.int32(sim_params.HEIGHT)
        ],
        device=device
    )

    wp.synchronize()


def launch_pass2(wp_arr, sim_params, device="cuda"):

    wp.launch(
        kernel=Pass2,
        dim=(sim_params.WIDTH, sim_params.HEIGHT),
        inputs=[
            wp_arr.Hnear,
            wp_arr.H,
            wp_arr.U,
            wp_arr.V,
            wp_arr.C,
            wp_arr.Bottom,
            wp_arr.Sed_C,
            wp_arr.XFlux,
            wp_arr.YFlux,
            wp_arr.XFlux_Sed,
            wp_arr.YFlux_Sed,
            wp.float32(sim_params.g),
            wp.float32(sim_params.delta),
            wp.int32(sim_params.WIDTH),
            wp.int32(sim_params.HEIGHT),
            wp.bool(sim_params.useSedTransModel)
        ],
        device=device
    )
    
    wp.synchronize()


def launch_Tridiag(
    wp_arr,
    sim_params,
    device="cuda"
):

    if sim_params.NLSW_or_Bous == 0:
        # Directly copy current_stateUVstar to NewState
        wp.copy(src=wp_arr.current_stateUVstar, dest=wp_arr.NewState)
    else:
        # TriDiag_PCRx
        current_buffer_x = wp_arr.temp_PCRx1
        next_buffer_x = wp_arr.temp_PCRx2

        for p in range(sim_params.Px):
            s = 1 << p

            # Launch TriDiag_PCRx kernel
            wp.launch(
                kernel=TriDiag_PCRx,
                dim=(sim_params.WIDTH, sim_params.HEIGHT),
                inputs=[
                    wp_arr.coefMatx,
                    wp_arr.current_stateUVstar,
                    current_buffer_x,
                    next_buffer_x,
                    wp_arr.temp2_PCRx,
                    wp_arr.NewState,
                    wp.int32(p),
                    wp.int32(s),
                    wp.int32(sim_params.WIDTH),
                    wp.int32(sim_params.HEIGHT)
                ],
                device=device
            )

            wp.synchronize()

            # Swap buffers: next_buffer becomes current_buffer for next pass
            current_buffer_x, next_buffer_x = next_buffer_x, current_buffer_x

        wp.copy(src=wp_arr.temp2_PCRx, dest=wp_arr.NewState)

        # TriDiag_PCRy
        current_buffer_y = wp_arr.temp_PCRy1
        next_buffer_y = wp_arr.temp_PCRy2

        for p in range(sim_params.Py):
            s = 1 << p

            # Launch TriDiag_PCRy kernel
            wp.launch(
                kernel=TriDiag_PCRy,
                dim=(sim_params.WIDTH, sim_params.HEIGHT),
                inputs=[
                    wp_arr.coefMaty,
                    wp_arr.current_stateUVstar,
                    current_buffer_y,
                    next_buffer_y,
                    wp_arr.temp2_PCRy,
                    wp_arr.NewState,
                    wp.int32(p),
                    wp.int32(s),
                    wp.int32(sim_params.WIDTH),
                    wp.int32(sim_params.HEIGHT)
                ],
                device=device
            )

            wp.synchronize()

            # Swap buffers: next_buffer becomes current_buffer for next pass
            current_buffer_y, next_buffer_y = next_buffer_y, current_buffer_y

        wp.copy(src=wp_arr.temp2_PCRy, dest=wp_arr.NewState)

    wp.synchronize()


def launch_pass3_nlsw(wp_arr, sim_params, pred_or_corrector, device="cuda"):

    wp.launch(
        kernel=Pass3_NLSW,
        dim=(sim_params.WIDTH, sim_params.HEIGHT),
        inputs=[
            wp_arr.NewState,
            wp_arr.State,
            wp_arr.stateUVstar,
            wp_arr.Bottom,
            wp_arr.BottomFriction,
            wp_arr.ShipPressure,
            wp_arr.ContSource,
            wp_arr.Breaking,
            wp_arr.XFlux,
            wp_arr.YFlux,
            wp_arr.oldGradients,
            wp_arr.oldOldGradients,
            wp_arr.predictedGradients,
            wp_arr.dU_by_dt,
            wp_arr.predictedF_G_star,
            wp_arr.current_stateUVstar,
            wp.int32(sim_params.timeScheme),
            wp.int32(pred_or_corrector),
            wp.bool(sim_params.useBreakingModel),
            wp.int32(sim_params.showBreaking),
            wp.float32(sim_params.g),
            wp.float32(sim_params.dt),
            wp.float32(sim_params.one_over_dx),
            wp.float32(sim_params.one_over_dy),
            wp.float32(sim_params.one_over_d2x),
            wp.float32(sim_params.one_over_d2y),
            wp.float32(sim_params.one_over_dxdy),
            wp.float32(sim_params.g_over_dx),
            wp.float32(sim_params.g_over_dy),
            wp.float32(sim_params.delta),
            wp.int32(sim_params.isManning),
            wp.float32(sim_params.friction),
            wp.float32(sim_params.base_depth),
            wp.float32(sim_params.whiteWaterDispersion),
            wp.float32(sim_params.whiteWaterDecayRate),
            wp.float32(sim_params.infiltrationRate),
            wp.int32(sim_params.WIDTH),
            wp.int32(sim_params.HEIGHT)
        ],
        device=device
    )

    wp.synchronize()


def launch_pass3_sedtrans(wp_arr, sim_params, pred_or_corrector, device="cuda"):

    wp.launch(
        kernel=Pass3_SedTrans,
        dim=(sim_params.WIDTH, sim_params.HEIGHT),
        inputs=[
            wp_arr.NewState_Sed,
            wp_arr.dU_by_dt_Sed,
            wp_arr.erosion_Sed,
            wp_arr.deposition_Sed,
            wp_arr.State_Sed,
            wp_arr.State,
            wp_arr.Bottom,
            wp_arr.XFlux_Sed,
            wp_arr.YFlux_Sed,
            wp_arr.oldGradients_Sed,
            wp_arr.oldOldGradients_Sed,
            wp_arr.predictedGradients_Sed,
            wp.int32(sim_params.timeScheme),
            wp.float32(sim_params.g),
            wp.float32(sim_params.dt),
            wp.float32(sim_params.one_over_dx),
            wp.float32(sim_params.one_over_dy),
            wp.float32(sim_params.one_over_d2x),
            wp.float32(sim_params.one_over_d2y),
            wp.float32(sim_params.one_over_dxdy),
            wp.float32(sim_params.epsilon),
            wp.int32(sim_params.isManning),
            wp.float32(sim_params.friction),
            wp.int32(pred_or_corrector),
            wp.float32(sim_params.sedC1_shields),
            wp.float32(sim_params.sedC1_criticalshields),
            wp.float32(sim_params.sedC1_erosion),
            wp.float32(sim_params.sedC1_fallvel),
            wp.float32(sim_params.sedC1_n),
            wp.int32(sim_params.WIDTH),
            wp.int32(sim_params.HEIGHT)
        ],
        device=device
    )

    wp.synchronize()


def launch_pass3_bous(wp_arr, sim_params, pred_or_corrector, device="cuda"):

    wp.launch(
        kernel=Pass3_Bous,
        dim=(sim_params.WIDTH, sim_params.HEIGHT),
        inputs=[
            wp_arr.NewState,
            wp_arr.dU_by_dt,
            wp_arr.predictedF_G_star,
            wp_arr.current_stateUVstar,
            wp_arr.State,
            wp_arr.stateUVstar,
            wp_arr.Bottom,
            wp_arr.BottomFriction,
            wp_arr.XFlux,
            wp_arr.YFlux,
            wp_arr.F_G_star_oldOldGradients,
            wp_arr.oldGradients,
            wp_arr.oldOldGradients,
            wp_arr.predictedGradients,
            wp_arr.ShipPressure,
            wp_arr.ContSource,
            wp_arr.Breaking,
            wp_arr.DissipationFlux,
            wp.int32(sim_params.timeScheme),
            wp.int32(pred_or_corrector),
            wp.int32(sim_params.WIDTH),
            wp.int32(sim_params.HEIGHT),
            wp.float32(sim_params.dt),
            wp.float32(sim_params.one_over_dx),
            wp.float32(sim_params.one_over_dy),
            wp.float32(sim_params.g_over_dx),
            wp.float32(sim_params.g_over_dy),
            wp.float32(sim_params.one_over_d2x),
            wp.float32(sim_params.one_over_d3x),
            wp.float32(sim_params.one_over_d2y),
            wp.float32(sim_params.one_over_d3y),
            wp.float32(sim_params.one_over_dxdy),
            wp.float32(sim_params.Bcoef),
            wp.float32(sim_params.Bcoef_g),
            wp.float32(sim_params.delta),
            wp.float32(sim_params.base_depth),
            wp.float32(sim_params.whiteWaterDispersion),
            wp.float32(sim_params.whiteWaterDecayRate),
            wp.bool(sim_params.useBreakingModel),
            wp.int32(sim_params.showBreaking),
            wp.float32(sim_params.g),
            wp.int32(sim_params.isManning),
            wp.float32(sim_params.friction),
            wp.float32(sim_params.infiltrationRate)
        ],
        device=device
    )
    
    wp.synchronize()


def launch_pass_breaking(wp_arr, sim_params, time, device="cuda"):

    wp.launch(
        kernel=Pass_Breaking,
        dim=(sim_params.WIDTH, sim_params.HEIGHT),
        inputs=[
            wp_arr.State,
            wp_arr.XFlux,
            wp_arr.YFlux,
            wp_arr.Breaking,
            wp_arr.DissipationFlux,
            wp_arr.Bottom,
            wp_arr.dU_by_dt,
            wp.float32(time),
            wp.int32(sim_params.WIDTH),
            wp.int32(sim_params.HEIGHT),
            wp.float32(sim_params.dt),
            wp.float32(sim_params.dx),
            wp.float32(sim_params.dy),
            wp.float32(sim_params.one_over_dx),
            wp.float32(sim_params.one_over_dy),
            wp.float32(sim_params.T_star_coef),
            wp.float32(sim_params.dzdt_I_coef),
            wp.float32(sim_params.dzdt_F_coef),
            wp.float32(sim_params.delta_breaking),
            wp.float32(sim_params.g),
            wp.float32(sim_params.epsilon)
        ],
        device=device
    )
    
    wp.synchronize()


def launch_Update_Bottom(wp_arr, sim_params, device="cuda"):

    wp.launch(
        kernel=Update_Bottom,
        dim=(sim_params.WIDTH, sim_params.HEIGHT),
        inputs=[
            wp_arr.Bottom,
            wp_arr.erosion_Sed,
            wp_arr.deposition_Sed,
            wp.float32(sim_params.dt),
            wp.float32(sim_params.sedC1_n),
            wp.int32(sim_params.WIDTH),
            wp.int32(sim_params.HEIGHT)
        ],
        device=device
    )
    
    wp.synchronize()