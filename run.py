import os
from simulator import Simulator

def main():
    """
    Name of the example to run
    Options: Balboa_Pier_CA, Barry_Arm, Blacks_Beach_CA, Crescent_City, DuckFRF_NC, Greenland,
    Greenland_Umanak, Half_Moon_Bay, Hania_Greece, Harrison_Lake, Hermosa_Beach_CA, Ipan_Guam,
    LA_River_Model, Mavericks, Miami_Beach_FL, Miami_FL, Morro_Rock_CA, Newport_Jetties_CA,
    Newport_OR, OSU_Flume, OSU_Seaside, OSU_WaveBasin, Oceanside_CA, POLALB, Pacifica_CA,
    Portage_Lake_AK, SF_Bay_tides, SantaBarbara, Santa_Cruz, Santa_Cruz_tsunami, Scripps_Canyon,
    Scripps_Pier, Taan_fjord, Toy_Config, Tyndall_FL, Ventura, Waimea_Bay
    """
    example_name = 'Hermosa_Beach_CA'

    # Initialize Simulator
    simulator = Simulator(example_name=example_name, device='cuda')

    # Simulation steps and logging/saving interval
    total_steps = 10000
    log_interval = 100

    # Run the simulation
    simulator.run_simulation(total_steps=total_steps, log_interval=log_interval)

if __name__ == "__main__":
    main()