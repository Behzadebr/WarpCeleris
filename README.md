# Python Wave Simulator Using NVIDIA Warp

## Overview
The Python Wave Simulator Project, part of the Celeris2023 initiative, aims to model wave dynamics with high accuracy using advanced computational technologies. This project utilizes NVIDIA Warp for writing high-performance simulation and graphics code, focusing on GPU programming for real-time and faster-than-real-time simulations. Key objectives include capturing detailed wave dynamics, achieving efficient computational performance, and providing a user-friendly interface for simulation control and visualization.

## Simulation Architecture
### Initial Setup and Data Generation
- **Depth Surface and Ocean Elevation Initialization:** Generate a depth surface representing the ocean's bathymetry and an initial ocean elevation state.
- **Wave Parameter Initialization:** Initialize wave characteristics based on predefined or user-specified conditions.

### Compute Kernel Pipeline
- **Pass1:** State reconstruction and preliminary calculations for water dynamics using Warp kernels.
- **Pass2:** Flux calculation and adjustment in the x and y directions using Warp kernels.
- **Pass3:** State update using NLSW and Boussinesq methods, handling wave dispersion and non-linearity using Warp kernels.
- **Boundary Handling:** Special handling for simulation boundaries, including walls, sponge layers, and wave generation using Warp kernels.
- **Tridiagonal Matrix Solver (PCR):** Efficiently handle tridiagonal matrix systems for Boussinesq simulations using Warp kernels.

### Rendering and Visualization
- Render the simulated ocean surface, converting data into visual forms for real-time visualization and analysis.

## Implementation Details
### Data Management
- **Warp Array Initialization:** Initialize Warp arrays for bathymetry, initial ocean state, and dynamic simulation data.
- **Data Encoding in Warp Arrays:** Utilize Warp kernels to encode simulation data into Warp arrays for efficient GPU processing.

### Simulation Parameters and Configuration
- **Environmental Settings:** Define computational domain size, cell size, and boundary conditions.
- **Physical Properties:** Control base depth, friction parameters, and dispersion characteristics.
- **Time Integration Scheme:** Choose from Euler Integration, Third-order Adams-Bashforth Predictor, or Fourth-order Predictor-Corrector.

## Contributions and Innovations
- **Efficient GPU Utilization:** Leverage modern GPUs through NVIDIA Warp for high-speed and accurate wave simulations.
- **High Level of User Control and Interactivity:** Offer control and interactivity for research and training purposes.
- **Modular Architecture:** Facilitate integration with various data processing and rendering systems.
- **Real-Time and Predictive Simulation Capabilities:** Enable real-time and faster-than-real-time simulations for predictive modeling.

## Getting Started
1. **Clone the Repository:**
    ```sh
    git clone https://github.com/Behzadebr/WarpCeleris.git
    ```
2. **Install Dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
3. **Run the Simulation:**
    - Configure simulation parameters in the provided JSON files.
    - Execute the simulation script:
    ```sh
    python simulator.py
    ```

## Contributions
- **Prof. Patrick Lynett:** Project lead, Advisor.
- **Behzad Ebrahimi:** Developer, focusing on simulation architecture, Warp kernels, and GPU utilization.

## Contact
For questions, please contact Behzad Ebrahimi at bebrahim@usc.edu

## License
This project is licensed under the MIT License - see the LICENSE file for details.
