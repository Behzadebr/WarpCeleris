# utils/wave_loader.py

import numpy as np

class WaveLoader:
    @staticmethod
    def load_waves(sim_params, file_path):
        """
        Load wave data from a file and update simulation parameters.
        """
        try:
            with open(file_path, 'r') as wavefile:
                lines = wavefile.readlines()

            # Parse the number of waves from lines containing 'NumberOfWaves'
            sim_params.numberOfWaves = 0
            for line in lines:
                if 'NumberOfWaves' in line:
                    sim_params.numberOfWaves = int(line.split()[1])
                    break

            if sim_params.numberOfWaves == 0:
                print("No waves found in the file.")
                return

            # Load wave data skipping the first 3 lines
            temp = np.loadtxt(file_path, skiprows=3, dtype=np.float32)

            # Initialize the waveVectors array
            sim_params.waveVectors = np.zeros((sim_params.numberOfWaves, 4), dtype=np.float32)

            for i in range(sim_params.numberOfWaves):
                if sim_params.numberOfWaves == 1:
                    sim_params.waveVectors[i, :] = temp
                else:
                    sim_params.waveVectors[i, :] = temp[i]

            # Expand dimensions to have similar shape to other data arrays
            sim_params.waveVectors = np.expand_dims(sim_params.waveVectors, axis=1)

        except IOError as e:
            print(f"Could not find waves file at {file_path}: {e}")
        except ValueError as e:
            print(f"Error parsing waves file: {e}")
