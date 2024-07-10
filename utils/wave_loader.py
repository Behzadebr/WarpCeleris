import numpy as np

class WaveLoader:
    @staticmethod
    def load_irr_waves(sim_params, file_path, rotation_angle=0):
        """
        Load irregular wave data from a file and update simulation parameters.

        Args:
            sim_params (SimulationParameters): The simulation parameters object to be updated.
            file_path (str): The path to the irregular waves file.
            rotation_angle (float): The angle by which to rotate wave directions.
        """
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Parse the number of waves from the second line of the file
            nwaves_str = lines[1].strip().replace('[NumberOfWaves]', '').strip()
            sim_params.numberOfWaves = int(nwaves_str)

            # Initialize waveVectors array
            sim_params.waveVectors = np.zeros((sim_params.numberOfWaves, 4), dtype=np.float32)

            for i in range(sim_params.numberOfWaves):
                wave_data_str = lines[i + 3].split()
                if len(wave_data_str) < 4:
                    raise ValueError(f"IrrWaves file at {file_path} is not in the correct format.")

                w1, w2, w3, w4 = map(float, wave_data_str)

                # Rotate theta by rotation_angle and ensure the result stays within [0, 360)
                w3 = (w3 + rotation_angle) % 360

                sim_params.waveVectors[i] = [w1, w2, w3, w4]

            sim_params.waveVectors = np.expand_dims(sim_params.waveVectors, axis=1)
        except IOError as e:
            print(f"Could not find IrrWaves file at {file_path}: {e}")
        except ValueError as e:
            print(e)

