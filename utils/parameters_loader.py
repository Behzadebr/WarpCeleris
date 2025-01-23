import json
import os
from utils.simulation_parameters import SimulationParameters

class ParametersLoader:
    def load_parameters(self, simulation_parameters: SimulationParameters, json_file_path: str):
        """
        Load simulation parameters from config.json file.
        """
        # Check if the file exists
        if not os.path.exists(json_file_path):
            print(f"Could not find JSON file at {json_file_path}")
            return

        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
        except Exception as e:
            print(f"Failed to read or parse JSON file: {e}")
            return

        # Update SimulationParameters with JSON data
        for key, value in data.items():
            if hasattr(simulation_parameters, key):
                setattr(simulation_parameters, key, value)
            else:
                print(f"Warning: SimulationParameters has no attribute '{key}'")