import json
import os

class ParametersLoader:
    def load_parameters(self, simulation_parameters, json_file_path):
        """
        Load simulation parameters from a JSON file into a SimulationParameters object.
        
        Args:
            simulation_parameters (SimulationParameters): The object to populate with parameters.
            json_file_path (str): The path to the JSON file.
        """
        
        # Check if the file exists at the provided path
        if not os.path.exists(json_file_path):
            print(f"Could not find JSON file at {json_file_path}")
            return
        
        try:
            # Read all the text from the file
            with open(json_file_path, 'r') as file:
                json_content = file.read()
        except Exception as e:
            print(f"Failed to read JSON file: {e}")
            return
        
        try:
            # Parse the JSON file and overwrite the values into simulation_parameters
            data = json.loads(json_content)
            for key, value in data.items():
                if hasattr(simulation_parameters, key):
                    setattr(simulation_parameters, key, value)
        except Exception as e:
            print(f"Failed to parse JSON file: {e}")
            return
