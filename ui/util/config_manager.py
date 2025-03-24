from json import load, dump, JSONDecodeError


class ConfigManager:
    """
    A class for managing configuration files used in the Start application.

    This class handles reading, writing, updating, and validating configuration files.
    It provides methods for easily adding or removing parameters from the configuration.
    """

    def __init__(self, log_console=None):
        """
        Initialize the ConfigManager.

        Args:
            log_console: Optional console for logging operations.
        """
        self.config_file_path = ""
        self.config_data = {}
        self.log_console = log_console

    def read_config_file(self, config_file_path):
        """
        Read a configuration file and load its contents.

        Args:
            config_file_path: Path to the configuration file.

        Returns:
            dict: The loaded configuration data, or None if an error occurred.
        """
        try:
            with open(config_file_path, "r") as file:
                config = load(file)
                self.config_file_path = config_file_path
                self.config_data = config
                
                if self.log_console:
                    self.log_console.printInfo(f"Loaded configuration from: {config_file_path}. Parameters: {self.config_data}")
                
                return config
        except FileNotFoundError:
            if self.log_console:
                self.log_console.logSignal.emit(f"File not found: {config_file_path}")
            return None
        except JSONDecodeError:
            if self.log_console:
                self.log_console.logSignal.emit(
                    f"Failed to decode JSON from {config_file_path}"
                )
            return None
        except Exception as e:
            if self.log_console:
                self.log_console.logSignal.emit(f"Error reading config file: {str(e)}")
            return None

    def save_config_to_file(self, config_data=None, file_path=None):
        """
        Save configuration data to a file.

        Args:
            config_data: Configuration data to save. If None, uses stored config_data.
            file_path: Path to save to. If None, uses stored config_file_path.

        Returns:
            bool: True if successful, False otherwise.
        """
        if config_data is None:
            config_data = self.config_data

        if file_path is None:
            file_path = self.config_file_path

        if not file_path:
            if self.log_console:
                self.log_console.logSignal.emit(
                    "No file path specified for saving configuration"
                )
            return False

        try:
            with open(file_path, "w") as file:
                dump(config_data, file, indent=4)

            self.config_file_path = file_path
            self.config_data = config_data

            if self.log_console:
                self.log_console.logSignal.emit(
                    f"Successfully saved configuration to {file_path}"
                )
            return True
        except Exception as e:
            if self.log_console:
                self.log_console.logSignal.emit(
                    f"Failed to save configuration: {str(e)}"
                )
            return False

    def add_parameter(self, key, value, file_path=None):
        """
        Add or update a parameter in the configuration.

        Args:
            key: Parameter key to add/update.
            value: Value to set for the parameter.
            file_path: Optional file path to save to. If None, uses stored path.

        Returns:
            bool: True if successful, False otherwise.
        """
        # If we have no config data, but we have a file path, try to load it first
        if not self.config_data and (file_path or self.config_file_path):
            self.read_config_file(file_path or self.config_file_path)

        # If we still have no config data, start with an empty dict
        if not self.config_data:
            self.config_data = {}

        # Add or update the parameter
        self.config_data[key] = value

        # Save the updated config
        return self.save_config_to_file(file_path=file_path)

    def remove_parameter(self, key, file_path=None):
        """
        Remove a parameter from the configuration.

        Args:
            key: Parameter key to remove.
            file_path: Optional file path to save to. If None, uses stored path.

        Returns:
            bool: True if successful, False otherwise.
        """
        # If we have no config data, but we have a file path, try to load it first
        if not self.config_data and (file_path or self.config_file_path):
            self.read_config_file(file_path or self.config_file_path)

        # If we still have no config data or the key doesn't exist, return False
        if not self.config_data or key not in self.config_data:
            if self.log_console:
                self.log_console.logSignal.emit(
                    f"Key '{key}' not found in configuration"
                )
            return False

        # Remove the parameter
        del self.config_data[key]

        # Save the updated config
        return self.save_config_to_file(file_path=file_path)

    def update_config(self, new_data, file_path=None):
        """
        Update multiple configuration parameters at once.

        Args:
            new_data: Dictionary of parameters to update.
            file_path: Optional file path to save to. If None, uses stored path.

        Returns:
            bool: True if successful, False otherwise.
        """
        # If we have no config data, but we have a file path, try to load it first
        if not self.config_data and (file_path or self.config_file_path):
            self.read_config_file(file_path or self.config_file_path)

        # If we still have no config data, start with an empty dict
        if not self.config_data:
            self.config_data = {}

        # Update the config data
        self.config_data.update(new_data)

        # Save the updated config
        return self.save_config_to_file(file_path=file_path)

    def get_parameter(self, key, default=None):
        """
        Get a parameter from the configuration.

        Args:
            key: Parameter key to get.
            default: Default value to return if the key is not found.

        Returns:
            The parameter value, or the default if not found.
        """
        return self.config_data.get(key, default)

    def has_parameter(self, key):
        """
        Check if a parameter exists in the configuration.

        Args:
            key: Parameter key to check.

        Returns:
            bool: True if the parameter exists, False otherwise.
        """
        return key in self.config_data

    def check_particle_sources(self):
        """
        Check if particle sources are defined in the configuration.

        Returns:
            bool: True if sources are defined, False otherwise.
        """
        if not self.config_data:
            return False

        sources = {}
        if "ParticleSourcePoint" in self.config_data:
            sources["ParticleSourcePoint"] = self.config_data["ParticleSourcePoint"]
        if "ParticleSourceSurface" in self.config_data:
            sources["ParticleSourceSurface"] = self.config_data["ParticleSourceSurface"]

        return bool(sources)

    def get_boundary_conditions(self):
        """
        Get the boundary conditions from the configuration.

        Returns:
            dict: The boundary conditions, or an empty dict if not found.
        """
        return self.config_data.get("Boundary Conditions", {})
