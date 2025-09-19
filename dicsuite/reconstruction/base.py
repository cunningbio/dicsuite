from abc import ABC, abstractmethod

class BaseReconstruction(ABC):
    def __init__(self, **kwargs):
        self.params = kwargs

    @classmethod
    def from_config(cls, user_params: dict):
        defaults = cls.default_params()
        defaults.update(user_params)
        return cls(**defaults)

    # Functions for handling iterating and QC over multiple input parameters
    @abstractmethod
    def param_combinations(self):
        """Create a sensible iterable from input parameters."""
        pass

    @abstractmethod
    def param_count(self):
        """Return the number of parameter combinations for iterating."""
        pass

    @abstractmethod
    def set_params(self, params):
        """Overwrite input parameters - intended for iterating."""
        pass

    @abstractmethod
    def get_params(self):
        """Return reconstruction-specific parameters for QC logging."""
        pass

    @abstractmethod
    def run(self, image, shear_angle):
        """Perform image reconstruction."""
        pass