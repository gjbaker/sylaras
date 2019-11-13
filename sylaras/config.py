import pathlib
from enum import Enum, auto
import yaml


class FilterChoice(Enum):
    full = auto()
    kernel = auto()
    kernel_bias = auto()


class Config:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_path(cls, path):
        config = cls()
        with open(path) as f:
            data = yaml.safe_load(f)
        config.id_channels = data['id_channels']
        config.other_channels = data['other_channels']
        config.random_sample_size = int(data['random_sample_size'])
        config.random_seed = int(data['random_seed'])
        config.kernel_low = float(data['kernel_low'])
        config.kernel_high = float(data['kernel_high'])
        config.jitter = float(data['jitter'])
        config.filter_choice = FilterChoice[data['filter_choice']]
        config.output_path = pathlib.Path(data['output_path']).resolve()
        return config

    @property
    def filtered_data_path(self):
        return self.output_path / 'filtered_data'

    @property
    def figure_path(self):
        return self.output_path / 'figures'

    def __repr__(self):
        kwargs_str = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Config({kwargs_str})"
