import pathlib
from dataclasses import dataclass
from enum import Enum, auto
import yaml


class FilterChoice(Enum):
    full = auto()
    kernel = auto()
    kernel_bias = auto()


@dataclass(frozen=True)
class BooleanTerm:
    name: str
    negated: bool

    @classmethod
    def parse_str(cls, s):
        if s.startswith('~'):
            negated = True
            name = s[1:]
        else:
            negated = False
            name = s
        return cls(name, negated)

    def __repr__(self):
        s = self.name
        if self.negated:
            s = '~' + self.name
        return s

    def __invert__(self):
        return BooleanTerm(self.name, ~self.negated)


class Config:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_path(cls, path):
        config = cls()
        with open(path) as f:
            data = yaml.safe_load(f)
        config.data_path = pathlib.Path(data['data_path']).resolve()
        config.id_channels = data['id_channels']
        if any('~' in c for c in config.id_channels):
            raise ValueError("The '~' character is not allowed in channel names")
        config.other_channels = data['other_channels']
        config.random_sample_size = int(data['random_sample_size'])
        config.random_seed = int(data['random_seed'])
        config.kernel_low = float(data['kernel_low'])
        config.kernel_high = float(data['kernel_high'])
        config.jitter = float(data['jitter'])
        config.filter_choice = FilterChoice[data['filter_choice']]
        config._parse_classes(data['classes'])
        config.alpha = data['alpha']
        config.output_path = pathlib.Path(data['output_path']).resolve()

        return config

    def _parse_classes(self, value):
        self.classes = {}
        if value is None:
            return
        for name, terms in value.items():
            terms = [BooleanTerm.parse_str(t) for t in terms]
            unknown_terms = set(t.name for t in terms) - set(self.id_channels)
            if unknown_terms:
                raise ValueError(
                    f"Class '{name}' includes terms {unknown_terms} which are"
                    " not in id_channels"
                )
            self.classes[name] = terms

    @property
    def filtered_data_path(self):
        return self.output_path / 'filtered_data'

    @property
    def figure_path(self):
        return self.output_path / 'figures'

    @property
    def checkpoint_path(self):
        return self.output_path / 'checkpoints'

    @property
    def alpha_vectors_path(self):
        return self.output_path / 'alpha_vectors'

    def __repr__(self):
        kwargs_str = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Config({kwargs_str})"
