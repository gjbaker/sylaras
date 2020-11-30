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
        config.control_name = str(data['control_name'])
        config.test_name = str(data['test_name'])
        config.jitter = float(data['jitter'])
        config.filter_choice = FilterChoice[data['filter_choice']]
        config._parse_classes(data['classes'])
        config.alpha = float(data['alpha'])
        config.output_path = pathlib.Path(data['output_path']).resolve()
        config.celltype1 = str(data['celltype1'])
        config.celltype2 = str(data['celltype2'])
        config.xaxis_marker = str(data['xaxis_marker'])
        config.yaxis_marker = str(data['yaxis_marker'])

        return config

    def _parse_classes(self, value):
        self.classes = {}
        self.lineages = {}
        self.landmarks = {}

        if value is None:
            return

        for name, terms in value.items():

            vector = [BooleanTerm.parse_str(t) for t in terms[0]]
            lineage = terms[1]
            if len(terms) == 3:
                landmark = terms[2]
            else:
                landmark = None

            unknown_terms = (
                set(t.name for t in vector) -
                set(self.id_channels)
                )
            if unknown_terms:
                raise ValueError(
                    f"Class '{name}' includes terms {unknown_terms} which are"
                    " not in id_channels"
                )

            self.classes[name] = vector
            self.lineages[name] = lineage
            self.landmarks[name] = landmark

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

    @property
    def dashboards_path(self):
        return self.output_path / 'dashboards'

    @property
    def stats_path(self):
        return self.output_path / 'stats'

    def __repr__(self):
        kwargs_str = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Config({kwargs_str})"
