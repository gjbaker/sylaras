import yaml


class Config:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_path(cls, path):
        config = cls()
        with open(path) as f:
            data = yaml.load(f)
        config.id_channels = data['id_channels']
        config.other_channels = data['other_channels']
        config.random_sample_size = int(data['random_sample_size'])
        config.random_seed = int(data['random_seed'])
        config.kernel_low = float(data['kernel_low'])
        config.kernel_high = float(data['kernel_high'])
        config.jitter = float(data['jitter'])
        return config

    def __repr__(self):
        kwargs_str = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Config({kwargs_str})"
