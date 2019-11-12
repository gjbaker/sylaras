import sys
import argparse
import pathlib
import pandas as pd
from ..config import Config
from .. import components


def main(argv=sys.argv):

    parser = argparse.ArgumentParser(
        description='Perform SYLARAS analysis on a data file')
    parser.add_argument(
        'data', type=path_resolved,
        help='Path to the input data CSV file'
    )
    parser.add_argument(
        'config', type=path_resolved,
        help='Path to the configuration YAML file'
    )
    args = parser.parse_args(argv[1:])
    if not validate_paths(args):
        return 1

    config = Config.from_path(args.config)
    create_output_folder(config)

    data = pd.read_csv(args.data)
    data2 = components.random_subset(data, config)

    return 0


def path_resolved(path_str):
    """Return a resolved Path for a string."""
    path = pathlib.Path(path_str)
    path = path.resolve()
    return path


def validate_paths(args):
    """Validate the Path entries in the argument list."""
    ok = True
    if not args.data.exists():
        print(f"Data path does not exist:\n    {args.data}\n", file=sys.stderr)
        ok = False
    if not args.config.exists():
        print(
            f"Config path does not exist:\n     {args.config}\n",
            file=sys.stderr
        )
        ok = False
    return ok


def create_output_folder(config):
    """Create the output folder given the configuration object."""
    config.output_path.mkdir(parents=True, exist_ok=True)
