import sys
import argparse
import pathlib
import logging
import pandas as pd
from ..config import Config
from .. import components


logger = logging.getLogger(__name__)


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

    logging.basicConfig(level=logging.INFO)

    logger.info("Reading configuration file")
    config = Config.from_path(args.config)
    create_output_directory(config)

    logger.info("Loading input data file")
    data = pd.read_csv(args.data)

    logger.info("Executing pipeline")
    for module in components.pipeline_modules:
        data = module(data, config)

    logger.info("Finished")

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


def create_output_directory(config):
    """Create the output directory structure given the configuration object."""
    config.output_path.mkdir(parents=True, exist_ok=True)
