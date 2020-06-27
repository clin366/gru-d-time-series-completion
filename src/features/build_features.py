import sys
sys.path.append('.')
import configparser
from configparser import ExtendedInterpolation

import click
import logging
from pathlib import Path
import os
from src.features.utils import save_pkl_dataset


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('train_data_path', type=click.Path(exists=True))
@click.argument('valid_data_path', type=click.Path(exists=True))
@click.argument('test_data_path', type=click.Path(exists=True))
def build_features(config_path, train_data_path, valid_data_path, test_data_path):
    logger = logging.getLogger(__name__)
    logger.info('build features')
    pars = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    pars.read(config_path)

    save_pkl_dataset(train_data_path, pars)
    save_pkl_dataset(valid_data_path, pars)
    save_pkl_dataset(test_data_path, pars)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    build_features()






























