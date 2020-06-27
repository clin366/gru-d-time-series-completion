'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/6/21 
    Python Version: 3.6
'''

# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import sys
sys.path.append('.')
from airpolnowcast.data.train_test_split import extract_file as train_test_app


def call_click_command(cmd, *args, **kwargs):
    """ Wrapper to call a click command

    :param cmd: click cli command function to call
    :param args: arguments to pass to the function
    :param kwargs: keywrod arguments to pass to the function
    :return: None
    """

    # Get positional arguments from args
    arg_values = {c.name: a for a, c in zip(args, cmd.params)}
    args_needed = {c.name: c for c in cmd.params
                   if c.name not in arg_values}

    # build and check opts list from kwargs
    opts = {a.name: a for a in cmd.params if isinstance(a, click.Option)}
    for name in kwargs:
        if name in opts:
            arg_values[name] = kwargs[name]
        else:
            if name in args_needed:
                arg_values[name] = kwargs[name]
                del args_needed[name]
            else:
                raise click.BadParameter(
                    "Unknown keyword argument '{}'".format(name))


    # check positional arguments list
    for arg in (a for a in cmd.params if isinstance(a, click.Argument)):
        if arg.name not in arg_values:
            raise click.BadParameter("Missing required positional"
                                     "parameter '{}'".format(arg.name))

    # build parameter lists
    opts_list = sum(
        [[o.opts[0], str(arg_values[n])] for n, o in opts.items()], [])
    args_list = [str(v) for n, v in arg_values.items() if n not in opts]
    print('call_list')
    print(opts_list)
    print(args_list)
    # call the command
    cmd(opts_list + args_list)


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('merged_file_path', type=click.Path(exists=True))
@click.argument('train_data_path', type=click.Path())
@click.argument('valid_data_path', type=click.Path())
@click.argument('test_data_path', type=click.Path())
def train_test_split(config_path, merged_file_path, train_data_path, valid_data_path, test_data_path):
    logger = logging.getLogger(__name__)
    logger.info('data train-test splitting')

    train_test_cmd = ({'config_path': config_path, 'merged_file_path': merged_file_path,
                       'train_data_path': train_data_path, 'valid_data_path': valid_data_path,
                       'test_data_path': test_data_path},)

    call_click_command(train_test_app, *train_test_cmd[:-1], **train_test_cmd[-1])


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    train_test_split()
