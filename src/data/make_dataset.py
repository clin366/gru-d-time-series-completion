# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from airpolnowcast.data.extract_search_trend import extract_search_trend
from airpolnowcast.data.extract_pol_label import extract_file as extract_pol_label
from airpolnowcast.data.extract_phys_meas import extract_file as extract_phy_meas
from airpolnowcast.data.process_phys_feature import extract_file as proc_phys_feature


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
@click.argument('search_file_path', type=click.Path())
@click.argument('pol_label_path', type=click.Path())
@click.argument('phys_meas_path', type=click.Path())
@click.argument('proc_phys_path', type=click.Path())
def main(config_path, search_file_path, pol_label_path, phys_meas_path, proc_phys_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    try:
        search_cmd = ({'config_path': config_path, 'output_filepath': search_file_path},)
        call_click_command(extract_search_trend, *search_cmd[:-1], **search_cmd[-1])
    except SystemExit as e:
        if e.code != 0:
            raise
    try:
        pol_cmd = ({'config_path': config_path, 'output_filepath': pol_label_path},)
        call_click_command(extract_pol_label, *pol_cmd[:-1], **pol_cmd[-1])
    except SystemExit as e:
        if e.code != 0:
            raise

    try:
        phy_cmd = ({'config_path': config_path, 'output_filepath': phys_meas_path},)
        call_click_command(extract_phy_meas, *phy_cmd[:-1], **phy_cmd[-1])
    except SystemExit as e:
        if e.code != 0:
            raise

    try:
        proc_cmd = ({'config_path': config_path, 'phys_file_path': phys_meas_path,  'output_filepath': proc_phys_path},)
        call_click_command(proc_phys_feature, *proc_cmd[:-1], **proc_cmd[-1])
    except SystemExit as e:
        if e.code != 0:
            raise


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
