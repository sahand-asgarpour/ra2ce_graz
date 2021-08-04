# -*- coding: utf-8 -*-
"""
Created on 26-7-2021

@author: F.C. de Groen, Deltares
"""

from pathlib import Path

# Local modules
from utils import parse_config, initiate_root_logger, configure_analyses
from checks import input_validation
from graph.networks import Network
from analyses.direct import analyses_direct
from analyses.indirect import analyses_indirect


def main():
    # Find the settings.ini file,
    root_path = Path(__file__).resolve().parent.parent
    setting_file = root_path / 'settings.ini'

    # Read the configurations in settings.ini and add the root path to the configuration dictionary.
    config = parse_config(path=setting_file)
    config['root_path'] = root_path

    # Initiate the log file, save in the output folder.
    initiate_root_logger(str(config['root_path'] / 'data' / config['project']['name'] / 'output' / 'RA2CE.log'))

    # Validate the configuration input.
    config = input_validation(config)
    config = configure_analyses(config)

    # Set the output paths in the configuration Dict for ease of saving to those folders.
    config['input'] = config['root_path'] / 'data' / config['project']['name'] / 'input'
    config['static'] = config['root_path'] / 'data' / config['project']['name'] / 'static'
    config['output'] = config['root_path'] / 'data' / config['project']['name'] / 'output'

    # Create the output folders
    if 'direct' in config:
        for a in config['direct']:
            output_path = config['output'] / a['analysis']
            output_path.mkdir(parents=True, exist_ok=True)

    if 'indirect' in config:
        for a in config['indirect']:
            output_path = config['output'] / a['analysis']
            output_path.mkdir(parents=True, exist_ok=True)

    # Create the network
    network = Network(config)
    g_indirect, g_direct = network.create()

    # Do the analyses
    if 'direct' in config:
        analyses_direct.DirectAnalyses(config, g_direct).execute()

    if 'indirect' in config:
        analyses_indirect.IndirectAnalyses(config, g_indirect).execute()


if __name__ == '__main__':
    main()
