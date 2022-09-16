# -*- coding: utf-8 -*-
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

from ra2ce.ra2ce_input import Ra2ceInput
from ra2ce.runners import AnalysisRunner, AnalysisRunnerFactory
from ra2ce.utils import initiate_root_logger

warnings.filterwarnings(
    action="ignore", message=".*initial implementation of Parquet.*"
)
warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Value *not successfully written.*")


class Ra2ceHandler:
    def __init__(self, network: Optional[Path], analysis: Optional[Path]) -> None:
        self.input_config = Ra2ceInput(network, analysis)
        self._initialize_logger(self.input_config)

    def _initialize_logger(self, input_config: Ra2ceInput) -> None:
        """
        Initializes the logger in the output directory, giving preference to the network output.

        Args:
            input_config (Ra2ceInput): Configuration containing ini data for both network and analysis.
        """
        _output_config = {}
        if input_config.network_config and input_config.network_config.is_valid():
            _output_config = input_config.network_config.config_data["output"]
        elif input_config.analysis_config.is_valid():
            _output_config = input_config.analysis_config.config_data["output"]
        else:
            raise ValueError()
        initiate_root_logger(_output_config / "RA2CE.log")

    def configure(self) -> None:
        self.input_config.configure()

    def run_analysis(self) -> None:
        if not self.input_config.is_valid_input():
            logging.error("Error validating input files. Ra2ce will close now.")
            sys.exit()
        _runner: AnalysisRunner = AnalysisRunnerFactory.get_runner(self.input_config)
        try:
            _runner().run(self.input_config.analysis_config)
        except BaseException as e:
            logging.exception(
                f"RA2CE crashed. Check the logfile for the Traceback message: {e}"
            )
