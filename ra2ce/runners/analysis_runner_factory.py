import logging

from ra2ce.configuration.config_wrapper import ConfigWrapper
from ra2ce.runners.analysis_runner_protocol import AnalysisRunner
from ra2ce.runners.direct_analysis_runner import DirectAnalysisRunner
from ra2ce.runners.indirect_analysis_runner import IndirectAnalysisRunner


class AnalysisRunnerFactory:
    @staticmethod
    def get_runner(ra2ce_input: ConfigWrapper) -> AnalysisRunner:
        """
        Gets the supported first analysis runner for the given input. The runner is initialized within this method.

        Args:
            ra2ce_input (Ra2ceInput): Input representing a set of network and analysis ini configurations.

        Returns:
            AnalysisRunner: Initialized Ra2ce analysis runner.
        """
        _available_runners = [DirectAnalysisRunner, IndirectAnalysisRunner]
        _supported_runners = [
            _runner for _runner in _available_runners if _runner.can_run(ra2ce_input)
        ]
        if not _supported_runners:
            _err_mssg = "No analysis runner found for the given configuration."
            logging.error(_err_mssg)
            raise ValueError(_err_mssg)

        # Initialized selected supported runner (First one available).
        _selected_runner = _supported_runners[0]()
        if len(_supported_runners) > 1:
            logging.warn(f"More than one runner available, using {_selected_runner}")
        return _selected_runner