from ra2ce.analyses.indirect import analyses_indirect
from ra2ce.configuration import AnalysisConfigBase
from ra2ce.configuration.config_wrapper import ConfigWrapper
from ra2ce.runners.analysis_runner_protocol import AnalysisRunner


class IndirectAnalysisRunner(AnalysisRunner):
    def __str__(self) -> str:
        return "Indirect Analysis Runner"

    @staticmethod
    def can_run(ra2ce_input: ConfigWrapper) -> bool:
        return "indirect" in ra2ce_input.analysis_config.config_data

    def run(self, analysis_config: AnalysisConfigBase) -> None:
        analyses_indirect.IndirectAnalyses(
            analysis_config.config_data, analysis_config.graphs
        ).execute()