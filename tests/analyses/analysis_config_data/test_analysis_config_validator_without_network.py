from pathlib import Path
from typing import Optional

import pytest

from ra2ce.analyses.analysis_config_data.analysis_config_data import (
    AnalysisConfigDataWithoutNetwork,
)
from ra2ce.analyses.analysis_config_data.analysis_config_data_validator_without_network import (
    AnalysisConfigDataValidatorWithoutNetwork,
)

from ra2ce.common.validation.validation_report import ValidationReport
from tests import test_data, test_results


class TestAnalysisConfigDataValidatorWithoutNetwork:
    def _validate_from_dict(self, dict_values: dict) -> ValidationReport:
        _test_config_data = AnalysisConfigDataWithoutNetwork.from_dict(dict_values)
        _validator = AnalysisConfigDataValidatorWithoutNetwork(_test_config_data)
        return _validator.validate()

    def test_validate_with_required_headers(self):
        # 1. Define test data.
        _output_test_dir = test_data / "acceptance_test_data"
        assert _output_test_dir.is_dir()

        # 2. Run test.
        _test_config_data = {"project": {}, "output": _output_test_dir}
        _report = self._validate_from_dict(_test_config_data)

        # 3. Verify expectations.
        assert _report.is_valid()

    @pytest.mark.parametrize(
        "output_entry",
        [
            pytest.param(dict(), id="No output given"),
            pytest.param(dict(output=None), id="Output is none"),
            pytest.param(
                dict(output=(test_data / "not_a_path.ini")), id="Not a valid path."
            ),
        ],
    )
    def test_validate_withoutput_output_reports_error(
        self, output_entry: Optional[Path]
    ):
        # 1. Define test data.
        _output_dir = output_entry.get("output", None)
        _expected_error = f"The configuration file 'network.ini' is not found at {_output_dir}.Please make sure to name your network settings file 'network.ini'."
        _test_config_data = {"project": {}}
        _test_config_data.update(output_entry)

        # 2. Run test.
        _report = self._validate_from_dict(_test_config_data)

        # 3. Verify expectations.
        assert not _report.is_valid()
        assert _expected_error in _report._errors

    def test_validate_files_no_path_value_list_returns_empty_report(self):
        # 1. Define test data.
        _validator = AnalysisConfigDataValidatorWithoutNetwork(
            AnalysisConfigDataWithoutNetwork()
        )

        # 2. Run test.
        _report = _validator._validate_files("does_not_matter", [])

        # 3. Verify expectations.
        assert isinstance(_report, ValidationReport)
        assert _report.is_valid()

    def test_validate_files_with_non_existing_files_reports_error(self):
        # 1. Define test data.
        _validator = AnalysisConfigDataValidatorWithoutNetwork(
            AnalysisConfigDataWithoutNetwork()
        )
        _test_file = test_data / "not_a_valid_file.txt"

        # 2. Run test.
        _report = _validator._validate_files("dummy_header", [_test_file])

        # 3. Verify expectations.
        assert isinstance(_report, ValidationReport)
        assert not _report.is_valid()
        assert len(_report._errors) == 2

    def test_validate_road_types_no_road_type_returns_empty_report(self):
        # 1. Define test data.
        _validator = AnalysisConfigDataValidatorWithoutNetwork(
            AnalysisConfigDataWithoutNetwork()
        )

        # 2. Run test.
        _report = _validator._validate_road_types("")

        # 3. Verify expectations.
        assert isinstance(_report, ValidationReport)
        assert _report.is_valid()

    def test_validate_road_types_with_unexpected_road_type_reports_error(self):
        # 1. Define test data.
        _validator = AnalysisConfigDataValidatorWithoutNetwork(
            AnalysisConfigDataWithoutNetwork()
        )
        _road_type = "not a valid road type"

        # 2. Run test.
        _report = _validator._validate_road_types(_road_type)

        # 3. Verify expectations.
        assert isinstance(_report, ValidationReport)
        assert not _report.is_valid()
        assert len(_report._errors) == 1

    def _validate_headers_from_dict(
        self, dict_values: dict, required_headers: list[str]
    ) -> ValidationReport:
        _test_config_data = AnalysisConfigDataWithoutNetwork.from_dict(dict_values)
        _validator = AnalysisConfigDataValidatorWithoutNetwork(_test_config_data)
        return _validator._validate_headers(required_headers)

    def test_validate_headers_fails_when_missing_expected_header(self):
        # 1. Define test data.
        _test_config_data = {}
        _missing_header = "Deltares"
        _expected_err = f"Property [ {_missing_header} ] is not configured. Add property [ {_missing_header} ] to the *.ini file. "

        # 2. Run test.
        _report = self._validate_headers_from_dict(
            _test_config_data, required_headers=[_missing_header]
        )

        # 3. Verify final expectations.
        assert not _report.is_valid()
        assert _expected_err in _report._errors

    def test_validate_headers_fails_when_wrong_file_value(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _required_header = "file_header"
        _test_config_data = {
            "root_path": test_results,
            "project": {"name": request.node.name},
            _required_header: {"polygon": [Path("sth")]},
        }

        # 2. Run test.
        _report = self._validate_headers_from_dict(
            _test_config_data, required_headers=[_required_header]
        )

        # 3. Verify final expectations.
        assert not _report.is_valid()
        assert len(_report._errors) == 3

    def test_validate_headers_fails_when_wrong_road_type(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _required_header = "road_header"
        _test_config_data = {
            "root_path": test_results,
            "project": {"name": request.node.name},
            _required_header: {"road_types": "not a valid road type"},
        }

        # 2. Run test.
        _report = self._validate_headers_from_dict(
            _test_config_data, required_headers=[_required_header]
        )

        # 3. Verify final expectations.
        assert not _report.is_valid()
        assert len(_report._errors) == 2

    def test_validate_headers_fails_when_unexpected_value(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _required_header = "unexpected_value"
        _test_config_data = {
            "root_path": test_results,
            "project": {"name": request.node.name},
            _required_header: {"network_type": "unmapped_value"},
        }

        # 2. Run test.
        _report = self._validate_headers_from_dict(
            _test_config_data, required_headers=[_required_header]
        )

        # 3. Verify final expectations.
        assert not _report.is_valid()
        assert len(_report._errors) == 2
