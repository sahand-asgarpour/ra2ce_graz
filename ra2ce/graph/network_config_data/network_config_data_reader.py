from configparser import ConfigParser
from pathlib import Path
from typing import Union
from ra2ce.common.configuration.ini_configuration_reader_protocol import (
    IniConfigurationReaderProtocol,
)

from ra2ce.graph.network_config_data.network_config_data import (
    CleanupSection,
    HazardSection,
    IsolationSection,
    NetworkConfigData,
    NetworkSection,
    OriginsDestinationsSection,
    ProjectSection,
)


class NetworkConfigDataReader(IniConfigurationReaderProtocol):
    _parser: ConfigParser

    def __init__(self) -> None:
        self._parser = ConfigParser(
            inline_comment_prefixes="#",
            converters={"list": lambda x: [x.strip() for x in x.split(",")]},
        )

    def read(self, file_to_parse: Path) -> NetworkConfigData:
        self._parser.read(file_to_parse)
        self._remove_none_values()

        _parent_dir = file_to_parse.parent

        _config_data = NetworkConfigData(
            input_path=_parent_dir.joinpath("input"),
            static_path=_parent_dir.joinpath("static"),
            output_path=_parent_dir.joinpath("output"),
            **self._get_sections(),
        )
        self._correct_paths(_config_data)
        return _config_data

    def _correct_paths(self, config_data: NetworkConfigData) -> None:
        """
        This method is created because we support defining Path properties with just their filename, but no relative or absolute references.
        These need to be done later on by combining the `input`, `static` or `output` paths. To avoid extra logic all over the solution,
        we can correct these paths here.

        Args:
            config_data (NetworkConfigData): The configuration data whose paths need to be corrected.
        """

        def _select_to_correct(path_value: Union[list[Path], Path]) -> bool:
            if not path_value:
                return False

            if isinstance(path_value, list):
                return _select_to_correct(path_value[0])
            return not path_value.exists()

        # Relative to network directory.
        _network_directory = config_data.static_path.joinpath("network")
        if _select_to_correct(config_data.origins_destinations.origins):
            config_data.origins_destinations.origins = _network_directory.joinpath(
                config_data.origins_destinations.origins
            )

        if _select_to_correct(config_data.origins_destinations.destinations):
            config_data.origins_destinations.destinations = _network_directory.joinpath(
                config_data.origins_destinations.destinations
            )

        # Relative to hazard directory.
        _hazard_directory = config_data.static_path.joinpath("hazard")
        if _select_to_correct(config_data.hazard.hazard_map):
            config_data.hazard.hazard_map = list(
                map(
                    lambda x: _hazard_directory.joinpath(x),
                    config_data.hazard.hazard_map,
                )
            )

    def _get_str_as_path(self, str_value: Union[str, Path]) -> Path:
        if str_value and not isinstance(str_value, Path):
            return Path(str_value)
        return str_value

    def _remove_none_values(self) -> None:
        # Remove 'None' from values, replace them with empty strings
        for _section in self._parser.sections():
            _keys_with_none = [
                k for k, v in self._parser[_section].items() if v == "None"
            ]
            for _key_with_none in _keys_with_none:
                self._parser[_section].pop(_key_with_none)

    def _get_sections(self) -> dict:
        return {
            "project": self.get_project_section(),
            "network": self.get_network_section(),
            "origins_destinations": self.get_origins_destinations_section(),
            "isolation": self.get_isolation_section(),
            "hazard": self.get_hazard_section(),
            "cleanup": self.get_cleanup_section(),
        }

    def get_project_section(self) -> ProjectSection:
        return ProjectSection(**self._parser["project"])

    def get_network_section(self) -> NetworkSection:
        _section = "network"
        _network_section = NetworkSection(**self._parser[_section])
        _network_section.directed = self._parser.getboolean(
            _section, "directed", fallback=_network_section.directed
        )
        _network_section.save_shp = self._parser.getboolean(
            _section, "save_shp", fallback=_network_section.save_shp
        )
        _network_section.road_types = self._parser.getlist(
            _section, "road_types", fallback=_network_section.road_types
        )
        return _network_section

    def get_origins_destinations_section(self) -> OriginsDestinationsSection:
        _section = "origins_destinations"
        _od_section = OriginsDestinationsSection(**self._parser[_section])
        _od_section.origin_out_fraction = self._parser.getint(
            _section, "origin_out_fraction", fallback=_od_section.origin_out_fraction
        )
        _od_section.origins = self._get_str_as_path(_od_section.origins)
        _od_section.destinations = self._get_str_as_path(_od_section.destinations)
        _od_section.region = self._get_str_as_path(_od_section.region)
        return _od_section

    def get_isolation_section(self) -> IsolationSection:
        _section = "isolation"
        if _section not in self._parser:
            return IsolationSection()
        return IsolationSection(**self._parser[_section])

    def get_hazard_section(self) -> HazardSection:
        _section = "hazard"
        if _section not in self._parser:
            return HazardSection()
        _hazard_section = HazardSection(**self._parser[_section])
        _hazard_section.hazard_map = list(
            map(
                self._get_str_as_path,
                self._parser.getlist(
                    _section, "hazard_map", fallback=_hazard_section.hazard_map
                ),
            )
        )
        _hazard_section.hazard_field_name = self._parser.getlist(
            _section, "hazard_field_name", fallback=_hazard_section.hazard_field_name
        )
        return _hazard_section

    def get_cleanup_section(self) -> CleanupSection:
        _section = "cleanup"

        _cleanup_section = CleanupSection()
        _cleanup_section.snapping_threshold = self._parser.getboolean(
            _section,
            "snapping_threshold",
            fallback=_cleanup_section.snapping_threshold,
        )
        _cleanup_section.pruning_threshold = self._parser.getboolean(
            _section,
            "pruning_threshold",
            fallback=_cleanup_section.pruning_threshold,
        )
        _cleanup_section.segmentation_length = self._parser.getfloat(
            _section,
            "segmentation_length",
            fallback=_cleanup_section.segmentation_length,
        )
        _cleanup_section.merge_lines = self._parser.getboolean(
            _section, "merge_lines", fallback=_cleanup_section.merge_lines
        )
        _cleanup_section.cut_at_intersections = self._parser.getboolean(
            _section,
            "cut_at_intersections",
            fallback=_cleanup_section.cut_at_intersections,
        )
        return _cleanup_section
