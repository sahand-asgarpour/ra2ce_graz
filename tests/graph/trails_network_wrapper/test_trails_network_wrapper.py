from ra2ce.graph.network_config_data.network_config_data import NetworkConfigData
from ra2ce.graph.network_wrapper_protocol import NetworkWrapperProtocol
from ra2ce.graph.trails_network_wrapper.trails_network_wrapper import TrailsNetworkWrapper


class TestTrailsNetworkWrapper:

    def test_initialize(self):
        # 1. Define test data.
        _config_data = NetworkConfigData()

        # 2. Create wrapper.
        _wrapper = TrailsNetworkWrapper(_config_data)

        # 3. Verify expectations.
        assert isinstance(_wrapper, TrailsNetworkWrapper)
        assert isinstance(_wrapper, NetworkWrapperProtocol)