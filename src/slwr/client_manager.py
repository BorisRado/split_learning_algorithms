from flwr.common import GetPropertiesIns
from flwr.server.client_manager import SimpleClientManager


class HeterogeneousClientManager(SimpleClientManager):
    def __init__(self) -> None:
        super().__init__()
        self._cid_to_props = {}

    def unregister(self, client):
        super().unregister(client)
        if client.cid in self._cid_to_props:
            del self._cid_to_props[client.cid]

    def get_het_client_properties(self, client_proxy):
        if client_proxy.cid in self._cid_to_props:
            return self._cid_to_props[client_proxy.cid]

        print(f"Getting properties for {client_proxy.cid}")
        get_props_ins = GetPropertiesIns({})
        res = client_proxy.get_properties(get_props_ins, None, None)
        props = res.properties
        props["cid"] = client_proxy.cid
        self._cid_to_props[client_proxy.cid] = props
        print("Got properties: ", props)
        return props
