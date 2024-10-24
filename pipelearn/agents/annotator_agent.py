from mbodied.agents.agent import Agent
from pipelearn.synaptic.synaptic.ipc import IPCConfig, IPCUrlConfig
from pipelearn.synaptic.synaptic.state import State
from datetime import datetime
import os
import asyncio
from pipelearn.data.data_source_data import DataSourceData
from embdata.sense.world import World, Image


class AnnotatorAgent(Agent):
    def __init__(self,):
        pass

    def act(self, *args, **kwargs):
        raise Exception("Not supported")

    async def async_act(self, config):
        backend_kwargs = config['backend_kwargs']
        ipc_settings = IPCConfig(
            shared_memory_size_kb=backend_kwargs['shared_memory_size_kb'],
            url=[IPCUrlConfig(**url) for url in backend_kwargs['url']],
            ipc_kind=backend_kwargs['ipc_kind'],
            serializer=backend_kwargs['serializer']
        )

        backend_kwargs = {
            "settings": ipc_settings,
            "root_key": backend_kwargs["root_key"]
        }

        subscriber_attr = config['subscribe_from']

        with State(**backend_kwargs) as state:
            while True:
                if state and hasattr(state, subscriber_attr):
                    data_source_data = DataSourceData(
                        **state.__getattr__(subscriber_attr)
                    )
                    print(data_source_data.world.image.path)

                await asyncio.sleep(2)
