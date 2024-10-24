from mbodied.agents.agent import Agent
from pipelearn.synaptic.synaptic.ipc import IPCConfig, IPCUrlConfig
from pipelearn.synaptic.synaptic.state import State
from datetime import datetime
import os
import asyncio
from pipelearn.data.data_source_data import DataSourceData
from embdata.sense.world import World, Image


class DataSourceAgent(Agent):
    def __init__(self):
        # Initialize data sources
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

        dir_source = config['data_source']

        async def image_repo(dir_source):
            """A generator function that yields one image path at a time."""
            for image_file in os.listdir(dir_source):
                image_path = os.path.join(dir_source, image_file)
                # Ensure it's a valid image file
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    yield image_path
                else:
                    print(f"Skipping non-image file: {image_file}")

        publish_attr = config["publish_to"]

        if publish_attr == "":
            raise ValueError(
                "The data source doesnt have a publish state. Please set publish_to attribute in your config")

        with State(**backend_kwargs) as state:
            while True:
                async for image_path in image_repo(dir_source):
                    try:
                        # Create DataSourceData and publish it
                        timestamp = datetime.now().isoformat()

                        data_source_data = DataSourceData(
                            world=World(
                                image=Image(image_path)
                            ),
                            uid=f"data_source_{image_path}",
                        )

                        state.__setattr__(publish_attr, data_source_data.model_dump_json())
                        print(
                            f"Data Publisher: Published {image_path} at {timestamp}")

                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                    finally:
                        await asyncio.sleep(0)

                await asyncio.sleep(5)
