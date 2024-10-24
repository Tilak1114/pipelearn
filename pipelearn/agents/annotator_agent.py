from mbodied.agents.agent import Agent
from pipelearn.synaptic.synaptic.ipc import IPCConfig, IPCUrlConfig
from pipelearn.synaptic.synaptic.state import State
import asyncio
from pipelearn.data.data_source_data import DataSourceData
from embdata.sense.world import World, Image


class AnnotatorAgent(Agent):
    def __init__(self):
        self.processing_queue = asyncio.Queue()  # Asynchronous queue for processing data
        self.processed_items = set()  # Set to track already processed items

    def act(self, *args, **kwargs):
        raise Exception("Not supported")

    async def subscribe_to_stream(self, config):
        """
        This method subscribes to the pub-sub stream and adds new data to the processing queue.
        """
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

        subscribe_attr = config['subscribe_from']

        with State(**backend_kwargs) as state:
            while True:
                # Check if the state has received new data on the subscription topic
                if hasattr(state, subscribe_attr) and state.__getattr__(subscribe_attr):
                    # Deserialize the published data
                    data_source_data = DataSourceData.model_validate_json(
                        state.__getattr__(subscribe_attr)
                    )

                    if data_source_data.uid not in self.processed_items:
                        # Add the new data to the processing queue and mark it as processed
                        await self.processing_queue.put(data_source_data)
                        self.processed_items.add(data_source_data.uid)
                        print(f"Annotator: Added {data_source_data.world.image.path} to the processing queue")

                    await asyncio.sleep(0)
                    
    async def process_queue(self):
        """
        This method processes data from the queue.
        """
        print("processing started")
        while True:
            # Get the next item from the processing queue (blocks until an item is available)
            data_source_data = await self.processing_queue.get()

            try:
                # Process the data (e.g., perform annotation)
                print(f"Annotator: Processing {data_source_data.world.image.path}")
                # Perform your annotation logic here...

            except Exception as e:
                print(f"Error processing data: {e}")

            # Indicate that the item has been processed
            self.processing_queue.task_done()

            await asyncio.sleep(2)  # Simulate some processing time

    async def async_act(self, config):
        # Start both the subscriber and the queue processor in parallel
        await asyncio.gather(
            self.subscribe_to_stream(config),
            self.process_queue()
        )
