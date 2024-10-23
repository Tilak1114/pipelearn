
import numpy as np
import asyncio
from pipelearn.synaptic.synaptic.ipc import IPCConfig, IPCUrlConfig
from pipelearn.synaptic.synaptic.state import State

# Define backend configuration
thermostat_backend_kwargs = {
    "settings": IPCConfig(
        shared_memory_size_kb=1024,
        url=[
            IPCUrlConfig(connect="tcp/0.0.0.0:1234")
        ], 
        ipc_kind="zenoh",
        serializer="msgpack",
    ),
    "root_key": "example/therm",
}

decider_backend_kwargs = {
    "settings": IPCConfig(
        shared_memory_size_kb=1024,
        url=[
            IPCUrlConfig(listen="tcp/0.0.0.0:1234"),
        ], 
        ipc_kind="zenoh",
        serializer="msgpack",
    ),
    "root_key": "example/therm",
}

async def decider():
    with State(**decider_backend_kwargs) as state:
        while True:
            current_temp = state.current_temp
            # state.should_cool = True if current_temp > 30 else False
            print(f"Decider : Current temp is {current_temp}")
            await asyncio.sleep(1)

async def thermostat():
    with State(**thermostat_backend_kwargs) as state:
        # while True:
        #     state.current_temp = np.random.uniform(20, 100)
        #     print(f"Thermo: {state.current_temp}")
        #     await asyncio.sleep(2)
        state.current_temp = np.random.uniform(20, 100)
        print(f"Thermo: {state.current_temp}")

async def start():
    await asyncio.gather(thermostat(), decider())

if __name__ == "__main__":
    # Load the image using the EmbData Image class
    asyncio.run(start())
