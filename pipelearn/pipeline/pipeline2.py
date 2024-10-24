import asyncio
import json
from pipelearn.agents.data_source_agent import DataSourceAgent
from pipelearn.agents.annotator_agent import AnnotatorAgent

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


async def run_pipeline(config):
    data_source_agent = DataSourceAgent()
    annotator_agent = AnnotatorAgent()
    await asyncio.gather(
        data_source_agent.async_act(config['data_publisher_config']),
        annotator_agent.async_act(config['annotator_config']),
    )

if __name__ == '__main__':
    config = load_config('./configs/pipeline.json')
    asyncio.run(run_pipeline(config))