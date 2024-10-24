import asyncio
from pipelearn.synaptic.synaptic.ipc import IPCConfig, IPCUrlConfig
from pipelearn.synaptic.synaptic.state import State
import json
from datetime import datetime
from pipelearn.annotator.grounding_dino_annotator import GroundingDinoAnnotator
from pipelearn.annotator.base_annotator import DataAnnotator
import os
from pipelearn.learning_components.yolo_learning_component import YOLOLearningComponent
import yaml
from pipelearn.data.learning_component_data import LearningComponentData, ModelType
from pipelearn.utils.yolo_utils import YOLOUtils

ANNOTATION_LOG_FILE = './logs/gd_annotator.json'

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

async def data_publisher_agent(data_publisher_config):
    backend_kwargs = data_publisher_config['backend_kwargs']
    ipc_settings = IPCConfig(
        shared_memory_size_kb=backend_kwargs['shared_memory_size_kb'],
        url=[IPCUrlConfig(**url) for url in backend_kwargs['url']],
        ipc_kind=backend_kwargs['ipc_kind'],
        serializer=backend_kwargs['serializer']
    )

    publisher_backend_kwargs = {
        "settings": ipc_settings,
        "root_key": backend_kwargs["root_key"]
    }

    dir_source = data_publisher_config['data_source']

    # Simulating data ingestion 
    with State(**publisher_backend_kwargs) as state:
        # Ideally should be in a loop that listens to data source changes and publishes batch info
       while True:
            timestamp = datetime.now().isoformat()
            state.data_batch = {
                "dir_path": dir_source,
                "timestamp": timestamp
            }
            state.test_var = "test_var"
            print(f"Data Publisher: Published batch {dir_source} at {timestamp}")
            await asyncio.sleep(5)


async def annotator_agent(annotator_config):
    backend_kwargs = annotator_config['backend_kwargs']
    ipc_settings = IPCConfig(
        shared_memory_size_kb=backend_kwargs['shared_memory_size_kb'],
        url=[IPCUrlConfig(**url) for url in backend_kwargs['url']],
        ipc_kind=backend_kwargs['ipc_kind'],
        serializer=backend_kwargs['serializer']
    )

    annotator_backend_kwargs = {
        "settings": ipc_settings,
        "root_key": backend_kwargs["root_key"]
    }

    annotator_type = annotator_config["type"]
    labels = annotator_config["labels"]
    device = annotator_config['device']

    if annotator_type == 'grounding_dino':
        annotator: DataAnnotator = GroundingDinoAnnotator(log_file=ANNOTATION_LOG_FILE, device=device)
    else:
        raise ValueError(f"Unknown annotator type {annotator_type}")

    with State(**annotator_backend_kwargs) as state:
        while True:
            if state and hasattr(state, "data_batch"):
                dir_path = state.data_batch["dir_path"]
                timestamp = state.data_batch["timestamp"]
                
                train_dir_path = os.path.join(dir_path, "train")
                val_dir_path = os.path.join(dir_path, "val")
                test_dir_path = os.path.join(dir_path, "test")

                if not train_dir_path or not test_dir_path or not val_dir_path or not labels:
                    raise ValueError("Invalid datasource format. The annotator expects the data source to contain images in the images dir within train, val and test subdirectories")
                
                if annotator.is_batch_annotated(batch_id=dir_path):
                    print(f"Annotator: Batch {dir_path} has already been annotated. Skipping...")
                else:
                    print(f"Annotator: Annotating batch {dir_path} ...")
                    
                    directories = {
                        'train': train_dir_path,
                        'test': test_dir_path,
                        'val': val_dir_path
                    }

                    detections = {}
                    for dataset_type, directory_path in directories.items():
                        detections[dataset_type] = []
                        print(f"Processing {dataset_type} data from {directory_path}...")
                        for image_file in os.listdir(directory_path):
                            image_path = os.path.join(directory_path, image_file)
                            # Annotate the image using Grounding Dino Annotator
                            results = annotator.detect_and_annotate(image=image_path, labels=labels)
                            detections[dataset_type].append({
                                'image_path': image_path,
                                'annotations': results
                            })
                    
                    for dataset_type, detection_list in detections.items():
                        for detection in detection_list:
                            image_path = detection['image_path']
                            annotations = detection['annotations']
                            
                            # Create the output directory by trimming train_dir_path and replacing 'images' with 'labels'
                            output_dir = directories[dataset_type].replace('images', 'labels')
                            
                            # Ensure the output directory exists
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # Convert and save YOLO annotations
                            yolo_annotations = YOLOUtils.convert_to_yolo_format(annotations, image_path)
                            YOLOUtils.save_yolo_annotations(yolo_annotations, image_path, output_dir)

                    root_dir = os.path.dirname(os.path.dirname(train_dir_path))  # Extracts './dataset'

                    # YAML data
                    yaml_data = {
                        'path': f"{root_dir}",
                        'train': 'train/images',
                        'val': 'val/images',
                        'test': 'test/images',
                        'nc': len(labels),
                        'names': labels  # Directly pass the list of labels
                    }

                    # Path to save the YAML file in the root directory of `train_dir_path`
                    yaml_file_path = os.path.join(root_dir, 'data.yaml')
                    
                    # Write the YAML data to file without converting the list to a string
                    with open(yaml_file_path, 'w') as yaml_file:
                        yaml.dump(yaml_data, yaml_file, sort_keys=False, default_flow_style=None)
                        
                    annotator.add_log_entry(dir_path, timestamp)  # Log the annotation as completed
            
                state.annotated_data_batch = {
                    "dir_path": dir_path,
                    "timestamp": timestamp,
                }
            else:
                print("Annotator: No data batch available.")
            
            await asyncio.sleep(2)  # Poll every 2 seconds to check for new data

async def yolo_training_agent(yolo_training_config):
    backend_kwargs = yolo_training_config['backend_kwargs']
    ipc_settings = IPCConfig(
        shared_memory_size_kb=backend_kwargs['shared_memory_size_kb'],
        url=[IPCUrlConfig(**url) for url in backend_kwargs['url']],
        ipc_kind=backend_kwargs['ipc_kind'],
        serializer=backend_kwargs['serializer']
    )

    yolo_agent_backend_kwargs = {
        "settings": ipc_settings,
        "root_key": backend_kwargs["root_key"]
    }

    train_epochs = yolo_training_config['train_epochs']
    device = yolo_training_config['device']

    yolo_learning_component = YOLOLearningComponent()

    with State(**yolo_agent_backend_kwargs) as state:
        while True:
            if state and hasattr(state, "annotated_data_batch"):
                dir_path = state.annotated_data_batch["dir_path"]

                timestamp = datetime.now().isoformat()
                model_ckpt_path = os.path.join(yolo_training_config['model_ckpt_dir'], timestamp)

                yolo_learning_component.train(
                    data=LearningComponentData(
                        data_path=dir_path, 
                        model_type=ModelType.YOLO),
                    epochs=train_epochs,
                    device=device,
                    model_ckpt_path=model_ckpt_path
                )

                



async def run_pipeline(config):
    await asyncio.gather(
        data_publisher_agent(config['data_publisher_config']),
        annotator_agent(config['annotator_config']),
    )

if __name__ == '__main__':
    config = load_config('./configs/pipeline.json')
    asyncio.run(run_pipeline(config))