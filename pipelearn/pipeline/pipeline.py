from typing import List
from pipelearn.ingestion_components.grounding_dino_ingestion import GroundingDINOIngestion
from pipelearn.learning_components.yolo_learning_component import YOLOLearningComponent
from pipelearn.types.types import IngestionType, LearningType
from pipelearn.utils.yolo_utils import YOLOUtils
import os
import yaml

class Pipeline:
    """
    Pipeline that manages the ingestion and learning components.
    """

    def __init__(self, ingestion_type: IngestionType, learning_type: LearningType):
        """
        Initializes the pipeline with the given ingestion and learning types.
        
        Args:
            ingestion_type: The type of ingestion component to use (from IngestionType enum).
            learning_type: The type of learning component to use (from LearningType enum).
        """
        self.ingestion_component = self._initialize_ingestion_component(ingestion_type)
        self.learning_component = self._initialize_learning_component(learning_type)
        self.learning_type = learning_type

    def _initialize_ingestion_component(self, ingestion_type: IngestionType):
        """
        Initialize the appropriate ingestion component based on the ingestion type.
        
        Args:
            ingestion_type: The ingestion type from the IngestionType enum.
        
        Returns:
            The appropriate ingestion component.
        """
        if ingestion_type == IngestionType.GROUNDING_DINO:
            return GroundingDINOIngestion()
        else:
            raise ValueError(f"Ingestion type {ingestion_type} is not supported.")

    def _initialize_learning_component(self, learning_type: LearningType):
        """
        Initialize the appropriate learning component based on the learning type.
        
        Args:
            learning_type: The learning type from the LearningType enum.
        
        Returns:
            The appropriate learning component.
        """
        if learning_type == LearningType.YOLO:
            return YOLOLearningComponent()
        else:
            raise ValueError(f"Learning type {learning_type} is not supported.")

    def execute(self, 
                train_dir_path: str, 
                test_dir_path: str, 
                val_dir_path: str, 
                labels: List[str], 
                should_train: bool,
                train_epochs=10,
                device='cuda',
                model_ckpt_path: str='./ckpt/yolo_model.pt',
                ) -> None:

        # Call get_data to perform detection
        detections = self.ingestion_component.get_data(
            train_dir_path=train_dir_path,
            test_dir_path=test_dir_path,
            val_dir_path=val_dir_path,
            labels=labels
        )

        # Define dataset directories
        directories = {
            'train': train_dir_path,
            'test': test_dir_path,
            'val': val_dir_path
        }

        # Save the detections in YOLO format if the learning type is YOLO
        if self.learning_type == LearningType.YOLO:
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

        # Extract the root directory from train_dir_path (assuming it's './dataset/train/images')
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

        # Process the data and train/evaluate the model
        preprocessed_data = self.ingestion_component.process(saved_detections_or_yaml=yaml_file_path)

        if should_train:
             # Ensure that the directory for model_ckpt_path exists
            ckpt_dir = os.path.dirname(model_ckpt_path)
            os.makedirs(ckpt_dir, exist_ok=True)

            self.learning_component.train(preprocessed_data, 
                                          epochs=train_epochs, 
                                          device=device, 
                                          model_ckpt_path=model_ckpt_path)
        else:
            accuracy = self.learning_component.evaluate(preprocessed_data)
            print(f"Model Accuracy: {accuracy}")
            
