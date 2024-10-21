from pipelearn.data.learning_component_data import LearningComponentData, ModelType
from pipelearn.annotator.grounding_dino_annotator import GroundingDinoAnnotator
import os

class GroundingDINOIngestion:
    """
    Data ingestion class that uses the Grounding Dino Annotator for object detection 
    and converts data to LearningComponentData.
    """

    def __init__(self):
        """
        Initialize the GroundingDINOIngestion with just the annotator.
        """
        self.annotator = GroundingDinoAnnotator()
    
    def get_data(self, **kwargs):
        """
        Perform detection and annotation on all the directories and return detection results.
        """
        train_dir_path = kwargs.get("train_dir_path")
        test_dir_path = kwargs.get("test_dir_path")
        val_dir_path = kwargs.get("val_dir_path")
        labels = kwargs.get("labels")
        
        if not train_dir_path or not test_dir_path or not val_dir_path or not labels:
            raise ValueError("train_dir_path, test_dir_path, val_dir_path, and labels must be provided.")

        # Process all directories
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
                results = self.annotator.detect_and_annotate(image=image_path, labels=labels)
                detections[dataset_type].append({
                    'image_path': image_path,
                    'annotations': results
                })
        return detections

    def process(self, saved_detections_or_yaml: str) -> LearningComponentData:
        """
        Process the data, which takes saved detections or a YAML configuration file path.
        
        Args:
            saved_detections_or_yaml: Path to saved detections in YOLO format or the YAML config.
        
        Returns:
            LearningComponentData: Data required for the learning component.
        """
        # If you have the path to a YAML config file, or the path to the saved detections, 
        # you can use it here to prepare LearningComponentData.
        return LearningComponentData(data_path=saved_detections_or_yaml, model_type=ModelType.YOLO)