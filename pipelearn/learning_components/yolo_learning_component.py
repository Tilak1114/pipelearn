from ultralytics import YOLOWorld
from pipelearn.data.learning_component_data import LearningComponentData, ModelType
from pipelearn.interfaces.base_learning_component import ILearningComponent

class YOLOLearningComponent(ILearningComponent):
    """
    A learning component that uses the YOLOWorld model to train and evaluate on the LearningComponentData object.
    """

    def __init__(self, model_path: str = "yolov8x-worldv2.pt"):
        """
        Initializes the YOLOLearningComponent with the given model path.
        
        Args:
            model_path: The path to the YOLOWorld model to be used for training and evaluation.
        """
        self.model = YOLOWorld(model_path)  # Load the YOLOWorld model

    def train(self, 
              data: LearningComponentData,
              epochs, 
              device,
              model_ckpt_path: str) -> None:
        """
        Train the YOLO model using the LearningComponentData.
        
        Args:
            data: The LearningComponentData object containing the path to the dataset configuration.
        """
        if data.model_type == ModelType.YOLO:
            # TODO: Change hardcoded imgsz
            results = self.model.train(data=data.data_path, epochs=epochs, imgsz=720, device=device)
            self.model.save(filename=model_ckpt_path)
            print(f"Training complete! Results: {results.results_dict}")
        else:
            raise ValueError("Incompatible data type for YOLO training")

    def evaluate(self, data: LearningComponentData) -> float:
        """
        Evaluate the YOLO model using the LearningComponentData.
        
        Args:
            data: The LearningComponentData object containing the path to the dataset configuration.
        
        Returns:
            A float representing the evaluation metric (e.g., accuracy).
        """
        if data.model_type == ModelType.YOLO:
            metrics = self.model.val()  # Validate the model on the validation set
            print(f"Evaluation metrics: {metrics}")
            return metrics["accuracy"]
        else:
            raise ValueError("Incompatible data type for YOLO evaluation")