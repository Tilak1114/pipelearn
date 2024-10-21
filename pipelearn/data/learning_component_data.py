from pydantic import BaseModel

from enum import Enum

class ModelType(Enum):
    YOLO = "yolo"
    OTHER = "other"

class LearningComponentData(BaseModel):
    """
    A Pydantic model to represent data required by the learning component.
    
    Attributes:
        data_path: The path to the dataset's configuration file.
        model_type: The type of model this data is used for (e.g., YOLO, etc.).
    """
    data_path: str
    model_type: ModelType