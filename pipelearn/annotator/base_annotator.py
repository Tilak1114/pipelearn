from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import List, Dict
from dataclasses import dataclass

class BoundingBox(BaseModel):
    """
    Model for a bounding box annotation.

    Attributes:
        xmin (int): The minimum x-coordinate of the bounding box.
        ymin (int): The minimum y-coordinate of the bounding box.
        xmax (int): The maximum x-coordinate of the bounding box.
        ymax (int): The maximum y-coordinate of the bounding box.
    """
    xmin: int = Field(..., description="The minimum x-coordinate of the bounding box.")
    ymin: int = Field(..., description="The minimum y-coordinate of the bounding box.")
    xmax: int = Field(..., description="The maximum x-coordinate of the bounding box.")
    ymax: int = Field(..., description="The maximum y-coordinate of the bounding box.")

@dataclass
class DetectionResult:
    """
    Container for a detection result.
    
    Attributes:
        score (float): The confidence score of the detection.
        label (str): The predicted class label.
        label_class_id (int): The ID of the predicted class.
        box (BoundingBox): The bounding box coordinates.
    """
    score: float
    label: str
    label_class_id: int
    box: BoundingBox
    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        """
        Create a DetectionResult instance from a dictionary.

        Args:
            detection_dict (Dict): A dictionary containing the detection result.
                It should have the following structure:
                    {
                        'score': float,
                        'label': str,
                        'label_class_id': int,
                        'box': {
                            'xmin': int,
                            'ymin': int,
                            'xmax': int,
                            'ymax': int
                        }
                    }

        Returns:
            DetectionResult: A DetectionResult instance.
        """
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   label_class_id=detection_dict['label_class_id'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

class DataAnnotator(ABC):
    """
    Base class for a data annotator.

    Any model used for automatic annotation should implement this interface.

    Attributes:
        None

    Methods:
        detect_and_annotate: Detect objects and annotate the image.
    """

    @abstractmethod
    def detect_and_annotate(self, image: str, labels: List[str], **kwargs):
        """
        Detect objects and annotate the image.

        Args:
            image (str): The path to the image to annotate.
            labels (List[str]): A list of class labels to use for annotation.
            **kwargs: Additional keyword arguments for custom implementation.

        Returns:
            None
        """
        pass
