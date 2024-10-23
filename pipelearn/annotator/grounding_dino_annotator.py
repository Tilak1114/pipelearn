from pipelearn.annotator.base_annotator import DataAnnotator, DetectionResult
from typing import List, Optional
import torch
from transformers import pipeline
import cv2
import os
import json

class GroundingDinoAnnotator(DataAnnotator):
    """
    A DataAnnotator class that uses the Google Grounding DINO model for object detection.

    Attributes:
        detector_id (str): The ID of the object detection model to use. Defaults to "IDEA-Research/grounding-dino-base".
        device (str): The device to run the model on, either "cuda" or "cpu".

    Methods:
        detect_and_annotate: Performs object detection and returns a list of DetectionResult objects.
    """
    def __init__(self, 
                 detector_id: Optional[str] = None, 
                 log_file='./logs/gd_annotator.json'):
        """
        Initializes the GroundingDinoAnnotator class.

        Args:
            detector_id (str): The ID of the object detection model to use. Defaults to "IDEA-Research/grounding-dino-base".
        """
        super().__init__()
        detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-base"
        self.object_detector = pipeline(
            model=detector_id, task="zero-shot-object-detection")
        self.log_file = log_file
    
    def is_batch_annotated(self, batch_id):
        log = self.load_annotation_log()
        return any(entry['batch_id'] == batch_id for entry in log)
    
    def load_annotation_log(self,):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        else:
            return []
    
    def add_log_entry(self, batch_id, timestamp):
        log_dict = {
            "batch_id":batch_id,
            "timestamp":timestamp
        }
        with open(self.log_file, 'a') as f:
            json.dump(log_dict, f)
            f.write('\n') 
            
        print(f"Logged annotation for batch {log_dict['batch_id']} at {log_dict['timestamp']}")

    def detect_and_annotate(self,
                            image: str,
                            labels: List[str],
                            device='cuda',
                            **kwargs):
        """
        Performs object detection on the input image and returns a list of DetectionResult objects.
        
        Args:
            image (str): The path to the input image.
            labels (List[str]): A list of labels to detect. Will be converted to end with "." if not already.
            kwargs: Additional keyword arguments, including "threshold" which defaults to 0.3.

        Returns:
            List[DetectionResult]: A list of DetectionResult objects containing the detected objects and their properties.
        """
        
        threshold = kwargs.get("threshold", 0.3)

        labels = [label if label.endswith(
            ".") else label+"." for label in labels]

        self.object_detector = self.object_detector.to(device=device)

        results = self.object_detector(
            image,  candidate_labels=labels, threshold=threshold)

        updated_results = []
        for result in results:
            result['label_class_id'] = labels.index(result['label'])
            result = DetectionResult.from_dict(result)
            updated_results.append(result)
    
        return updated_results
    
    def plot_detections(self, image_path: str, 
                        detection_results: List[DetectionResult], 
                        save_file_name: str):
        """
        Plot the detections by drawing bounding boxes on the image and saving it.
        
        Args:
            image_path: Path to the input image file.
            detection_results: List of DetectionResult objects.
            save_file_name: Name of the output file to save the image with drawn bounding boxes.
        """
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        
        # Iterate through the detection results and draw the bounding boxes
        for result in detection_results:
            # Get bounding box coordinates
            bbox = result.box
            
            # Draw the bounding box (red color, thickness=2)
            cv2.rectangle(image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 0, 255), 2)
            
            # Prepare label and score
            label = f"{result.label}: {result.score:.2f}"
            
            # Get text size and calculate the position for text background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_position = (bbox.xmin, bbox.ymin - 10)
            background_top_left = (bbox.xmin, bbox.ymin - text_height - 10)
            background_bottom_right = (bbox.xmin + text_width, bbox.ymin)
            
            # Draw a white rectangle behind the text for readability
            cv2.rectangle(image, background_top_left, background_bottom_right, (255, 255, 255), -1)
            
            # Draw the text (label and score)
            cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save the image with the drawn bounding boxes and text
        cv2.imwrite(save_file_name, image)


