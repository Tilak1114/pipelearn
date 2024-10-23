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
                 device='cuda',
                 log_file='./logs/gd_annotator.json'):
        """
        Initializes the GroundingDinoAnnotator class.

        Args:
            detector_id (str): The ID of the object detection model to use. Defaults to "IDEA-Research/grounding-dino-base".
        """
        super().__init__()
        detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-base"
        self.object_detector = pipeline(
            model=detector_id, task="zero-shot-object-detection", device=device)
        self.log_file = log_file
    
    def is_batch_annotated(self, batch_id):
        log = self.load_annotation_log()
        return any(entry['batch_id'] == batch_id for entry in log)
    
    def load_annotation_log(self):
        """Load the annotation log as a list of log entries."""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                try:
                    return json.load(f)  # Load the list of log entries
                except json.JSONDecodeError:
                    return []  # Return an empty list if the file is empty or corrupted
        else:
            return []

    def add_log_entry(self, batch_id, timestamp):
        """Add a new log entry to the log file as part of a list."""
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)  # Create the directory if it doesn't exist

        # Load existing log entries
        log_entries = self.load_annotation_log()

        # Add the new log entry to the list
        log_dict = {
            "batch_id": batch_id,
            "timestamp": timestamp
        }
        log_entries.append(log_dict)  # Append new log entry to the list

        # Write the updated list back to the file
        with open(self.log_file, 'w') as f:  # Overwrite the file with the updated list
            json.dump(log_entries, f, indent=4)  # indent for pretty printing

        print(f"Logged annotation for batch {log_dict['batch_id']} at {log_dict['timestamp']}")

    def detect_and_annotate(self,
                            image: str,
                            labels: List[str],
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


