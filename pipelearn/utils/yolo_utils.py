import os
from typing import List
from PIL import Image
from pipelearn.annotator.base_annotator import DetectionResult

class YOLOUtils:
    @staticmethod
    def convert_to_yolo_format(annotations: List[DetectionResult], image_path: str) -> List[str]:
        """
        Convert bounding boxes to YOLO format.

        Args:
            annotations: List of DetectionResult objects.
            image_path: Path to the image file.

        Returns:
            A list of YOLO-format strings for each annotation.
        """
        # Load the image to get its dimensions
        image = Image.open(image_path)
        img_width, img_height = image.size

        yolo_annotations = []

        for annotation in annotations:
            bbox = annotation.box
            class_id = annotation.label_class_id

            # Get the center coordinates, width, and height in pixel format
            x_center = (bbox.xmin + bbox.xmax) / 2.0
            y_center = (bbox.ymin + bbox.ymax) / 2.0
            width = bbox.xmax - bbox.xmin
            height = bbox.ymax - bbox.ymin

            # Normalize the values to [0, 1]
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            # Format the annotation in YOLO format: class_id x_center y_center width height
            yolo_format = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_annotations.append(yolo_format)

        return yolo_annotations

    @staticmethod
    def save_yolo_annotations(yolo_annotations: List[str], image_path: str, output_dir: str):
        """
        Save the YOLO annotations to a *.txt file corresponding to the image.

        Args:
            yolo_annotations: List of YOLO-formatted annotation strings.
            image_path: Path to the image file.
            output_dir: Directory to save the annotation file.
        """
        # Get the image file name without the extension
        image_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Create the corresponding *.txt file path
        annotation_file = os.path.join(output_dir, f"{image_filename}.txt")

        # Write the YOLO annotations to the file
        with open(annotation_file, "w") as f:
            f.write("\n".join(yolo_annotations))