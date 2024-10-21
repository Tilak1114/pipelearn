from pipelearn.pipeline.pipeline import Pipeline
from pipelearn.types.types import IngestionType, LearningType

if __name__ == '__main__':
    # Define the pipeline using enums for the types
    ingestion_type = IngestionType.GROUNDING_DINO
    learning_type = LearningType.YOLO
    
    # Create the pipeline
    pipeline = Pipeline(ingestion_type, learning_type)

    dataset_dir = './dataset'
    
    # Set directory paths and labels
    train_dir_path = f"{dataset_dir}/train/images"
    test_dir_path = f"{dataset_dir}/test/images"
    val_dir_path = f"{dataset_dir}/val/images"
    labels = ["remote", "spoon"]
    
    # Train the model and save YOLO annotations
    pipeline.execute(train_dir_path=train_dir_path, 
                     test_dir_path=test_dir_path, 
                     val_dir_path=val_dir_path, 
                     labels=labels, 
                     should_train=True,)