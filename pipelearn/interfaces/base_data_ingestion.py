from abc import ABC, abstractmethod
from pipelearn.data.learning_component_data import LearningComponentData

class IDataIngestion(ABC):
    """
    Interface for Data Ingestion components.
    """

    @abstractmethod
    def process(self, **kwargs) -> LearningComponentData:
        """
        Preprocess the data and convert it to LearningComponentData for the learning component.
        
        Args:
            kwargs: Additional keyword arguments for preprocessing (e.g., thresholds).
            
        Returns:
            LearningComponentData: The preprocessed data for the learning component.
        """
        pass