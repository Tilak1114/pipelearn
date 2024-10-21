from abc import ABC, abstractmethod
from typing import Any

class ILearningComponent(ABC):
    """
    Interface for Learning components.
    """

    @abstractmethod
    def train(self, data: Any) -> None:
        """
        Train the model using the provided preprocessed data.

        Args:
            data: The preprocessed data (e.g., World object).
        """
        pass

    @abstractmethod
    def evaluate(self, data: Any) -> float:
        """
        Evaluate the model using the provided preprocessed data.

        Args:
            data: The preprocessed data (e.g., World object).
        
        Returns:
            The evaluation metric (e.g., accuracy).
        """
        pass