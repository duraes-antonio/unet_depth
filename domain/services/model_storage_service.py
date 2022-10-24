from abc import ABC, abstractmethod
from typing import Optional


class ModelStorageService(ABC):

    @abstractmethod
    def save(self, file_path: str, new_filename: Optional[str] = None) -> str:
        """
        Prepare and store a trained model
        :param new_filename: File name after stored
        :param file_path: Path of the file to be stored
        :return: Stored file id
        """
        pass

    @abstractmethod
    def remove(self, file_id: str) -> None:
        """
        Remove a previously saved model
        :param file_id: ID of model for delete
        """
        pass

    @abstractmethod
    def recover(self, file_id: str) -> None:
        """
        Retrieves and prepares a stored model to its original format
        :param file_id: Stored file id
        """
        pass
