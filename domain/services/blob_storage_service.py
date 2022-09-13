from abc import ABC, abstractmethod
from typing import Optional

from domain.models.named_entity import NamedEntity


class BlobStorageService(ABC):

    @abstractmethod
    def save(self, model_file_path: str) -> str:
        pass

    @abstractmethod
    def download(self, model_id: str) -> Optional[NamedEntity]:
        pass

    @abstractmethod
    def download_last(self) -> Optional[NamedEntity]:
        pass
