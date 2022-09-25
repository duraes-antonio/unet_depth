from abc import ABC, abstractmethod
from typing import Optional

from domain.models.named_entity import NamedEntity


class BlobStorageService(ABC):

    @abstractmethod
    def save(self, current_file_path: str, new_file_name: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def download(self, file_id: str) -> Optional[NamedEntity]:
        pass

    @abstractmethod
    def download_last(self) -> Optional[NamedEntity]:
        pass
