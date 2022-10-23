import os
import shutil
from pathlib import Path
from typing import Optional

from domain.services.blob_storage_service import BlobStorageService
from domain.services.model_storage_service import ModelStorageService


class ModelStorageServiceGoogleDrive(ModelStorageService):

    def __init__(self, storage_service: BlobStorageService):
        self.storage_service = storage_service

    def save(self, file_path: str, new_filename: Optional[str] = None) -> str:
        current_path = Path(file_path)

        zip_filename = new_filename or current_path.name
        shutil.make_archive(zip_filename, 'zip', file_path)

        zip_complete_path = f'{zip_filename}.zip'
        file_id = self.storage_service.save(zip_complete_path)

        os.remove(zip_complete_path)
        return file_id

    def remove(self, file_id: str) -> None:
        file_basic = self.storage_service.remove(file_id)

        if file_basic:
            shutil.unpack_archive(file_basic['name'], file_basic['name'].replace('.zip', ''))

    def recover(self, file_id: str) -> None:
        file_basic = self.storage_service.download(file_id)

        if file_basic:
            shutil.unpack_archive(file_basic['name'], file_basic['name'].replace('.zip', ''))
