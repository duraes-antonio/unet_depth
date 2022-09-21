import os
import shutil
from pathlib import Path

from domain.services.blob_storage_service import BlobStorageService
from domain.services.model_storage_service import ModelStorageService


class ModelStorageServiceGoogleDrive(ModelStorageService):

    def __init__(self, storage_service: BlobStorageService):
        self.storage_service = storage_service

    def save(self, file_path: str) -> str:
        current_path = Path(file_path)

        zip_filename = current_path.name
        shutil.make_archive(zip_filename, 'zip', file_path)

        zip_complete_path = f'{zip_filename}.zip'
        file_id = self.storage_service.save(zip_complete_path)

        os.remove(zip_complete_path)
        return file_id

    def recover(self, file_id: str) -> None:
        file_basic = self.storage_service.download(file_id)

        if file_basic:
            print(file_basic)
            shutil.unpack_archive(file_basic['name'])
