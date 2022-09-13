import io
from pathlib import Path
from typing import Generic, TypeVar, Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from domain.models.named_entity import NamedEntity
from domain.services.blob_storage_service import BlobStorageService
from infra.services.blob_storage.google_drive_auth import GoogleDriveTokenManager

T = TypeVar('T')


class GoogleDriveBlobStorageService(BlobStorageService, Generic[T]):
    directory = 'models'

    def __init__(self):
        self.__credentials__ = GoogleDriveTokenManager().load_credentials()
        self.__service__ = build('drive', 'v3', credentials=self.__credentials__)

    def __create_folder__(self, name: str):
        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        files = self.__service__.files()
        file = files.create(body=file_metadata, fields='id').execute()
        print(file)
        return file.download('id')

    def __get_metadata_by_id__(self, file_id: str) -> Optional[NamedEntity]:
        files = self.__service__.files().get(fileId=file_id).execute()

        if not files:
            return None

        return NamedEntity(name=files['name'], id=files['id'])

    def __search_folder__(self, name: str) -> str:
        files = self.__service__.files()
        folders_dict = files.list(
            q=f"name = '{name}' and mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)"
        ).execute()
        folders = folders_dict['files']
        return folders[0]['id'] if folders else None

    def __upload_file__(self, file_path: str, folder_id: str) -> str:
        file_metadata = {
            'name': Path(file_path).name,
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)
        file = self.__service__.files().create(
            body=file_metadata, media_body=media, fields='id'
        ).execute()
        return file['id']

    def __download_file__(self, new_name: str, file_id: str):
        request = self.__service__.files().get_media(fileId=file_id)
        file_io = io.FileIO(new_name, mode='wb')
        downloader = MediaIoBaseDownload(file_io, request)
        done = False

        while not done:
            _, done = downloader.next_chunk()

    def save(self, model_file_path: str) -> str:
        folder_id = self.__search_folder__(self.directory)

        if not folder_id:
            folder_id = self.__create_folder__(self.directory)

        return self.__upload_file__(model_file_path, folder_id)

    def download_last(self) -> Optional[NamedEntity]:
        folder_id = self.__search_folder__(self.directory)

        if not folder_id:
            return None

        files_service = self.__service__.files()
        files_dict = files_service.list(
            q=f"'{folder_id}' in parents",
            orderBy="modifiedTime desc",
            fields="files(id, name)"
        ).execute()

        files = files_dict['files']

        if not files:
            return None

        self.__download_file__(files[0]['name'], files[0]['id'])

    def download(self, file_id: str) -> Optional[NamedEntity]:
        file_metadata: NamedEntity = self.__get_metadata_by_id__(file_id)
        self.__download_file__(file_metadata['name'], file_metadata['id'])
        return file_metadata
