from keras import callbacks

from domain.services.blob_storage_service import BlobStorageService


class CSVResultsSave(callbacks.Callback):
    def __init__(
            self,
            blob_storage_service: BlobStorageService,
            csv_log_path: str,
            start_epoch: int
    ):
        super().__init__()
        self.blob_storage = blob_storage_service
        self.csv_log_path = csv_log_path
        self.start_epoch = start_epoch

    def on_epoch_end(self, execution_epoch: int, logs=None):
        csv_new_name = self.csv_log_path

        try:
            self.blob_storage.save(self.csv_log_path, csv_new_name)
            print(f"\n\nSaved CSV log: '{csv_new_name}'")

        except Exception as error:
            print(error)
            print(f"\n\nError on save CSV log: '{error}'")
