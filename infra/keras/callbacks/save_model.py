from keras import callbacks

from domain.services.model_storage_service import ModelStorageService


class TrainedModelSaveRemote(callbacks.Callback):
    def __init__(
            self,
            model_storage: ModelStorageService,
            filename: str,
            start_epoch: int
    ):
        super().__init__()
        self.model_storage = model_storage
        self.filename = filename
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch_index: int, logs=None):
        epoch = self.start_epoch + epoch_index
        epoch_suffix = f'_epoch-{epoch}'
        new_filename = self.filename + epoch_suffix

        try:
            file_id = self.model_storage.save(self.filename, new_filename)
            print(f"\n\nSaved model! Name: '{new_filename} | Id: {file_id}'")

        except Exception as error:
            print(error)
            print(f"\n\nError on save keras model: '{error}'")
            exit(1)
