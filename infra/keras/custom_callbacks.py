from keras import Model
from tensorflow.python.keras.callbacks import Callback

from domain.services.blob_storage_service import BlobStorageService
from domain.services.model_storage_service import ModelStorageService


class CSVResultsSave(Callback):
    def __init__(self, blob_storage_service: BlobStorageService = None, csv_log_path: str = None):
        self.blob_storage = blob_storage_service
        self.csv_log_path = csv_log_path
        super(CSVResultsSave, self).__init__()

    def on_epoch_end(self, epoch: int, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

        try:
            self.blob_service.save(self.csv_log_path)
            print(f"Saved CSV log: '{self.csv_log_path}'")

        except Exception as error:
            print(error)
            print(f"Error on save CSV log: '{error}'")


class TrainedModelSave(Callback):
    def __init__(self, model_storage: ModelStorageService, filename: str):
        self.model_storage = model_storage
        self.filename = filename
        super(TrainedModelSave, self).__init__()

    def on_epoch_end(self, epoch: int, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

        try:
            model: Model = self.model
            model.save(model.save(self.filename))
            file_id = self.model_storage.save(self.filename)
            print(f"Saved model! Name: '{self.filename} | Id: {file_id}'")

        except Exception as error:
            print(error)
            print(f"Error on save keras model: '{error}'")
