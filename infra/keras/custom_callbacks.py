from typing import Optional

from cpuinfo import get_cpu_info
from tensorflow import keras
from tensorflow.python.client import device_lib

from domain.models.test_case_execution_history import TestCaseExecutionHistory
from domain.services.blob_storage_service import BlobStorageService
from domain.services.model_storage_service import ModelStorageService
from domain.services.test_case_execution_service import TestCaseExecutionService


class TestCallback(keras.callbacks.Callback):

    def __init__(self, param):
        self.p = param

    def on_epoch_end(self, epoch, logs=None):
        print(self.p)


class CSVResultsSave(keras.callbacks.Callback):
    def __init__(self, blob_storage_service: BlobStorageService = None, csv_log_path: str = None):
        super().__init__()
        self.blob_storage = blob_storage_service
        self.csv_log_path = csv_log_path

    def on_epoch_end(self, epoch: int, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

        try:
            self.blob_service.save(self.csv_log_path)
            print(f"Saved CSV log: '{self.csv_log_path}'")

        except Exception as error:
            print(error)
            print(f"Error on save CSV log: '{error}'")


class TrainedModelSave(keras.callbacks.Callback):
    def __init__(
            self, model_storage: ModelStorageService,
            execution_service: TestCaseExecutionService,
            filename: str,
            test_case_id: str
    ):
        super().__init__()
        self.model_storage = model_storage
        self.execution_service = execution_service
        self.filename = filename
        self.test_case_id = test_case_id

    def on_epoch_end(self, epoch: int, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

        try:
            model: keras.Model = self.model
            model.save(model.save(self.filename))
            file_id = self.model_storage.save(self.filename)
            print(f"Saved model! Name: '{self.filename} | Id: {file_id}'")

        except Exception as error:
            print(error)
            print(f"Error on save keras model: '{error}'")

        try:
            data: TestCaseExecutionHistory = {
                'test_case_id': self.test_case_id,
                'model_id': file_id,
                'model_name': self.filename,
                'last_epoch': epoch,
                'cpu_description': self.__get_cpu_info__(),
                'gpu_description': self.__get_gpu_info__(),
            }
            self.execution_service.save(data)
            print(f"Execution item saved!")

        except Exception as error:
            print(error)
            print(f"Error on update execution history: '{error}'")

    def __get_cpu_info__(self) -> dict:
        cpu_info = get_cpu_info()
        ignored_keys = {k for k in cpu_info if 'hz_' in k or k == 'flags'}
        cpu_info = {k: cpu_info[k] for k in cpu_info if k not in ignored_keys}
        return cpu_info

    def __get_gpu_info__(self) -> Optional[dict]:
        devices = device_lib.list_local_devices()
        gpus = [device for device in devices if device.device_type == "GPU"]
        return gpus if gpus else None
