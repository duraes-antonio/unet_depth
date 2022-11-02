import pandas
from tensorflow import keras

from domain.services.blob_storage_service import BlobStorageService
from domain.services.results_service import ResultService


class PrepareSaveCSV(keras.callbacks.Callback):
    def __init__(
            self,
            blob_storage_service: BlobStorageService,
            result_service: ResultService,
            csv_name: str,
            test_case_id: str
    ):
        super().__init__()
        self.blob_storage = blob_storage_service
        self.result_service = result_service
        self.test_case_id = test_case_id
        self.csv_name = csv_name

    def on_train_end(self, logs=None):
        metric_values = self.result_service.get_and_unify(self.test_case_id)
        dataframe = pandas.DataFrame(metric_values)
        dataframe.to_csv(self.csv_name, header=True)

        try:
            self.blob_storage.save(self.csv_name)
            print(f"\n\nSaved CSV log: '{self.csv_name}'")

        except Exception as error:
            print(error)
            print(f"\n\nError on save CSV log: '{error}'")
