import pandas
from tensorflow import keras

from domain.models.data.data_generator import NyuV2Generator
from domain.services.blob_storage_service import BlobStorageService


def __save_csv__(blob_storage: BlobStorageService, filename: str):
    try:
        blob_storage.save(filename)
        print(f"\n\nSaved CSV log: '{filename}'")

    except Exception as error:
        print(error)
        print(f"\n\nError on save CSV log: '{error}'")


def evaluate(
        model: keras.Model,
        data_generator: NyuV2Generator,
        blob_storage_service: BlobStorageService,
        csv_name: str
) -> None:
    metric_values = model.evaluate(data_generator, batch_size=8)
    metric_names = [
        'loss',
        'abs_rel',
        'sq_rel',
        'rmse',
        'rmse_log',
        'log_10',
        'threshold_1',
        'threshold_2',
        'threshold_3'
    ]
    metrics_dict = {name: [value] for name, value in zip(metric_names, metric_values)}
    dataframe = pandas.DataFrame(metrics_dict)

    filename = csv_name.replace('.csv', '_test.csv')
    dataframe.to_csv(filename, header=True)
    __save_csv__(blob_storage_service, filename)
