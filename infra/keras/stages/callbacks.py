from datetime import datetime
from typing import List, Optional

from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint, Callback

from domain.models.test_case import TestCase
from domain.models.test_case_execution_history import TestCaseExecutionHistory
from domain.services.blob_storage_service import BlobStorageService
from domain.services.model_storage_service import ModelStorageService
from domain.services.test_case_execution_service import TestCaseExecutionService
from domain.services.test_case_service import TestCaseService
from infra.keras.callbacks import CSVResultsSave, TrainedModelSaveRemote


def get_model_name(test_case: TestCase) -> str:
    """
    Obtém o nome do modelo (ex.: 'attention-unet_epoch-15_adam_resnet-101_imagenet-0') a partir de um caso de teste
    :param test_case: Casos de teste com as informações da execução
    :return: Nome completo do modelo
    """
    network = test_case['network'].value
    optimizer = str(test_case['optimizer'].value).lower()
    backbone = str(test_case['backbone'].value).lower()
    use_image_net = int(test_case['use_imagenet_weights'])
    return f'{network}_{optimizer}_{backbone}_imagenet-{use_image_net}'


def build_callbacks(
        test_case: TestCase,
        last_execution: Optional[TestCaseExecutionHistory],
        blob_storage: BlobStorageService,
        model_storage: ModelStorageService,
        execution_service: TestCaseExecutionService,
        test_case_service: TestCaseService,
) -> List[Callback]:
    epoch = last_execution['last_epoch'] + 1 if last_execution else 1

    trained_model_name = get_model_name(test_case)
    csv_log_name = f'{trained_model_name}.csv'

    tensorboard_log_path = "logs/fit/"
    tensorboard_current_log_path = tensorboard_log_path + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_checkpoint_callback = ModelCheckpoint(
        filepath=trained_model_name,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    save_model_remote_callback = TrainedModelSaveRemote(
        model_storage,
        execution_service,
        test_case_service,
        trained_model_name,
        test_case['id'],
        epoch
    )
    return [
        EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True),
        CSVLogger(csv_log_name),
        CSVResultsSave(blob_storage, csv_log_name, epoch),
        model_checkpoint_callback,
        save_model_remote_callback,
        TensorBoard(log_dir=tensorboard_current_log_path, histogram_freq=1),
    ]
