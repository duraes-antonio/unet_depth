from datetime import datetime
from typing import List, Optional

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, Callback

from domain.models.test_case.test_case import TestCase, TestCaseConfig
from domain.models.test_case.test_case_execution_history import TestCaseExecutionHistory
from domain.services.blob_storage_service import BlobStorageService
from domain.services.model_storage_service import ModelStorageService
from domain.services.results_service import ResultService
from domain.services.test_case_execution_service import TestCaseExecutionService
from domain.services.test_case_service import TestCaseService
from infra.keras.callbacks.prepare_save_csv import PrepareSaveCSV
from infra.keras.callbacks.save_execution import ExecutionSave


def get_model_name(test_case_config: TestCaseConfig) -> str:
    """
    Obtém o nome do modelo (ex.: 'attention-unet_adam_resnet-101_imagenet-0') a partir de um caso de teste
    :param test_case_config:
    :return: Nome completo do modelo
    """
    network = test_case_config['network'].value
    optimizer = str(test_case_config['optimizer'].value).lower()
    backbone = str(test_case_config['backbone'].value).lower()
    use_image_net = int(test_case_config['use_imagenet_weights'])
    min_filter = test_case_config['filter_min']
    max_filter = test_case_config['filter_max']
    size = test_case_config['size']
    read_mode = test_case_config['read_mode'].value
    network_config = f'n-{network}_o-{optimizer}_b-{backbone}_imagenet-{use_image_net}_f-{min_filter}-{max_filter}'
    another_params = f's-{size}x{size}_r-{read_mode}_batch-4'
    return f'{network_config}_{another_params}'


def build_callbacks(
        test_case: TestCase,
        last_execution: Optional[TestCaseExecutionHistory],
        blob_storage: BlobStorageService,
        model_storage: ModelStorageService,
        execution_service: TestCaseExecutionService,
        test_case_service: TestCaseService,
        result_service: ResultService,
        network_config: TestCaseConfig
) -> List[Callback]:
    epoch = last_execution['epoch'] + 1 if last_execution else 1

    trained_model_name = get_model_name(network_config)
    csv_log_name = f'{trained_model_name}.csv'

    tensorboard_log_path = "logs/fit/"
    tensorboard_current_log_path = tensorboard_log_path + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_checkpoint = ModelCheckpoint(
        filepath=trained_model_name,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    save_execution = ExecutionSave(
        model_storage, execution_service, test_case_service,
        trained_model_name, test_case['id'], epoch
    )
    prepare_and_save_csv = PrepareSaveCSV(
        blob_storage, result_service, csv_log_name, test_case['id']
    )
    return [
        EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True),
        prepare_and_save_csv,
        model_checkpoint,
        save_execution,
        TensorBoard(log_dir=tensorboard_current_log_path, histogram_freq=1),
    ]
