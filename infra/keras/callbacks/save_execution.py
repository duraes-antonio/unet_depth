from typing import Optional

from cpuinfo import get_cpu_info
from keras import callbacks
from tensorflow.python.client import device_lib

from domain.models.test_case.test_case import TestCaseState
from domain.models.test_case.test_case_execution_history import TestCaseExecutionHistory
from domain.services.test_case_execution_service import TestCaseExecutionService
from domain.services.test_case_service import TestCaseService


class ExecutionSave(callbacks.Callback):
    def __init__(
            self,
            execution_service: TestCaseExecutionService,
            test_case_service: TestCaseService,
            filename: str,
            test_case_id: str,
            start_epoch: int
    ):
        super().__init__()
        self.execution_service = execution_service
        self.test_case_service = test_case_service
        self.filename = filename
        self.test_case_id = test_case_id
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch_index: int, logs=None):
        epoch = self.start_epoch + epoch_index
        epoch_suffix = f'_epoch-{epoch}'
        new_filename = self.filename + epoch_suffix
        file_id = None

        try:
            data: TestCaseExecutionHistory = {
                'test_case_id': self.test_case_id,
                'model_id': file_id,
                'model_name': new_filename,
                'last_epoch': epoch,
                'cpu_description': str(self.__get_cpu_info__()),
                'gpu_description': str(self.__get_gpu_info__()),
            }
            self.execution_service.save(data)
            self.test_case_service.update_state(self.test_case_id, TestCaseState.Busy)
            print(f"\n\nExecution item saved!")

        except Exception as error:
            print(error)
            print(f"\n\nError on update execution history: '{error}'")
            exit(1)

    def __get_cpu_info__(self) -> dict:
        cpu_info = get_cpu_info()
        ignored_keys = {k for k in cpu_info if 'hz_' in k or k == 'flags'}
        cpu_info = {k: cpu_info[k] for k in cpu_info if k not in ignored_keys}
        return cpu_info

    def __get_gpu_info__(self) -> Optional[dict]:
        devices = device_lib.list_local_devices()
        gpus = [device for device in devices if device.device_type == "GPU"]
        return gpus if gpus else None
