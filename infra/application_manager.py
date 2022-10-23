from typing import Tuple

from tensorflow import keras

from domain.models.data.data_generator import NyuV2Generator
from domain.models.test_case.test_case import TestCaseState, TestCase
from domain.models.test_case.test_case_execution_history import TestCaseExecutionHistory
from domain.services.blob_storage_service import BlobStorageService
from domain.services.model_storage_service import ModelStorageService
from domain.services.test_case_execution_service import TestCaseExecutionService
from domain.services.test_case_service import TestCaseService
from infra.keras.consts.metrics import all_metrics, metrics_custom_object
from infra.keras.loss import depth_loss
from infra.keras.model import build_model
from infra.keras.stages.callbacks import build_callbacks, get_model_name
from infra.keras.stages.evaluation import evaluate
from infra.util.dataset import load_nyu_train_paths, read_nyu_csv
from infra.util.output import print_test_case


class ApplicationManager:
    training_generator: NyuV2Generator
    validation_generator: NyuV2Generator
    test_generator: NyuV2Generator
    model: keras.Model

    def __init__(
            self,
            blob_storage: BlobStorageService,
            model_storage: ModelStorageService,
            execution_service: TestCaseExecutionService,
            test_case_service: TestCaseService,
            size: Tuple[int, int] = (256, 256),
            epochs: int = 20,
    ):
        self.__blob_storage__ = blob_storage
        self.__model_storage__ = model_storage
        self.__execution_service__ = execution_service
        self.__test_case_service__ = test_case_service
        self.__image_size__ = size
        self.__max_epochs__ = epochs

    def __get_trained_model__(self, model_id: str, model_name: str):
        self.__model_storage__.recover(model_id)
        return keras.models.load_model(model_name, metrics_custom_object)

    def __train__(self, test_case: TestCase, last_execution: TestCaseExecutionHistory):
        callbacks = build_callbacks(
            test_case, last_execution, self.__blob_storage__, self.__model_storage__,
            self.__execution_service__, self.__test_case_service__
        )
        last_epoch = last_execution['last_epoch'] if last_execution else 0
        remaining_epochs = self.__max_epochs__ - last_epoch
        self.model.fit(
            self.training_generator, validation_data=self.validation_generator,
            callbacks=callbacks, epochs=remaining_epochs
        )

    def __test__(self, model_name: str):
        evaluate(self.model, self.test_generator, self.__blob_storage__, model_name + '.csv')

    def prepare_train_data(
            self, csv_train_path: str, batch_size=8, shuffle=False, seed=42,
            dataset_usage: float = 1
    ) -> 'ApplicationManager':
        partition = load_nyu_train_paths(
            csv_train_path, val_percent=0.3, seed=seed, dataset_usage_percent=dataset_usage
        )
        self.training_generator = NyuV2Generator(
            partition['train'], batch_size, shuffle=shuffle, seed=seed,
            image_size=self.__image_size__
        )
        self.validation_generator = NyuV2Generator(
            partition['validation'], batch_size, shuffle=shuffle, seed=seed,
            image_size=self.__image_size__
        )
        return self

    def prepare_test_data(
            self, csv_test_path: str, batch_size=8, shuffle=False, seed=42,
    ) -> 'ApplicationManager':
        test_path_pairs = read_nyu_csv(csv_test_path)
        self.test_generator = NyuV2Generator(
            test_path_pairs, batch_size, shuffle, seed, self.__image_size__
        )
        return self

    def run(self) -> None:
        test_case = self.__test_case_service__.get_first_available()

        while test_case:
            print_test_case(test_case)
            test_case_id = test_case['id']

            self.__test_case_service__.update_state(
                test_case['id'], TestCaseState.Busy
            )
            print(f'Caso de teste marcado como ocupado! ID: {test_case_id}')

            # Buscar última execução do caso de teste
            last_execution = self.__execution_service__.get_last_execution(
                test_case['id']
            )
            print(f"Última execução do casos de teste: {last_execution['id'] if last_execution else None}")

            model_name = get_model_name(test_case)
            width, height = self.__image_size__
            input_shape = (width, height, 3)

            # Buscar o blob do último modelo atualizado
            if last_execution and last_execution['model_id']:
                model_id = last_execution['model_id']
                model_name = last_execution['model_name']
                print(f"""\nModelo treinado encontrado:\nID:   {model_id}\nNome: {model_name}\n""")
                self.model = self.__get_trained_model__(model_id, model_name)

            else:
                print('Iniciado: build do modelo')
                self.model = build_model(test_case, input_shape)
                print('Finalizado: build do modelo')

                optimizer = str(test_case['optimizer'].value).lower()

                print('Iniciado: compilação do modelo')
                self.model.compile(loss=depth_loss, metrics=all_metrics, optimizer=optimizer)
                print('Finalizado compilação do modelo')

            print('Iniciado: treinamento')
            self.__train__(test_case, last_execution)
            print('Finalizado: treinamento')

            print('Iniciado: teste')
            self.__test__(model_name)
            print('Finalizado: teste')

            self.__test_case_service__.update_state(test_case['id'], TestCaseState.Done)
            print(f"Caso de teste finalizado! ID {test_case_id}")

            test_case = self.__test_case_service__.get_first_available()

        print('Não há casos de testes para executar!')
