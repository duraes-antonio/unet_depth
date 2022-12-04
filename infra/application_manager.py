import gc
from typing import Optional

import tensorflow as tf
from tensorflow import keras

from domain.models.data.data_generator import NyuV2Generator
from domain.models.test_case.test_case import TestCaseState, TestCase, TestCaseConfig
from domain.models.test_case.test_case_execution_history import TestCaseExecutionHistory
from domain.services.blob_storage_service import BlobStorageService
from domain.services.model_storage_service import ModelStorageService
from domain.services.results_service import ResultService
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
            result_service: ResultService,
            epochs: int = 20,
            seed: Optional[int] = None,
            batch_size: int = 4,
            dataset_usage: float = 1
    ):
        self.__blob_storage__ = blob_storage
        self.__model_storage__ = model_storage
        self.__execution_service__ = execution_service
        self.__test_case_service__ = test_case_service
        self.__result_service__ = result_service
        self.__max_epochs__ = epochs
        self.__seed__ = seed
        self.__batch_size__ = batch_size
        self.__dataset_usage__ = dataset_usage

    def __get_trained_model__(self, model_id: str, model_name: str):
        self.__model_storage__.recover(model_id)
        return keras.models.load_model(model_name, metrics_custom_object)

    def __train__(self, csv_train_path: str, test_case: TestCase, last_execution: TestCaseExecutionHistory):
        seed = self.__seed__
        batch_size = self.__batch_size__
        dataset_usage = self.__dataset_usage__
        shuffle = False
        test_config = test_case['config']
        size = test_config['size']
        size = size, size
        read_mode = test_config['read_mode']

        partition = load_nyu_train_paths(csv_train_path, 0.3, seed, dataset_usage)
        training_gen = NyuV2Generator(partition['train'], batch_size, shuffle, seed, size, read_mode=read_mode)
        validation_gen = NyuV2Generator(partition['validation'], batch_size, shuffle, seed, size, read_mode=read_mode)

        callbacks = build_callbacks(
            test_case, last_execution, self.__blob_storage__, self.__model_storage__,
            self.__execution_service__, self.__test_case_service__, self.__result_service__,
            test_config
        )
        last_epoch = last_execution['epoch'] if last_execution else 0
        remaining_epochs = self.__max_epochs__ - last_epoch
        self.model.fit(
            training_gen, validation_data=validation_gen,
            callbacks=callbacks, epochs=remaining_epochs
        )

    def __test__(self, model_name: str, csv_test_path: str, test_config: TestCaseConfig):
        test_path_pairs = read_nyu_csv(csv_test_path)
        size = test_config['size'], test_config['size']
        read_mode = test_config['read_mode']

        self.test_generator = NyuV2Generator(
            test_path_pairs, batch_size=self.__batch_size__, shuffle=False,
            seed=self.__seed__, image_size=size, read_mode=read_mode
        )
        evaluate(self.model, self.test_generator, self.__blob_storage__, model_name + '.csv')

    def run(self, train_data_path: str, test_data_path: str) -> None:
        test_case = self.__test_case_service__.get_first_available()

        try:
            while test_case is not None:
                print_test_case(test_case)
                test_case_id = test_case['id']
                test_config = test_case['config']

                self.__test_case_service__.update_state(
                    test_case['id'], TestCaseState.Busy
                )
                print(f'Caso de teste marcado como ocupado! ID: {test_case_id}')

                # Buscar última execução do caso de teste
                last_execution = self.__execution_service__.get_last_execution(
                    test_case['id']
                )

                if last_execution:
                    print(f"Última execução do casos de teste: {last_execution['id']}")

                model_name = get_model_name(test_config)

                # Buscar o blob do último modelo atualizado
                if last_execution:
                    model_id = last_execution['model_id']
                    model_name = last_execution['model_name']
                    print(f"""\nModelo treinado encontrado:\nID:   {model_id}\nNome: {model_name}\n""")
                    self.model = self.__get_trained_model__(model_id, model_name)

                else:
                    print('Iniciado: build do modelo')
                    self.model = build_model(test_config)
                    print('Finalizado: build do modelo')

                    optimizer = str(test_config['optimizer'].value).lower()

                    print('Iniciado: compilação do modelo')
                    self.model.compile(loss=depth_loss, metrics=all_metrics, optimizer=optimizer)
                    print('Finalizado compilação do modelo')

                print('Iniciado: treinamento')
                self.__train__(train_data_path, test_case, last_execution)
                print('Finalizado: treinamento')

                print('Iniciado: teste')
                self.__test__(model_name, test_data_path, test_config)
                print('Finalizado: teste')

                self.__test_case_service__.update_state(test_case['id'], TestCaseState.Done)
                print(f"Caso de teste finalizado! ID {test_case_id}")

                test_case = self.__test_case_service__.get_first_available()
                del self.model
                tf.keras.backend.clear_session()
                gc.collect()

        except Exception as ex:
            raise ex

        print('Não há casos de testes para executar!')
