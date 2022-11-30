from abc import ABC, abstractmethod
from typing import Optional, Iterable

from domain.models.network import Networks, KerasBackbone, Optimizers
from domain.models.test_case.test_case import TestCase, TestCaseState, InputReadMode


class TestCaseService(ABC):

    @abstractmethod
    def get(self, _id: str) -> TestCase:
        pass

    # TODO: Considerar casos de teste sem atualização há de 6h como disponíveis
    @abstractmethod
    def get_first_available(self) -> Optional[TestCase]:
        """
        Retorna o primeiro caso de teste disponível para execução
        :return: Primeiro teste disponível, ou None, se não houver algum disponível
        """

        pass

    @abstractmethod
    def update_state(self, _id: str, state: TestCaseState) -> TestCase:
        pass

    @abstractmethod
    def populate(
            self,
            networks: Iterable[Networks],
            backbones: Iterable[KerasBackbone],
            optimizers: Iterable[Optimizers],
            read_modes: Iterable[InputReadMode],
            use_imagenet_weights: Iterable[bool] = (True, False),
            sizes: Iterable[int] = (256, 512),
            filters_min: Iterable[int] = (64,),
            filters_max: Iterable[int] = (512, 1024),
    ) -> None:
        """
        Gera todas as combinações possíveis de casos de teste e os persiste
        :param use_imagenet_weights:
        :param networks: Conjunto de redes a serem usadas
        :param backbones: Conjunto de backbones a serem usados
        :param optimizers: Conjunto de otimizadores
        :param config: Configuração do caso de teste
        :param read_modes: Modos de leitura da imagem de entrada
        :param sizes: Dimensões a serem consideradas
        :param filters_min: Valor mínimo para os filtros da rede
        :param filters_max: Valor máximo para os filtros da rede
        """
        pass

    @abstractmethod
    def remove_all(self) -> None:
        pass
