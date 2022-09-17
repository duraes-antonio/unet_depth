from abc import ABC, abstractmethod
from typing import Optional, Iterable

from domain.models.network import Networks, KerasBackbone, Optimizers
from domain.models.test_case import TestCase


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
    def save(self, result: TestCase) -> TestCase:
        pass

    @abstractmethod
    def populate(
            self, networks: Iterable[Networks],
            backbones: Iterable[KerasBackbone],
            optimizers: Iterable[Optimizers]
    ) -> None:
        """
        Gera todas as combinações possíveis de casos de teste e os persiste
        :param networks: Conjunto de redes a serem usadas
        :param backbones: Conjunto de backbones a serem usados
        :param optimizers: Conjunto de otimizadores
        """
        pass

    @abstractmethod
    def remove_all(self) -> None:
        pass
