import os
from datetime import datetime
from typing import Optional, Iterable, List

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

from domain.models.network import Networks, KerasBackbone, Optimizers
from domain.models.test_case import TestCase, TestCaseState
from domain.services.test_case_service import TestCaseService

load_dotenv()


class TestCaseServiceMongoDB(TestCaseService):

    def __init__(self, db_name):
        self.db_client = MongoClient(os.environ['DATABASE_URL'])
        self.db = self.db_client[db_name]
        self.collection = self.db['test_cases']

    def get(self, _id: str) -> TestCase:
        test_case = self.collection.find_one(ObjectId(_id))
        return self.__dict_to_object__(test_case)

    def get_first_available(self) -> Optional[TestCase]:
        test_case: TestCase = self.collection.find_one({'state': TestCaseState.Available.value})
        return self.__dict_to_object__(test_case)

    def populate(
            self,
            networks: Iterable[Networks],
            backbones: Iterable[KerasBackbone],
            optimizers: Iterable[Optimizers],
    ) -> None:
        use_imagenet_weights = [False, True]
        test_cases: List[TestCase] = []
        initial_state = TestCaseState.Available

        for net in networks:
            for back in backbones:
                for opt in optimizers:
                    for use_imgnet in use_imagenet_weights:
                        test_case = TestCase(
                            network=net.value, backbone=back.value, optimizer=opt.value,
                            created_at=datetime.utcnow(), state=initial_state.value,
                            use_imagenet_weights=use_imgnet,
                        )
                        test_cases.append(test_case)
        self.collection.insert_many(test_cases)

    def save(self, result: TestCase) -> TestCase:
        self.collection.insert_one(result)
        return result

    def remove_all(self):
        self.collection.delete_many({})

    def __dict_to_object__(self, dict_instance: TestCase) -> TestCase:
        if not dict_instance:
            return dict_instance

        dict_instance['last_modified'] = dict_instance['last_modified'] if 'last_modified' in dict_instance else None
        return TestCase(
            network=Networks(dict_instance['network']),
            backbone=KerasBackbone(dict_instance['backbone']),
            optimizer=Optimizers(dict_instance['optimizer']),
            state=TestCaseState(dict_instance['state']),
            created_at=dict_instance['created_at'],
            use_imagenet_weights=dict_instance['use_imagenet_weights'],
            last_modified=dict_instance['last_modified'],
            id=dict_instance['_id'],
        )
