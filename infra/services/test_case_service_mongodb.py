import os
from datetime import datetime, timedelta
from typing import Optional, Iterable

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

from domain.models.network import Networks, KerasBackbone, Optimizers
from domain.models.test_case.test_case import TestCase, TestCaseState, TestCaseConfig, InputReadMode
from domain.services.test_case_service import TestCaseService

load_dotenv()


class TestCaseServiceMongoDB(TestCaseService):

    def __init__(self, db_name):
        self.db_client = MongoClient(os.environ['DATABASE_URL'])
        self.db = self.db_client[db_name]
        self.db_name = db_name
        self.collection = self.db['test_cases']

    def get(self, _id: str) -> TestCase:
        test_case = self.collection.find_one(ObjectId(_id))
        return self.__dict_to_object__(test_case)

    def get_first_available(self) -> Optional[TestCase]:
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        query = {
            '$or': [
                {'state': TestCaseState.Available.value},
                {
                    '$or': [
                        {'last_modified': None},
                        {
                            '$and': [
                                {
                                    'last_modified': {
                                        '$lte': one_hour_ago
                                    },
                                },
                                {'state': TestCaseState.Busy.value}
                            ]
                        },
                    ]
                }
            ],
        }
        test_case: TestCase = self.collection.find_one(query)
        return self.__dict_to_object__(test_case)

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
        initial_state = TestCaseState.Available
        test_case_configs: Iterable[dict] = [
            TestCaseConfig(
                network=net.value, backbone=back.value, optimizer=opt.value,
                use_imagenet_weights=use_imagenet, size=size,
                filter_min=f_min, filter_max=f_max, read_mode=read.value
            )
            for net in networks
            for back in backbones
            for opt in optimizers
            for read in read_modes
            for use_imagenet in use_imagenet_weights
            for size in sizes
            for f_min in filters_min
            for f_max in filters_max
        ]

        test_cases: Iterable[TestCase] = [
            TestCase(
                created_at=datetime.utcnow(), state=initial_state.value,
                config=config, last_modified=None
            )
            for config in test_case_configs
        ]
        self.collection.insert_many(test_cases)

    def update_state(self, _id: str, state: TestCaseState) -> TestCase:
        query = {'_id': _id}
        update = {
            'last_modified': datetime.utcnow(),
            'state': state.value,
        }
        updated: TestCase = self.collection.find_one_and_update(query, {'$set': update})
        updated['state'] = state
        updated['last_modified'] = update['last_modified']
        return self.__dict_to_object__(updated)

    def remove_all(self):
        self.collection.delete_many({})

    @staticmethod
    def __dict_to_object__(dict_instance: TestCase) -> TestCase:
        if not dict_instance:
            return dict_instance

        dict_instance['last_modified'] = dict_instance['last_modified'] if 'last_modified' in dict_instance else None
        config_in = dict_instance['config']
        config_out = TestCaseConfig(
            network=Networks(config_in['network']),
            backbone=KerasBackbone(config_in['backbone']),
            optimizer=Optimizers(config_in['optimizer']),
            read_mode=InputReadMode(config_in['read_mode']),
            filter_min=config_in['filter_min'],
            filter_max=config_in['filter_max'],
            size=config_in['size'],
            use_imagenet_weights=config_in['use_imagenet_weights'],
        )
        return TestCase(
            id=dict_instance['_id'],
            config=config_out,
            state=TestCaseState(dict_instance['state']),
            created_at=dict_instance['created_at'],
            last_modified=dict_instance['last_modified'],
        )
