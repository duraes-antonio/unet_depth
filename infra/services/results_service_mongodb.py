import os
from typing import List

import pymongo
from bson import ObjectId
from pymongo import MongoClient

from domain.consts.metrics import TRAIN_VAL_METRICS_NAME
from domain.models.train_result import TrainResult, FinalTrainResult
from domain.services.results_service import ResultService


class ResultServiceMongoDB(ResultService):

    def __init__(self, db_name):
        self.db_client = MongoClient(os.environ['DATABASE_URL'])
        self.db = self.db_client[db_name]
        self.collection = self.db['execution_history']

    def get_by_test(self, test_case_id: str) -> List[TrainResult]:
        executions_cursor = self.collection.find(
            {'test_case_id': ObjectId(test_case_id)},
            projection={'result': True, '_id': False},
            sort=[('created_at', pymongo.ASCENDING)]
        )
        return [item['result'] for item in executions_cursor]

    def get_and_unify(self, test_case_id: str) -> FinalTrainResult:
        executions = self.get_by_test(test_case_id)
        dict_metrics: FinalTrainResult = {}
        key: TRAIN_VAL_METRICS_NAME
        metric_keys: List[TRAIN_VAL_METRICS_NAME] = [key for key in TrainResult.__annotations__]

        for key in metric_keys:
            dict_metrics[key] = []

        for key in metric_keys:
            dict_metrics[key] = [execution[key] for execution in executions]

        return dict_metrics
