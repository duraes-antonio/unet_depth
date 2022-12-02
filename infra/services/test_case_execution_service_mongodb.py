import os
from datetime import datetime
from typing import Optional

import pymongo
from dotenv import load_dotenv
from pymongo import MongoClient

from domain.models.test_case.test_case_execution_history import TestCaseExecutionHistory
from domain.services.test_case_execution_service import TestCaseExecutionService

load_dotenv()


class TestCaseExecutionServiceMongoDB(TestCaseExecutionService):

    def __init__(self, db_name):
        self.db_client = MongoClient(os.environ['DATABASE_URL'])
        self.db = self.db_client[db_name]
        self.collection = self.db['execution_history']

    def save(self, result: TestCaseExecutionHistory) -> TestCaseExecutionHistory:
        result['created_at'] = datetime.utcnow()
        result_insert = self.collection.insert_one(result)
        result['id'] = result_insert.inserted_id
        return result

    def get_last_execution(self, test_case_id: str) -> Optional[TestCaseExecutionHistory]:
        test_case: TestCaseExecutionHistory = self.collection.find_one(
            {'test_case_id': test_case_id},
            sort=[('created_at', pymongo.DESCENDING)]
        )
        test_case['id'] = test_case['_id']
        return test_case
