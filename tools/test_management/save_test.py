import os
from typing import List, Optional, Type

import chainlit as cl
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pymongo import MongoClient
from regex import F

from tools.test_management.models import StepModel, TestModel


def _save_test(test: TestModel) -> str:
    """Saves a test to the database"""

    user = cl.user_session.get("user")
    if user:
        test.creator = user.identifier
    mongo_connection_string = os.getenv("MONGO_URI")
    mongo_database = os.getenv("MONGO_DATABASE_NAME")
    collection_name = os.getenv("COLLECTION_NAME")

    client = MongoClient(mongo_connection_string)

    db = client[mongo_database]
    collection = db[collection_name]
    
    collection.insert_one(test.dict())
    return "Test saved"


async def _asave_test(test: TestModel) -> str:
    """Saves a test to the database"""
    return _save_test(test)


class SaveTestTool(BaseTool):
    name = "save_test"
    description = "Saves test steps to a database, does not actually execute/run the test"
    args_schema: Type[TestModel] = TestModel

    def _run(
            self,
            application: str,
            feature: str,
            creator: str,
            tags: List[str],
            steps: List[StepModel],
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        test = TestModel(
            application=application,
            feature=feature,
            creator=creator,
            tags=tags,
            steps=steps,
        )
        return _save_test(test)

    async def _arun(
            self,
            application: str,
            feature: str,
            creator: str,
            tags: List[str],
            steps: List[StepModel],
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        test = TestModel(
            application=application,
            feature=feature,
            creator=creator,
            tags=tags,
            steps=steps,
        )
        return await _asave_test(test)


save_test_tool = SaveTestTool(handle_tool_error=True)