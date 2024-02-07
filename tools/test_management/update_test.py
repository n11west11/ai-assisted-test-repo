import os
from typing import List, Optional, Type

import chainlit as cl
from chainlit import user
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pymongo import MongoClient
from regex import F
from bson import ObjectId

from tools.test_management.models import MongoTestModel, StepModel


def _update_test(test: MongoTestModel) -> str:
    """Updates a test to the database"""

    user = cl.user_session.get("user")
    if user:
        test.creator = user.identifier
    mongo_connection_string = os.getenv("MONGO_URI")
    mongo_database = os.getenv("MONGO_DATABASE_NAME")
    collection_name = os.getenv("COLLECTION_NAME")

    client = MongoClient(mongo_connection_string)

    db = client[mongo_database]
    collection = db[collection_name]

    test = test.dict()
    id = test.pop("id")
    
    res = collection.update_one({"_id": ObjectId(id)}, {"$set": test})
    if res.modified_count == 0:
        return "Test Not Found"
    
    return "Test Updated"


async def _aupdate_test(test: MongoTestModel) -> str:
    """Updates a test to the database"""
    return _update_test(test)



class UpdateTestTool(BaseTool):
    name = "update_test"
    description = "Updates test steps to a database, does not actually execute/run the test"
    args_schema: Type[MongoTestModel] = MongoTestModel

    def _run(
            self,
            id: str,
            application: str,
            feature: str,
            creator: str,
            tags: List[str],
            steps: List[StepModel],
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        test = MongoTestModel(
            id=id,
            application=application,
            feature=feature,
            creator=creator,
            tags=tags,
            steps=steps,
        )
        return _update_test(test)

    async def _arun(
            self,
            id,
            application: str,
            feature: str,
            creator: str,
            tags: List[str],
            steps: List[StepModel],
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        test = MongoTestModel(
            id=id,
            application=application,
            feature=feature,
            creator=creator,
            tags=tags,
            steps=steps,
        )
        return await _aupdate_test(test)


update_test_tool = UpdateTestTool(handle_tool_error=True)