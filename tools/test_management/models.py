from typing import List, Type, Optional
from dns.immutable import Dict

from langchain.pydantic_v1 import BaseModel, Field


class StepModel(BaseModel):
    description: str = Field(..., description="The description of the step")
    tool: str = Field(..., description="The name of the tool to use for the step")
    tool_args: dict = Field(..., description="The arguments to pass to the tool")
    expected_result: str = Field(..., description="The expected result of the step")

    @classmethod
    def schema(cls):
        schema = super().schema()
        schema['properties']['tool_args'] = {
            "title": "Tool Args",
            "type": "object",
            "properties": {}
        }
        return schema
        


class TestModel(BaseModel):
    application: str = Field(..., description="The application that the test is for, ideally this is a url")
    feature: str = Field(..., description="The feature that the test is testing")
    creator: str = Field(..., description="The creator of the test")
    tags: List[str] = Field(..., description="Tags for the test")
    steps: List[StepModel]

    @classmethod
    def schema(cls):
        schema = super().schema()
        schema['properties']['steps'] = {
            "title": "Steps",
            "type": "array",
            "items": StepModel.schema()
        }
        return schema
    


class MongoTestModel(TestModel):
    id: str = Field(..., description="The id of the test in the database")
