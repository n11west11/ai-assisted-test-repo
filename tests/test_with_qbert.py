import pytest
import os
from pymongo import MongoClient


@pytest.fixture
def mongo_client():
    mongo_connection_string = os.getenv("MONGO_URI")
    mongo_database = os.getenv("MONGO_DATABASE_NAME")
    collection_name = os.getenv("COLLECTION_NAME")

    client = MongoClient(mongo_connection_string)

    db = client[mongo_database]
    collection = db[collection_name]
    return collection

@pytest.mark.qbert
def test_aware_agent(request, page, mongo_client):
    # first get tags
    tags = request.node.get_closest_marker("tags")
    assert tags is not None

    # filter the collection
    collection = mongo_client
    query = {"tags": {"$in": tags.args}}
    collection = collection.find(query)

    # get the first document
    document = collection[0]

    # go through the steps
    for step in document["steps"]:
        if "selector" in step["tool_args"] and "value" in step["tool_args"]:
            page.fill(step["tool_args"]["selector"], step["tool_args"]["value"])
        elif "selector" in step["tool_args"]:
            page.click(step["tool_args"]["selector"])
        elif "url" in step["tool_args"]:
            page.goto(step["tool_args"]["url"])
        else:
            raise ValueError("Invalid step")
            