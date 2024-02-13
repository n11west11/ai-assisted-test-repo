from __future__ import annotations

import json
import os
from typing import Dict, Optional, Type

from dotenv import load_dotenv
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_debug
from langchain.tools import BaseTool
from langchain_community.tools.graphql.tool import BaseGraphQLTool
from langchain_community.utilities.graphql import GraphQLAPIWrapper
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField, Runnable, RunnablePassthrough
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, validator

from ai_assisted_test_repo.cookbooks.graphql_assistant.introspection import *

load_dotenv()
set_debug(True)

GRAPHQL_DEFAULT_ENDPOINT = os.getenv("GRAPHQL_DEFAULT_ENDPOINT")

llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-3.5-turbo")
llm_big = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-4")


class ToolInputSchema(BaseModel):
    """Input schema for the GraphQLExecute tool"""

    query: str = Field(description="The users request")

    @validator("query", allow_reuse=True)
    @classmethod
    def validate_query(cls, value):
        print("validate_query", value)
        return json.dumps(value)


# region query_builder_prompts
query_builder_text = """Create a GraphQL query using the following information that attempts to answer the question. 
Remove any fields that look to be incorrect based on the introspection data.
Output to this tool is a detailed and correct GraphQL query:

Introspection Data: 
{introspection}

Available Queries:
{queries}

Available Mutations:
{mutations}

Request: 
{request}

Query: 
"""
# endregion

query_builder_prompt = ChatPromptTemplate.from_template(query_builder_text)

# original idea was
# prompt | introspection | graphql_examples | query_builder | graphql_execute | llm | StrOutputParser()
# So the logic is as follows;
# Given a command to run a graphql query (prompt)
# Introspect the graphql endpoint to get the schema (introspection_db)
# Use the info in the schema to find pre-made examples of queries that can be run (graphql_examples)
# Use the prompt, introspection, adn examples to build a query (query_builder)
# Execute the query on the endpoint (graphql_execute)
# Use the results to generate a response (llm)
# Parse the response into a string in human-readable form (StrOutputParser)


wrapper = GraphQLAPIWrapper(graphql_endpoint=GRAPHQL_DEFAULT_ENDPOINT)

graphql_tool = BaseGraphQLTool(graphql_wrapper=wrapper).configurable_fields(
    graphql_wrapper=ConfigurableField(
        id="graphql_wrapper",
        name="GraphQL Wrapper",
        description="Wrapper for the GraphQL API",
    )
)


def graphql_chain(
    endpoint: str = GRAPHQL_DEFAULT_ENDPOINT,
    custom_headers: Dict[str, str] | None = None,
) -> Runnable:
    graphql_wrapper = GraphQLAPIWrapper(
        graphql_endpoint=endpoint, custom_headers=custom_headers
    )
    chain = (
        {"request": lambda text: json.dumps(text)}
        | {
            "introspection": introspection_db(endpoint).as_retriever(),
            "request": RunnablePassthrough(),
        }
        | query_builder_prompt.with_config(callback=StdOutCallbackHandler())
        | llm
        | StrOutputParser()
        | graphql_tool
        | llm
        | StrOutputParser()
    ).with_config(configurable={"graphql_wrapper": graphql_wrapper})
    return chain


if __name__ == "__main__":
    default_chain = graphql_chain()
    print(default_chain.invoke("How many capsules are there?"))

    star_wars_endpoint = "https://swapi-graphql.netlify.app/.netlify/functions/index"
    star_wars_chain = graphql_chain(
        endpoint=star_wars_endpoint,
    )
    print(star_wars_chain.invoke("What is the name of the first movie in star wars?"))
