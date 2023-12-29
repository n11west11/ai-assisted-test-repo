from __future__ import annotations

import json
import os
from calendar import c
from re import search

import graphql
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_community.tools.graphql.tool import BaseGraphQLTool
from langchain_community.utilities.graphql import GraphQLAPIWrapper
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (ConfigurableField, Runnable,
                                      RunnableParallel, RunnablePassthrough)
from langchain_core.tools import Tool
from pydantic.v1 import BaseModel, Field, validator

from ai_assisted_test_repo.cookbooks.graphql_assistant.introspection import *

load_dotenv()
GRAPHQL_DEFAULT_ENDPOINT = os.getenv("GRAPHQL_DEFAULT_ENDPOINT")

llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-3.5-turbo")
llm_big = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-4")


class ToolInputSchema(BaseModel):
    request: str = Field(description="The users original request")

    @validator("request", allow_reuse=True)
    def validate_query(cls, value):  #  noqa
        print("validate_query", value)
        return json.dumps(value)


# region query_builder_prompts
query_builder_text = """Create a GraphQL query using the following information that attempts to answer the question. 
Output to this tool is a detailed and correct GraphQL query:

Introspection Data: 
{introspection}

Request: 
{request}
"""

query_builder_text_with_examples = """Create a GraphQL query using the following information that attempts to answer 
the question. 

Output to this tool is a detailed and correct GraphQL query:

Working Query Examples: 
{examples}

Introspection Data: 
{introspection}

Request: 
{request}
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

setup_and_retrieval = RunnableParallel(
    introspection=introspection_db(GRAPHQL_DEFAULT_ENDPOINT).as_retriever(),
    request=RunnablePassthrough()
)


wrapper = GraphQLAPIWrapper(graphql_endpoint=GRAPHQL_DEFAULT_ENDPOINT)

graphql_tool = BaseGraphQLTool(graphql_wrapper=wrapper).configurable_fields(
    graphql_wrapper=ConfigurableField(
        id="graphql_wrapper",
        name="GraphQL Wrapper",
        description="Wrapper for the GraphQL API",
    )
)

chain = (
    query_builder_prompt
    | llm
    | StrOutputParser()
    | graphql_tool
    | llm
    | StrOutputParser()
).with_config(
    callbacks=[StdOutCallbackHandler()],
    verbose=True,
)

def graphql_chain(endpoint: str=GRAPHQL_DEFAULT_ENDPOINT) -> Runnable:
    setup_and_retrieval = RunnableParallel(
        introspection=introspection_db(endpoint).as_retriever(),
        request=RunnablePassthrough(),
    )
    graphql_wrapper = GraphQLAPIWrapper(graphql_endpoint=endpoint)
    return setup_and_retrieval | chain.with_config(configurable={"graphql_wrapper": graphql_wrapper})


def graphql_execute_tool(endpoint: str=GRAPHQL_DEFAULT_ENDPOINT) -> Tool:
    return Tool(
        name="GraphQLExecute",
        func=graphql_chain(endpoint).invoke,
        description="Useful tool for executing graphql queries, input should be the users original request.",
        args_schema=ToolInputSchema,
        handle_tool_error=graphql.error.graphql_error.GraphQLError,
    )

if __name__ == "__main__":
    default_chain = graphql_chain()
    print(default_chain.invoke("How many capsules are there?"))

    star_wars_chain = graphql_chain(endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index")
    print(star_wars_chain.invoke("What is the name of the last movie in star wars?"))
