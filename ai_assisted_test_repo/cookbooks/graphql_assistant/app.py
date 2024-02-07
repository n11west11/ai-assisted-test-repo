import json
import os

import chainlit as cl
from chainlit.input_widget import TextInput
from dotenv import load_dotenv
# from ai_assisted_test_repo.tools.test_management.fetch_test import test_retriever_tool
# from ai_assisted_test_repo.tools.test_management.save_test import save_test_tool
from graphql_execute_chain import GraphQLExecuteTool
from introspection import aintrospect, aintrospection_db
from langchain.agents import AgentExecutor
from langchain.globals import set_debug
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities.graphql import GraphQLAPIWrapper
from langchain_openai import ChatOpenAI

from ai_assisted_test_repo.cookbooks.graphql_assistant.create_conversational_retrieval_agent import \
    create_conversational_retrieval_agent

load_dotenv()
graphql_endpoint = os.getenv("GRAPHQL_DEFAULT_ENDPOINT")

set_debug(True)


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    agent = cl.user_session.get("agent")
    db = await aintrospection_db(settings["endpoint"])
    agent.tools[0] = create_retriever_tool(
        db.as_retriever(),
        "GraphQLIntrospect",
        """Searches and returns relevant information from the GraphQL schema.
            Useful for finding the correct query to use, and searching the GraphQL""",
    )
    wrapper = GraphQLAPIWrapper(
        custom_headers=json.loads(settings["headers"]),
        graphql_endpoint=settings["endpoint"],
    )
    agent.tools[1] = GraphQLExecuteTool(
        graphql_wrapper=wrapper,
        endpoint=settings["endpoint"],
        handle_tool_error=True,
    )
    cl.user_session.set("agent", agent)

async def load_tools(): 
    
    db = await aintrospection_db(settings["endpoint"])
    # endregion

    graphql_wrapper = GraphQLAPIWrapper(
        graphql_endpoint=settings["endpoint"],
        custom_headers=json.loads(settings["headers"])
    )
    return [
        create_retriever_tool(
            db.as_retriever(search_kwargs={"k": 7}),
            "GraphQLIntrospect",
            """Searches and returns relevant information from the GraphQL schema.
            Useful for finding general information about the GraphQL schema, not needed for the GraphQLExecute tool""",
        ),
        GraphQLExecuteTool(
            graphql_wrapper=graphql_wrapper,
            endpoint=settings["endpoint"],
            custom_headers=json.loads(settings["headers"]),
            handle_tool_error=True,
        )
        # save_test_tool,
        # test_retriever_tool,
    ]
    # endregion


@cl.on_chat_start
async def start():
    # region Settings
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="endpoint",
                label="GraphQL endpoint",
                initial=graphql_endpoint,
            ),
            TextInput(
                id="headers",
                label="GraphQL headers",
                initial="{}",
            ),
        ]
    ).send()
    # endregion
    # region Main Bot Setup
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613", streaming=True)

    graphql_wrapper = GraphQLAPIWrapper(
        graphql_endpoint=settings["endpoint"],
        custom_headers=json.loads(settings["headers"])
    )

    # region Tools
    tools = [
        create_retriever_tool(
            db.as_retriever(search_kwargs={"k": 7}),
            "GraphQLIntrospect",
            """Searches and returns relevant information from the GraphQL schema.
            Useful for finding general information about the GraphQL schema, not needed for the GraphQLExecute tool""",
        ),
        GraphQLExecuteTool(
            graphql_wrapper=graphql_wrapper,
            endpoint=settings["endpoint"],
            custom_headers=json.loads(settings["headers"]),
            handle_tool_error=True,
        )
        # save_test_tool,
        # test_retriever_tool,
    ]
    # endregion

    # region Agent Setup
    agent = create_conversational_retrieval_agent(
        llm,
        tools,
        verbose=True,
        max_token_limit=5000,
        handle_parsing_errors=True,
    )
    cl.user_session.set("agent", agent)
    # endregion


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    res = await agent.arun(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res).send()
