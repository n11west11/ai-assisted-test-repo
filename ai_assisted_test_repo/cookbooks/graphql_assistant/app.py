import json
import os

import chainlit as cl
from chainlit.input_widget import TextInput
from dotenv import load_dotenv
import graphql
from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain.vectorstores.faiss import FAISS
from langchain_community.utilities.graphql import GraphQLAPIWrapper
from langchain_core.vectorstores import VectorStore

from graphql_execute_chain import chain as gql_execute_chain, ToolInputSchema
from database_vector import cached_embedder
from introspection import get_introspection_texts, aintrospect

load_dotenv()
graphql_endpoint = os.getenv("GRAPHQL_DEFAULT_ENDPOINT")


async def introspection_db(endpoint) -> VectorStore:
    introspection_result = await aintrospect(endpoint)
    introspection_texts = get_introspection_texts(introspection_result)
    db = await FAISS.afrom_documents(introspection_texts, cached_embedder)
    return db


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    agent = cl.user_session.get("agent")
    db = await introspection_db(settings["endpoint"])
    agent.tools[0]= create_retriever_tool(
            db.as_retriever(),
            "GraphQLIntrospect",
            """Searches and returns relevant information from the GraphQL schema.
            Useful for finding the correct query to use, and searching the GraphQL"""
        )
    wrapper = GraphQLAPIWrapper(
        custom_headers=json.loads(settings["headers"]),
        graphql_endpoint=settings["endpoint"])
    agent.tools[1].func = gql_execute_chain.with_config(configurable={"graphql_wrapper": wrapper}).invoke
    cl.user_session.set("agent", agent)


@cl.on_chat_start
async def start():
    # region Settings
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="endpoint",
                label="GraphQL endpoint",
                initial=graphql_endpoint,
                initial_index=0,
            ),
            TextInput(
                id="headers",
                label="GraphQL headers",
                initial="{}",
                initial_index=0,
            )
        ]
    ).send()
    # endregion
    # region Main Bot Setup
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613", streaming=True)
    db = await introspection_db(graphql_endpoint)
    # endregion

    # region Tools
    tools = [
        create_retriever_tool(
            db.as_retriever(),
            "GraphQLIntrospect",
            """Searches and returns relevant information from the GraphQL schema.
            Useful for finding the correct query to use, and searching the GraphQL"""
        ),
        Tool(
            name="GraphQLExecute",
            func=gql_execute_chain.invoke,
            description="Useful tool for executing graphql queries, input should be the users original request.",
            args_schema=ToolInputSchema,
            handle_tool_error=graphql.error.graphql_error.GraphQLError
        )
    ]
    # endregion

    # region Agent Setup
    agent = create_conversational_retrieval_agent(llm, tools, 
                                                  verbose=True, 
                                                  remember_intermediate_steps=False,
                                                  handle_parsing_errors=True,
                                                  max_token_limit=5000)
    cl.user_session.set("agent", agent)
    # endregion


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    res = await agent.arun(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res).send()
