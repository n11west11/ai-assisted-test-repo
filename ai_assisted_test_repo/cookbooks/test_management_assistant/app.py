import os
from typing import Text

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch, TextInput
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools
from langchain.agents.agent_toolkits import (
    PlayWrightBrowserToolkit,
    create_conversational_retrieval_agent,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain.vectorstores.faiss import FAISS
from langchain_community.document_loaders import MongodbLoader
from langchain_community.tools.playwright.utils import aget_current_page
from playwright.async_api import async_playwright
from pymongo import MongoClient

from ai_assisted_test_repo.tools.embeddings import cached_embedder

load_dotenv()
MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = "knowledge_base"


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    agent = cl.user_session.get("agent")
    db = MongoClient(MONGO_URI)["knowledge_base"][settings["collection"]]
    loader = MongodbLoader(MONGO_URI, DB_NAME, settings["collection"])
    docs = await loader.aload()
    db = await FAISS.afrom_documents(docs, cached_embedder)
    agent.tools[0] = create_retriever_tool(
        db.as_retriever(search_kwargs={"k": 7}),
        "Test_Repository_Knowledge_Base",
        """Searches and returns relevant information from the Test Repository Knowledge Base.
                Useful for finding the correct query to use, and searching the Test Repository""",
    )


@cl.on_chat_start
async def start():
    db = MongoClient(MONGO_URI)["knowledge_base"]
    collection_list = db.list_collection_names()

    settings = await cl.ChatSettings(
        [
            Select(
                id="collection",
                label="Collection",
                initial_value=collection_list[0],
                values=collection_list,
            )
        ]
    ).send()
    loader = MongodbLoader(MONGO_URI, DB_NAME, settings["collection"])
    docs = await loader.aload()
    db = await FAISS.afrom_documents(docs, cached_embedder)
    for collection in collection_list:
        loader = MongodbLoader(MONGO_URI, DB_NAME, collection)
    db = await FAISS.afrom_documents([], cached_embedder)
    for collection in collection_list:
        loader = MongodbLoader(MONGO_URI, DB_NAME, settings["collection"])
        docs = await loader.aload()
        await db.aadd_documents(docs)

    llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-3.5-turbo")

    # region Tools
    tools = [
        create_retriever_tool(
            db.as_retriever(
                search_type="mmr", search_kwargs={"k": 6}  # Also test "similarity"
            ),
            "Test_Repository_Knowledge_Base",
            """Searches and returns relevant information from the Test Repository Knowledge Base.
                Useful for finding the correct query to use, and searching the Test Repository""",
        )
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
    res = await agent.acall(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res["output"]).send()
