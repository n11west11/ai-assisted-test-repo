import json
import os
from langchain.requests import RequestsWrapper
from nltk.chat import Chat

import requests
from cffi import api
import chainlit as cl
from chainlit.input_widget import TextInput
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits import \
    create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_debug
from langchain_community.llms import Ollama
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.agent_toolkits.openapi import planner

load_dotenv()

set_debug(True)

# region Bot Setup
llm = Ollama(model="llama2")
# endregion


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    raw_api_spec = json.loads(settings["api_spec"])
    api_spec = reduce_openapi_spec(raw_api_spec)
    requests_wrapper = RequestsWrapper()
    api_agent = planner.create_openapi_agent(api_spec, requests_wrapper, llm)
    cl.user_session.set("agent", api_agent)


@cl.on_chat_start
async def start():
    with open("default_api.json") as f:
        raw_api_spec = json.load(f)

    # region Settings
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="api_spec",
                label="Open API Schema",
                initial=json.dumps(raw_api_spec, indent=2),
            )
        ]
    ).send()
    # endregion


    # region OpenAPI Agent Setup
    headers = {"x-user-id": f"{os.getenv('API_KEY')}"}
    requests_wrapper = RequestsWrapper(headers=headers)

    api_spec = reduce_openapi_spec(raw_api_spec)
    api_agent = planner.create_openapi_agent(api_spec, requests_wrapper, llm)
    # endregion

    # region Tools
    tools = []
    # endregion

    # region Agent Setup
    cl.user_session.set("agent", api_agent)
    # endregion


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    res = await agent.arun(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res).send()
