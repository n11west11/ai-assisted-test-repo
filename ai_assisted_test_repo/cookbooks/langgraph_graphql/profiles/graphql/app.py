from email import header
import functools
from gc import callbacks
import json
import os
from typing import Any, Dict, List, Sequence, TypedDict, Union, cast
from unittest.mock import Base
from botocore import endpoint
from langchain.memory import ConversationTokenBufferMemory

import chainlit as cl
from chainlit.input_widget import TextInput
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_react_agent
from langchain.globals import set_debug
from langchain.memory import ChatMessageHistory
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage
from langchain_community.tools.playwright.utils import aget_current_page
from langchain_community.utilities.graphql import GraphQLAPIWrapper
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, agent_executor

from ai_assisted_test_repo.openai.num_tokens_from_messages import (
    num_tokens_from_messages,
)
from multidict import CIMultiDict
from tools.graphql.graphql_execute_tool import GraphQLExecuteTool
from tools.graphql.introspection import aintrospect, aintrospection_db, schema_mutations, schema_queries
from tools.test_management.fetch_test import loader as test_loader

# TODO: remove all cookbok imports
from tools.toolkits import (
    test_documentation_toolkit
)

set_debug(True)
load_dotenv()
graphql_endpoint = os.getenv("GRAPHQL_DEFAULT_ENDPOINT")

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")


# region Graph
class AgentState(TypedDict):
    # The input string
    input: str
    # The list of previous messages in the conversation
    messages: Sequence[BaseMessage]
    # The list of new messages in the conversation
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[list[AgentAction], AgentFinish, None]
    # List of actions and corresponding observations
    # GraphQL endpoint
    endpoint: str
    introspection: str
    queries: str
    mutations: str
    # The next agent to call
    next: str

def create_agent(llm, tools, system_message: str, agent_name: str = ""):
    """Create an agent."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{system_message}" + "\n"
                " You are a GraphQL Assistant, capable of executing graphql quueries and mutations."
                " You have access to the following tools: {tool_names}\n"
                " Use the information provided to provide an answer to the user's request."
            ),
            ("ai", "Current GraphQL Endpoint: {endpoint}"),
            ("ai", "Relevant Introspection Data: {introspection}"),
            ("ai", "Available Queries: {queries}"),
            ("ai", "Available Mutations: {mutations}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    agent = create_openai_tools_agent(llm, tools, prompt)
    memory = ConversationTokenBufferMemory(llm=llm, memory_key="messages", max_token_limit=4000, return_messages=True, input_key="input")
    agent_executor = AgentExecutor(name=agent_name, agent=agent, tools=tools, memory=memory, callbacks=[cl.LangchainCallbackHandler()])
    return agent_executor 


async def agent_node(state, agent: AgentExecutor, name):
    # to decrease the chances of running out of memory we trim the messages
    # warning if you see tool errors, this is the first place to look

    # await trim_intermediate_steps(state)
    # cast messages to a sequence
    state = await agent.ainvoke(state)
    state.pop("output")
    # Add the agent outcome to the state
    # state["messages"] = state["messages"] + [AIMessage(content=state["output"])]
    return state


async def introspection_node(state):
    """
    This agent is responsible for finding the introspection data
    
    """
    schema = cl.user_session.get("schema")
    schema_db = cl.user_session.get("schema_db") # type: FAISS
    search = "\n".join(message.content for message in state["messages"])
    search = search.join(state["input"])
    results = str(await schema_db.asimilarity_search(search, k=10))
    if results:
        state["introspection"] = results
    else:
        state["introspection"] = "No introspection data found"
    state["queries"] = schema_queries(schema)
    state["mutations"] = schema_mutations(schema)
    return state
    

def router(state):
    match state["agent_outcome"]:
        case AgentFinish():
            return "end"
        case list():
            return "call_tool"
        case _:
            return "continue"

# endregion


# region Chainlit Setup
def set_main_agent_node():
    graphql_tool = cl.user_session.get("graphql_tool")


    agent = create_agent(
        llm,
        [graphql_tool],
        system_message="You should execute the test information provided by the notes you have been given"
        "and provide accurate details on what you have performed. Please stop when you appear to be unable to"
        "perform an action after 4 attempts",
        agent_name="Test Executor",
    )
    main_agent_node = functools.partial(
        agent_node, agent=agent, name="Test Executor"
    )
    cl.user_session.set("main_agent_node", main_agent_node)

@cl.step
async def log_message(message):
    await cl.Step(
        name="log_message",
        elements=[
            cl.Text(
                content=f"Logged message: {message}",
                language="python",
                display="inline",
            )
        ],
        parent_id=cl.context.current_step.parent_id,
    ).send()

# endregion


class GraphQLProfile:
    name = "GraphQL"
    markdown_description = "Ask questions about a GraphQL API."
    icon = "https://picsum.photos/200"

    chat_profile = cl.ChatProfile(
        name=name,
        markdown_description=markdown_description,
        icon=icon,
    )

    @staticmethod 
    async def update_settings(settings):

        headers_list = json.loads(settings["headers"])
        # For now we support from GraphQL lists or dict
        if isinstance(headers_list, list):
            headers = {header['name']: header['value'] for header in headers_list}
            # remove all headers that start with ":"
            # headers = {k: v for k, v in headers.items() if not k.startswith(":")}
            ## Step 3: Create a CIMultiDictProxy from the CIMultiDict
            # headers = CIMultiDict(headers)
        elif isinstance(headers_list, dict):
            headers = headers_list

        wrapper = GraphQLAPIWrapper(
            custom_headers=json.loads(settings["headers"]),
            graphql_endpoint=settings["endpoint"],
        )
        
        graphql_tool = GraphQLExecuteTool(
            graphql_wrapper=wrapper,
            custom_headers=json.loads(settings["headers"]),
            endpoint=settings["endpoint"],
            handle_tool_error=True,
        )

        schema = await aintrospect(settings["endpoint"], headers)
        cl.user_session.set("schema_db", await aintrospection_db(schema))
        cl.user_session.set("schema", schema)
        cl.user_session.set("endpoint", settings["endpoint"])
        cl.user_session.set("graphql_tool", graphql_tool)
        cl.user_session.set("tools", [graphql_tool] + test_documentation_toolkit.get_tools())
        set_main_agent_node() 



    @staticmethod
    async def initialize():
        await cl.Avatar(name="Qbert", path="./public/logo_dark.png").send()
        
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

        schema = await aintrospect(graphql_endpoint, json.loads(settings["headers"]))

        # region Tools
        graphql_wrapper = GraphQLAPIWrapper(
            graphql_endpoint=settings["endpoint"],
            custom_headers=json.loads(settings["headers"]),
        )
        graphql_tool = GraphQLExecuteTool(
                graphql_wrapper=graphql_wrapper,
                endpoint=settings["endpoint"],
                custom_headers=json.loads(settings["headers"]),
                handle_tool_error=True,
            )
        
        cl.user_session.set("graphql_tool", graphql_tool)
        # endregion
        

        # region test_executer
        set_main_agent_node()
        # endregion

        
        cl.user_session.set("endpoint", settings["endpoint"])
        cl.user_session.set("schema", schema)
        cl.user_session.set("schema_db", await aintrospection_db(schema))
        cl.user_session.set("messages", [])
        cl.user_session.set("tools", [graphql_tool] + test_documentation_toolkit.get_tools())


    @staticmethod
    @cl.on_message
    async def process_message(message):

        history = cl.user_session.get("messages", [])  # type: ChatMessageHistory
        endpoint = cl.user_session.get("endpoint")
        main_agent_node = cl.user_session.get("main_agent_node")
        state = cl.user_session.get("state", {})

        workflow = StateGraph(AgentState)
        
        workflow.add_node("Introspection", introspection_node)
        workflow.add_node("Main Agent", main_agent_node)
        

        workflow.add_edge("Introspection", "Main Agent")

        workflow.add_edge("Main Agent", END)

        workflow.set_entry_point("Introspection")
        graph = workflow.compile()

        state["input"] = message.content
        state["messages"] = history
        state["agent_outcome"] = None
        state["endpoint"] = endpoint
        state["introspection"] = "None"
        state["queries"] = ""
        state["mutations"] = ""

        res = await graph.ainvoke(
            state,
            config={
                "callbacks": [cl.LangchainCallbackHandler()],
                "recursion_limit": 100,
            },
        )

        # add messages back to history
        history = history + res["messages"]
        
        cl.user_session.set("state", res)
        await cl.Message(content=res["messages"][-1].content).send()