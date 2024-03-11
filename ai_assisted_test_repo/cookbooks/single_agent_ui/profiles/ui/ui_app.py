import functools
import json
from math import log
import operator
from json import tool
from textwrap import indent
from typing import Annotated, Dict, List, Sequence, TypedDict, Union, cast, Any

import chainlit as cl
from langchain.agents import (
    create_openai_tools_agent,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.globals import set_debug
from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import aget_current_page
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, create_agent_executor
from playwright.async_api import async_playwright, Browser
from tools.test_management.fetch_test import loader as test_loader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# TODO: remove all cookbok imports
from tools.toolkits import default_toolkit
from tools.ui.custom_chain import playwright_chain
from tools.ui.fill_element import FillTool
from tools.ui.get_elements import GetElementsTool

set_debug(True)

# Load the test and action databases
test_docs = test_loader.load()

# Create the databases
test_db = FAISS.from_documents(test_docs, OpenAIEmbeddings())

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
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: list[tuple[AgentAction, str]]
    # playwight context, gives the agent recommendations on locators
    playwright_context: str
    # test context, gives the agent recommendations on tests that exist
    test_context: str


class UIAppHandler(BaseCallbackHandler):
    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        if "tags" in kwargs and "playwright" in kwargs["tags"]:
            await screenshot()

    async def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        if "tags" in kwargs and "playwright" in kwargs["tags"]:
            await screenshot()


def create_agent(
    llm: ChatOpenAI,
    tools: list,
) -> str:
    """Create a function-calling agent and add it to the graph."""
    # Get the prompt to use - you can modify this!
    assistant_system_message = """You are a helpful assistant. \
    Only reference tools if they appear relevant to the user request. """

    message_history = ChatMessageHistory()
    message_history.add_message(SystemMessage(content=assistant_system_message))

    # agent_runnable = create_openai_tools_agent(llm, tools, prompt)
    llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-3.5-turbo")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                assistant_system_message,
            ),
            ("system", "{test_context}"),
            ("system", "{playwright_context}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    # memory = ConversationSummaryBufferMemory(
    #     llm=ChatOpenAI(temperature=0, streaming=True, model_name="gpt-3.5-turbo"),
    #     return_messages=True,
    #     input_key="messages",
    #     max_token_limit=2500,
    # )
    # agent = AgentExecutor(agent=agent, tools=tools, memory=memory, max_iterations=50)
    # test_aware_agent = test_prelude | agent
    # agent = playright_prelude | test_aware_agent
    return agent


async def playwright_context_agent(state):
    """
    This agent is responsible for setting up the playwright context
    """
    # Remove all tool messages from the Playwright Context
    browser = cl.user_session.get("browser")
    page = await aget_current_page(browser)
    outcome = await playwright_chain(await page.content()).ainvoke(state["input"])
    if "Not Applicable" in str(outcome):
        return state
    # We use the response to create a FunctionMessage

    state["playwright_context"] = outcome
    return state


async def test_context_agent(state):
    """
    This agent is responsible for finding the test context
    We use the last human message to find the test context
    Note: because we are using the last human message, this limits our test executions.
    Need to find a way to get the test context in more scenarios like

    Also, to manage memory, we only return the top 2 results
    Thus, only 2 tests can be executed at a time
    """
    # get the test db
    test_db = cl.user_session.get("test_db")
    # search tests and actions for similarity
    logs_pre_filter = test_db.similarity_search_with_score(
        state["input"]
    )  # type: list[tuple[Document, float]]
    # sort by similarity score, lowest first
    logs_pre_filter.sort(key=lambda x: x[1])

    # only return results with a score of 0.5 or lower (The score seems dynamic, so this can be improved)
    logs_post_filter = [(log, score) for log, score in logs_pre_filter if score <= 0.5]

    # only return top 2 results
    if len(logs_post_filter) > 2:
        logs_post_filter = logs_post_filter[:2]

    observation = "I can use these logs I if I need to perform an action" + "\n".join(
        json.dumps(log_.page_content, indent=4) for log_, _ in logs_post_filter
    )

    await log_message(observation)
    # insert as second to last message
    state["test_context"] = observation
    return state


async def run_agent(state, agent, name):
    # to decrease the chances of running out of memory
    # we only return the last 5 results of intermediate steps
    if len(state["intermediate_steps"]) > 5:
        state["intermediate_steps"] = state["intermediate_steps"][-5:]
    # cast messages to a sequence
    response = await agent.ainvoke(state)
    state["agent_outcome"] = response
    state["agent_scratchpad"] = []
    # We return a list, because this will get added to the existing list
    if isinstance(response, AgentFinish):
        state["messages"] += [AIMessage(content=response.log, name=name)]

    return state


# Define the function to execute tools
async def execute_tools(state):
    tools = cl.user_session.get("tools")
    tool_executor = ToolExecutor(tools)
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    for action in state["agent_outcome"]:
        output = await tool_executor.ainvoke(action)
        state["intermediate_steps"].append((action, str(output)))
    return {"intermediate_steps": state["intermediate_steps"]}


def should_continue(state):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(state["agent_outcome"], AgentFinish):
        return "end"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"
    # return "end"


@cl.step
async def screenshot():
    browser = cl.user_session.get("browser")
    page = await aget_current_page(browser)
    current_step = cl.context.current_step
    # Simulate a running task
    # wait for network idle
    await page.wait_for_load_state("networkidle")

    _screenshot = cl.Image(
        name="screenshot",
        content=await page.screenshot(),
        display="inline",
        size="medium",
    )
    await cl.Step(
        name="screenshot", elements=[_screenshot], parent_id=current_step.parent_id
    ).send()

@cl.step
async def load_files(message):
    browser = cl.user_session.get("browser")  # type: Browser
    tools = cl.user_session.get("tools") # type:  [Tool]
    for element in message.elements:
        if isinstance(element, cl.File):
            await cl.Step(
                elements=[
                    cl.Text(
                        content=f"Loading file: {element.name} as context to browser",
                        language="python",
                    )
                ], parent_id=cl.context.current_step.parent_id
            ).send()
            try:
                with open(element.path, "r") as f:
                    element.content = f.read()
                state_data = json.loads(element.content)
                await browser.contexts[0].add_cookies(state_data["cookies"])
                for tool in tools:
                    if hasattr(tool, "async_browser"):
                        tool.async_browser = browser
                await cl.Step(
                    elements=[
                        cl.Text(
                            content=f"Loaded file: {element.name} as context to browser",
                            language="python",
                        )
                    ], 
                    parent_id=cl.context.current_step.parent_id
                ).send()
            except Exception as e:
                await cl.Step(
                    elements=[
                        cl.Text(
                            content=f"Failed to load file: {element.name} as context to browser\n"
                            f" with error: {e}",
                            language="python",
                        )
                    ], 
                    parent_id=cl.context.current_step.parent_id
                ).send()

@cl.step
async def log_message(message):
    await cl.Step(
        elements=[
            cl.Text(
                content=f"Logged message: {message}",
                language="python",
            )
        ], 
        parent_id=cl.context.current_step.parent_id
    ).send()


class UITestProfile:
    name = "UI Test"
    markdown_description = "Get started writing UI Tests"
    icon = "https://picsum.photos/200"

    chat_profile = cl.ChatProfile(
        name=name,
        markdown_description=markdown_description,
        icon=icon,
    )

    @staticmethod
    async def initialize():
        await cl.Avatar(name="Qbert", path="./public/logo_dark.png").send()

        # region Playwright Setup
        playwright = await async_playwright().start() # type: playwright
        async_browser = await playwright.chromium.connect_over_cdp(
            "ws://localhost:3000", 
        )
        # async_browser = await playwright.chromium.launch(headless=False)

        cl.user_session.set("browser", async_browser)
        playwright_toolkit = PlayWrightBrowserToolkit.from_browser(
            async_browser=async_browser,
        )
        # endregion

        # region setup tools
        tools = playwright_toolkit.get_tools()
        for tool in tools:
            tool.tags = ["playwright"]

        tools += default_toolkit.get_tools()
        # we want to override the get_elements tool to use the playwright tool
        # TODO: in the ui tools we should create a custom playwright tool
        for i, tool in enumerate(tools):
            if tool.name == "get_elements":
                tools[i] = GetElementsTool(
                    sync_browser=tool.sync_browser,
                    async_browser=tool.async_browser,
                    tags=["playwright"],
                )
                break
        tools.append(
            FillTool(
                sync_browser=tools[0].sync_browser,
                async_browser=tools[0].async_browser,
                tags=["playwright"],
            )
        )

        llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-3.5-turbo")

        # Construct the OpenAI Functions agent
        agent_runnable = create_agent(llm, tools)
        
        cl.user_session.set("tools", tools)
        cl.user_session.set("agent", agent_runnable)
        cl.user_session.set("messages", [])
        cl.user_session.set("test_db", test_db)

    @staticmethod
    @cl.action_callback("Open debug window")
    async def open_debug_window():
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("http://localhost:3000")

    @staticmethod
    @cl.on_message
    async def process_message(message):
        history = cl.user_session.get("messages")  # type: ChatMessageHistory
        agent = cl.user_session.get("agent")

        actions = [
            cl.Action(
                name="Open debug window", value="example_value", description="Click me!"
            ),
            cl.Action(name="Jira", value="TSK-123", description="Interact with Jira"),
        ]

        await load_files(message)

        workflow = StateGraph(schema=AgentState)

        # Add the first agent node, we give it name `test_context_agent` which we will use later
        workflow.add_node("test_context_agent", test_context_agent)
        # Add the second agent node, we give it name `playwright_context_agent` which we will use later
        workflow.add_node("playwright_context_agent", playwright_context_agent)
        # Add the base agent node, we give it name `agent` which we will use later
        agent_node = functools.partial(run_agent, agent=agent, name="BaseAgent")
        workflow.add_node("agent", agent_node)
        # Add the action node, we give it name `action` which we will use later
        workflow.add_node("action", execute_tools)

        # Set the entrypoint as `test_context_agent`
        # This means that this node is the first one called
        workflow.set_entry_point("test_context_agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
            {
                # If `continue`, then we update the context
                "continue": "action",
                # Otherwise we finish.
                "end": END,
            },
        )

        # For the action node, we want to go back to the entry point,
        # so that we can refresh the context
        workflow.add_edge("action", "test_context_agent")

        # We now add a normal edge from `test_context_agent` to `playwright_context_agent`.
        # This means that after `test_context_agent` is called, `playwright_context_agent` node is called next.
        workflow.add_edge("test_context_agent", "playwright_context_agent")

        # We also define a new edge, from the "playwright_context_agent" to the agent node
        workflow.add_edge("playwright_context_agent", "agent")

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        chain = workflow.compile()

        res = await chain.ainvoke(
            {
                "input": message.content,
                "messages": history,
                "intermediate_steps": [],
            },
            config={
                "callbacks": [cl.LangchainCallbackHandler(), UIAppHandler()],
                "recursion_limit": 100,
            },
        )

        cl.user_session.set("messages", res["messages"])

        await cl.Message(content=res["messages"][-1].content, actions=actions).send()
        # await cl.Message(content=res["agent_outcome"]["output"], actions=actions).send()


    @staticmethod
    @cl.on_chat_end
    async def update_tools():
       # first update the db
        test_db = FAISS.from_documents(test_loader.load(), OpenAIEmbeddings())
        cl.user_session.set("test_db", test_db)
        