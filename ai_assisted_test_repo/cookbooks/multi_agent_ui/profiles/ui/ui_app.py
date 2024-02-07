import functools
import json
from typing import Any, Dict, List, Sequence, TypedDict, Union, cast

import chainlit as cl
from langchain.agents import create_openai_tools_agent
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.globals import set_debug
from langchain.memory import ChatMessageHistory
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage
from langchain_community.tools.playwright.utils import aget_current_page
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor
from playwright.async_api import async_playwright

from ai_assisted_test_repo.openai.num_tokens_from_messages import (
    num_tokens_from_messages,
)
from tools.test_management.fetch_test import loader as test_loader

# TODO: remove all cookbok imports
from tools.toolkits import (
    test_documentation_toolkit,
    test_execution_toolkit,
)
from tools.ui.chain import playwright_chain

set_debug(True)

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
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: list[tuple[AgentAction, str]]
    # playwight context, gives the agent recommendations on locators
    playwright_context: str
    # The next agent to call
    next: str


def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="intermediate_steps"),
            ("human", "{input}"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )


def create_observation_agent(llm, agent_name: str = ""):
    """Observation agent. This agent is responsible for observing the conversation and providing feedback."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                " You are a observation agent, you're task is to determine what the next action should be."
                " If you believe the next action would be to interact with an element, please ALWAYS provide a locator"
                " that can be used by playwright to interact with the element. Do not ask for additional information"
            ),
            ("ai", "{playwright_context}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            )
        )
        | prompt
        | llm
        | OpenAIToolsAgentOutputParser()
    )
    agent.name = agent_name
    return agent


def create_agent(llm, tools, system_message: str, agent_name: str = ""):
    """Create an agent."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{system_message}" + "\n"
                " Use the provided tools to progress towards aiding in performing the users request."
                " If you are unable to fully act, that's OK"
                " Execute what you can to make progress."
                " You have access to the following tools: {tool_names}.\n",
            ),
            ("ai", "{playwright_context}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent.name = agent_name
    return agent


async def _supervisor_node(state, agent, name):
    result = agent.invoke(state)
    return result


async def _observation_node(state, agent, name):
    # response = await agent.ainvoke(state)
    # action = AgentAction(tool="Observation Agent", log=response.log, tool_input="")
    # # add observation to intermediate steps
    # state["intermediate_steps"].append((action, action.log))
    return state


async def tool_node(state):
    tools = cl.user_session.get("tools")
    tool_executor = ToolExecutor(tools)
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    for action in state["agent_outcome"]:
        output = await tool_executor.ainvoke(action)
        state["intermediate_steps"].append((action, output))

    return state


async def agent_node(state, agent, name):
    # to decrease the chances of running out of memory we trim the messages
    # warning if you see tool errors, this is the first place to look

    await trim_intermediate_steps(state)
    # cast messages to a sequence
    response = await agent.ainvoke(state)
    state["agent_outcome"] = response
    # We return a list, because this will get added to the existing list
    if isinstance(response, AgentFinish):
        state["messages"] += [AIMessage(content=response.log, name=name)]

    return state




async def set_context(state):
    """
    This agent is responsible for setting up the context
    """
    state = await playwright_context_agent(state)
    return state


async def playwright_context_agent(state):
    """
    This agent is responsible for finding the playwright context
    the playwright chain is responsible for finding the locators
    the playwright retriever is responsible for finding relevant links on the page
    """
    browser = cl.user_session.get("browser")
    page = await aget_current_page(browser)
    content = await page.content()
    last_input = ""
    last_outcome = ""
    if state["agent_outcome"]:
        for action in state["agent_outcome"]:
            last_input += action.log

    if state["intermediate_steps"]:
        last_outcome = state["intermediate_steps"][-1][1]

    # check if there are previous messages
    if len(state["messages"]) > 0:
        last_ai_message = next(
            (
                message.content
                for message in reversed(state["messages"])
                if isinstance(message, AIMessage)
            ),
            None,
        )
    else:
        last_ai_message = ""

    # Context is test context + last intermediate step + input
    context_input = (
        state["input"]
        + "\n"
        + last_input
        + "\n"
        + last_outcome
        + "\n"
        + last_ai_message
    )
    outcome = await playwright_chain(content).ainvoke(context_input)
    if "Not Applicable" in str(outcome):
        return state
    state["playwright_context"] = outcome
    return state


def router(state):
    match state["agent_outcome"]:
        case AgentFinish():
            return "end"
        case list():
            return "call_tool"
        case _:
            return "continue"


def supervisor_router(state):
    # This is the router
    match state["next"]:
        case "Test Executor":
            return "Test Executor"
        case "Test Documentor":
            return "Test Documentor"
        case "FINISH":
            return "FINISH"


# endregion


# region Chainlit Setup
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


def set_test_executor_node(async_browser):
    test_executioner_tools = test_execution_toolkit.get_tools(
        async_browser=async_browser
    )
    test_executioner_agent = create_agent(
        llm,
        test_executioner_tools,
        system_message="You should execute the test information provided by the notes you have been given"
        "and provide accurate details on what you have performed. Please stop when you appear to be unable to"
        "perform an action after 4 attempts",
        agent_name="Test Executor",
    )
    test_executioner_node = functools.partial(
        agent_node, agent=test_executioner_agent, name="Test Executor"
    )
    cl.user_session.set("test_executioner_node", test_executioner_node)

def set_test_documentor_node():
    test_documentor_tools = test_documentation_toolkit.get_tools()
    test_documentor_agent = create_agent(
        llm,
        test_documentor_tools,
        system_message="You should document the test information provided by the test_executioner."
        "You should perform the action most relevant to the user request using the tools provided",
        agent_name="Test Documentor",
    )
    test_documentor_node = functools.partial(
        agent_node, agent=test_documentor_agent, name="Test Documentor"
    )
    cl.user_session.set("test_documentor_node", test_documentor_node)

def set_observation_node():
    observation_agent = create_observation_agent(
        llm,
        agent_name="Observation Agent",
    )
    observation_node = functools.partial(
        _observation_node, agent=observation_agent, name="Observation Agent"
    )
    cl.user_session.set("observation_node", observation_node)

def set_supervisor_node():
    supervisor_agent = create_team_supervisor(
        llm,
        " You are a supervisor tasked with managing a conversation between the"
        " following workers: Test Executor, Test Documentor."
        " If a user asks for a test to be executed, or actions to be taken use the Test Executor."
        " Common actions include: clicking, filling, and selecting, run, rerun, and execute"
        " Correspond the following user verbs to the Test Executor:"
        " execute, run, rerun, and perform, click, fill, select, and choose.\n\n"
        " If a user asks for a test to be saved, updated, or managed, use the Test Documentor."
        " Common actions include: saving, updating, and managing, add a step. \n\n"
        " Correspond the following user verbs to the Test Documentor:"
        " save, update, manage, and document."
        " Given the following user request, "
        " respond with the worker to act next. Each worker will perform a"
        " to the best of their ability and then report back", 
        [
            "Test Executor",
            "Test Documentor",
        ]
    )
    supervisor_node = functools.partial(
        _supervisor_node, agent=supervisor_agent, name="supervisor"
    )
    cl.user_session.set("supervisor_node", supervisor_node)

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
    tools = cl.user_session.get("tools")  # type:  [Tool]
    for element in message.elements:
        if isinstance(element, cl.File):
            await cl.Step(
                elements=[
                    cl.Text(
                        content=f"Loading file: {element.name} as context to browser",
                        language="python",
                    )
                ],
                parent_id=cl.context.current_step.parent_id,
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
                    parent_id=cl.context.current_step.parent_id,
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
                    parent_id=cl.context.current_step.parent_id,
                ).send()


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


@cl.step
async def trim_messages(messages):
    cl.user_session.get("messages")
    # count the tokens in the messages
    num_tokens = num_tokens_from_messages(messages)

    # our total limit is 16000 tokens
    # right now we split the context in half between permanent(history)
    # and temporary messages (intermediate steps)
    # Also we need to think about user input (input), I doubt well need more
    # than 1000 tokens for user input, but we should keep an eye on it, because this is a guess
    context_limit = 16000
    input_token_limit = 1000
    history_token_limit = (context_limit - input_token_limit) / 2

    if num_tokens > context_limit:
        # we need to trim the messages
        # Usually AI messages will have better context than human messages
        # so we want to keep the AI messages
        # we also want to keep the last human message
        # and the first human message

        human_messages = [
            message for message in messages if isinstance(message, HumanMessage)
        ]
        ai_messages = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        while num_tokens_from_messages(messages) > history_token_limit:
            # if there are 3 occurences of the human message, then we pop the middle one
            # this is because we want to keep the first and last human message
            # but we don't want to keep the middle human message
            if len(human_messages) > 2:
                human_messages.pop(1)
                messages.pop(human_messages[1])
            # if we can't pop a human message, then we pop an ai message
            elif len(ai_messages) > 0:
                ai_messages.pop(0)
                messages.pop(ai_messages[0])

    cl.user_session.set("messages", messages)


@cl.step
async def trim_intermediate_steps(state):
    max_intermediate_steps = 10
    if state["agent_outcome"]:
        max_intermediate_steps += len(state["agent_outcome"])

    if len(state["intermediate_steps"]) > max_intermediate_steps:
        # Loop in reverse to avoid index issues when removing elements
        # We want to remove the last element if it is the same as the second last element
        for i in range(
            len(state["intermediate_steps"]) - 2, -1, -1
        ):  # Start from second-last element
            if (
                state["intermediate_steps"][i][0].tool
                == state["intermediate_steps"][i + 1][0].tool
                and state["intermediate_steps"][i][0].tool_input
                == state["intermediate_steps"][i + 1][0].tool_input
            ):
                state["intermediate_steps"].pop(i)

        # If we still have too many intermediate steps, then we remove the from the top
        while len(state["intermediate_steps"]) > max_intermediate_steps:
            last_intermediate_step = state["intermediate_steps"].pop(0)
            # save the last intermediate step into the session, just in case we need it
            intermediate_step_history = cl.user_session.get(
                "intermediate_step_history", []
            )
            intermediate_step_history.append(last_intermediate_step)
            cl.user_session.set("intermediate_step_history", intermediate_step_history)

    return state


# endregion


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
        tools = []
        await cl.Avatar(name="Qbert", path="./public/logo_dark.png").send()

        # region Playwright Setup
        playwright = await async_playwright().start()
        async_browser = await playwright.chromium.connect_over_cdp(
            "ws://localhost:3000",
        )
        cl.user_session.set("browser", async_browser)
        # endregion

        # region test_executer
        set_test_executor_node(async_browser=async_browser)
        # endregion

        # region observation agent
        set_observation_node()
        # endregion

        # region test_documentor
        set_test_documentor_node()
        # endregion

        # region supervisor
        set_supervisor_node()

        cl.user_session.set("messages", [])

    @staticmethod
    @cl.action_callback("Open debug window")
    async def open_debug_window():
        # TODO: make this a hyperlink
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("http://localhost:3000")

    @staticmethod
    @cl.on_message
    async def process_message(message):

        history = cl.user_session.get("messages")  # type: ChatMessageHistory
        actions = [
            cl.Action(
                name="Open debug window", value="example_value", description="Click me!"
            ),
            cl.Action(name="Jira", value="TSK-123", description="Interact with Jira"),
        ]

        await load_files(message)

        test_executor_node = cl.user_session.get("test_executioner_node")
        test_documentor_node = cl.user_session.get("test_documentor_node")
        observation_node = cl.user_session.get("observation_node")
        supervisor_node = cl.user_session.get("supervisor_node")

        workflow = StateGraph(AgentState)

        workflow.add_node("Context Manager", set_context)
        workflow.add_node("Test Executor", test_executor_node)
        workflow.add_node("Test Documentor", test_documentor_node)
        workflow.add_node("Observation Agent", observation_node)
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("call_tool", tool_node)

        workflow.add_conditional_edges(
            "Test Documentor",
            router,
            {"continue": "Test Documentor", "call_tool": "call_tool", "end": END},
        )

        workflow.add_conditional_edges(
            "Test Executor",
            router,
            {"continue": "Test Executor", "call_tool": "call_tool", "end": END},
        )

        workflow.add_conditional_edges(
            "supervisor",
            supervisor_router,
            {
                test_executor_node.keywords["name"]: "Context Manager",
                test_documentor_node.keywords["name"]: test_documentor_node.keywords[
                    "name"
                ],
                "FINISH": END,
            },
        )

        workflow.add_edge("call_tool", "Context Manager")
        workflow.add_edge("Context Manager", "Observation Agent")

        workflow.add_conditional_edges(
            "Observation Agent",
            # Each agent node updates the 'sender' field
            # the tool calling node does not, meaning
            # this edge will route back to the original agent
            # who invoked the tool
            lambda x: x["next"],
            {
                "Test Executor": "Test Executor",
                "Test Documentor": "Test Documentor",
            },
        )
        workflow.set_entry_point("supervisor")
        graph = workflow.compile()

        state = cl.user_session.get("state", {})

        state["input"] = message.content
        state["messages"] = history
        state["intermediate_steps"] = []
        state["agent_outcome"] = None
        state["playwright_context"] = (
            state["playwright_context"] if "playwright_context" in state else ""
        )

        res = await graph.ainvoke(
            state,
            config={
                "callbacks": [cl.LangchainCallbackHandler(), UIAppHandler()],
                "recursion_limit": 100,
            },
        )
        # pop last message from history
        last_message = history.pop()

        if last_message.content == message.content:
            # this means that the agent did not respond
            # we need to add a message to the history
            # indicating that the agent did not respond
            history.append(
                SystemMessage(content="The agent failed to respond to the user request")
            )
        # add messages back to history
        history.append(HumanMessage(content=message.content))
        history.append(last_message)

        await trim_messages(history)

        cl.user_session.set("state", res)
        await cl.Message(content=res["messages"][-1].content, actions=actions).send()

        set_test_documentor_node() # reset the test documentor node, so it refreshes the db

