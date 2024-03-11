import functools
import html
import json
import operator
from typing import Annotated, Any, Dict, List, Sequence, Tuple, TypedDict, Union, cast, final

import chainlit as cl
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.chains.openai_functions import (
    create_openai_fn_runnable,
    create_structured_output_runnable,
)
from langchain.globals import set_debug
from langchain.memory import ChatMessageHistory, ConversationTokenBufferMemory
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage
from langchain_community.tools.playwright.utils import aget_current_page
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, create_agent_executor
from playwright.async_api import async_playwright
from pyee import executor

from ai_assisted_test_repo.openai.num_tokens_from_messages import (
    num_tokens_from_messages,
    num_tokens_from_string,
)
from tools.test_management.fetch_test import loader as test_loader
from tools.toolkits import test_documentation_toolkit, test_execution_toolkit
from tools.ui.basic_chain import html_vector_store, playwright_chain as basic_chain, split_html
from tools.ui.manage_html import condense_html, get_page_content

load_dotenv()
set_debug(True)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


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
    current_html: str
    current_url: str
    current_page_text: str
    # The next agent to call
    next: str


# region Plan
class PlanExecute(AgentState):
    plan: List[str]
    past_steps: List[Tuple]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    completed: bool = Field(description="Whether the plan is completed or not")
    final_answer: str = Field(description="The final answer, if the plan is completed")
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )




planner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan.
     This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    You have currently done the follow steps:
    playwright = await async_playwright().start()
    async_browser = await playwright.chromium.connect_over_cdp(
        "ws://localhost:3000",
    )
    page = await async_browser.new_page()
    You are currently on the following page:
    {current_url}

    You have access to a Chrome Browser that is represented by the following Page Object Model. 
    {current_html}

    Objective:
    {objective}"""
)
planner_prompt = ChatPromptTemplate.from_template(
    """
    Imaging you are an engineer navigating webpages and you have been given a task to complete.

    You have already done the following steps:
    playwright = await async_playwright().start()
    async_browser = await playwright.chromium.connect_over_cdp(
        "ws://localhost:3000",
    )
    page = await async_browser.new_page()

    For the given objective, come up with a simple step by step plan.
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    You are currently on the following page:
    {current_url}

    You have access to a Chrome Browser that is represented by the following Page Object Model. 
    {current_html}

    Objective:
    {objective}"""
)
planner = create_structured_output_runnable(Plan, llm, planner_prompt)

replanner_prompt = ChatPromptTemplate.from_template(
    """
    Imaging you are an engineer navigating webpages and you have been given a task to complete.
    For the given objective, determine if any further steps are needed to fulfill the objective. 
    Provide a final answer if no further steps are needed.
    Otherwise, fill out the plan.
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. 
    Do not add any superfluous steps.
    Make sure that each step has all the information needed - do not skip steps.
    Assume that each step in the plan involves interacting with a playwright page object in python. 

    Also some notes on navigating websites:
    * After you fill a field, if you expect the page to change, you shoud press enter, or click a submit button.

    You are currently on the following page:
    {current_url}
    
    You have access to a Chrome Browser that is represented by the following Page Object Model. 
    {current_html}

    Your objective was this:
    {input}

    Your original plan was this:
    {plan}

    You have currently done the follow steps:
    {past_steps}

    Only add steps to the plan that still NEED to be done. 
    Do not return previously done steps as part of the plan.
    Do not include steps that verify the final answer. Only include steps that are actionable. 
    """
)

replanner = create_openai_fn_runnable(
    functions=[Plan],
    llm=llm,
    prompt=replanner_prompt,
)
# endregion

# region Graph


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
                " that can be used by playwright to interact with the element. Do not ask for additional information",
            ),
            ("ai", "{current_html}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = prompt | llm | OpenAIToolsAgentOutputParser()
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
            ("ai", "{current_html}"),
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


def create_nav_agent(llm, tools, system_message: str, agent_name: str = ""):
    """Create an agent executor"""

    _prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                " Use the provided tools to progress towards aiding in performing the users request."
                " If you are unable to fully act, that's OK"
                " Execute what you can to make progress."
                " You have access to the following tools: {tool_names}.\n"
            ),
            ("ai", "you are currently on the following page: {current_url}"),
            ("ai", "Here is a playwright model of the current page your tools have access to:\n"
             "{current_html}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "I have executed the following steps: {past_steps}"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    _prompt = _prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    agent = create_openai_tools_agent(llm, tools, _prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[cl.AsyncLangchainCallbackHandler()],
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=20
    )
    return agent_executor


async def execute_plan_node(state: PlanExecute, agent, name):
    task = state["plan"][0]
    _state = state.copy()
    _state["input"] = task
    _state.pop("intermediate_steps", None)

    # Map state to AgentState, removing all keys that are not in AgentState
    
    agent_response = await agent.ainvoke(
        _state
    )

    state["past_steps"].append((task, agent_response.get("output", "")))
    return state


async def _supervisor_node(state, agent, name):
    result = agent.invoke(state)
    return result


async def _observation_node(state, agent, name):
    # response = await agent.ainvoke(state)
    # action = AgentAction(tool="Observation Agent", log=response.log, tool_input="")
    # # add observation to intermediate steps
    # state["agent_scratchpad"].append((action, action.log))
    return state


async def tool_node(state):
    tools = cl.user_session.get("tools")
    tool_executor = ToolExecutor(tools)
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    outcomes = []
    for action in state["agent_outcome"]:
        output = await tool_executor.ainvoke(action)
        state["intermediate_steps"].append((action, output))

    return state


async def agent_node(state, agent, name):
    # to decrease the chances of running out of memory we trim the messages
    # warning if you see tool errors, this is the first place to look
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
    browser = cl.user_session.get("browser")
    page = await aget_current_page(browser)
    content = await page.content()
    stripped_content = condense_html(content)
    if num_tokens_from_string(stripped_content) > 4000:
        html_docs = html_vector_store(stripped_content)

        most_relevant_html = html_docs.max_marginal_relevance_search(state["input"] + "\n".join(state["plan"]), k=4)
    else:
        most_relevant_html = stripped_content
    state["current_html"] = most_relevant_html
    state["current_page_text"] = get_page_content(content)
    state["current_url"] = page.url
    return state


async def set_starting_context(state):
    """
    This agent is responsible for setting up the starting context
    """
    browser = cl.user_session.get("browser")
    page = await aget_current_page(browser)
    content = await page.content()
    if num_tokens_from_string(content) > 4000:
        html_docs = html_vector_store(content)
        most_relevant_html = html_docs.max_marginal_relevance_search(state["input"] + "\n".join(state["plan"]), k=4)
    else:
        most_relevant_html = condense_html(content)
    state["current_html"] = most_relevant_html
    state["current_url"] = page.url
    state["current_page_text"] = get_page_content(content)
    state["past_steps"] = []
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


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke(
        {"objective": state["input"], "current_html": state["current_html"], "current_url": state["current_url"]}
    )
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)

    if output.final_answer:
        return {
                "response": output.final_answer,
                "messages": state["messages"] + [AIMessage(content=output.final_answer, name="Test Executor")]
                }
    else:
        return {"plan": output.steps}


def should_end(state: PlanExecute):
    if state["response"]:
        return True
    else:
        return False


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
    tools = cl.user_session.get("tools", [])
    test_executer_tools = test_execution_toolkit.get_tools(async_browser=async_browser)
    test_executer_agent = create_nav_agent(llm, test_executer_tools, "")
    test_executor_node = functools.partial(
        execute_plan_node, agent=test_executer_agent, name="Test Executor"
    )
    cl.user_session.set("test_executor_node", test_executor_node)
    cl.user_session.set("tools", tools + test_executer_tools)


def set_test_documentor_node():
    tools = cl.user_session.get("tools", [])
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
    cl.user_session.set("tools", tools + test_documentor_tools)


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
        ],
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
    # await page.wait_for_load_state("networkidle")

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
        history = ChatMessageHistory(messages=history)
        memory = ConversationTokenBufferMemory(
            llm=llm, chat_memory=history, memory_key="messages", return_messages=True
        )

        actions = [
            cl.Action(
                name="Open debug window", value="example_value", description="Click me!"
            ),
            cl.Action(name="Jira", value="TSK-123", description="Interact with Jira"),
        ]

        await load_files(message)

        # region Executor Workflow
        execulte_plan_node = cl.user_session.get("test_executor_node")
        executor_workflow = StateGraph(PlanExecute)

        executor_workflow.add_node("start_context", set_starting_context)
        # add the html context to the state
        executor_workflow.add_node("set_context", set_context)

        # Add the plan node
        executor_workflow.add_node("planner", plan_step)

        # Add the execution step
        executor_workflow.add_node("agent", execulte_plan_node)

        # Add a replan node
        executor_workflow.add_node("replan", replan_step)

        executor_workflow.set_entry_point("start_context")

        # From start context we go to planner
        executor_workflow.add_edge("start_context", "planner")
        # From plan we go to agent
        executor_workflow.add_edge("planner", "agent")

        # From agent, we set the context and then replan
        executor_workflow.add_edge("agent", "set_context")

        executor_workflow.add_edge("set_context", "replan")

        executor_workflow.add_conditional_edges(
            "replan",
            # Next, we pass in the function that will determine which node is called next.
            should_end,
            {
                # If `tools`, then we call the tool node.
                True: END,
                False: "agent",
            },
        )

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        execute_graph = executor_workflow.compile()
        # endregion

        test_documentor_node = cl.user_session.get("test_documentor_node")
        observation_node = cl.user_session.get("observation_node")
        supervisor_node = cl.user_session.get("supervisor_node")

        workflow = StateGraph(AgentState)

        workflow.add_node("Test Executor", execute_graph)
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
                execulte_plan_node.keywords["name"]: "Test Executor",
                test_documentor_node.keywords["name"]: test_documentor_node.keywords[
                    "name"
                ],
                "FINISH": END,
            },
        )

        workflow.add_edge("call_tool", "Observation Agent")

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
        main_workflow = workflow.compile()

        state = cl.user_session.get("state", {})

        state["input"] = message.content
        state["messages"] = memory.buffer_as_messages
        state["intermediate_steps"] = []
        state["agent_outcome"] = None
        state["current_html"] = (
            state["current_html"] if "current_html" in state else ""
        )

        res = await execute_graph.ainvoke(
            state,
            config={
                "callbacks": [cl.LangchainCallbackHandler(), UIAppHandler()],
                "recursion_limit": 100,
            },
        )
        # pop last message from history
        last_message = history.messages.pop() if history.messages else None

        history.messages.append(HumanMessage(content=message.content))
        if not last_message:
            # this means that the agent did not respond
            # we need to add a message to the history
            # indicating that the agent did not respond
            history.messages.append(
                SystemMessage(content="The agent failed to respond to the user request")
            )
        # add messages back to history
        else:
            history.messages.append(res["messages"][-1])

        cl.user_session.set("state", res)
        await cl.Message(content=res["messages"][-1].content, actions=actions).send()

        set_test_documentor_node()  # reset the test documentor node, so it refreshes the db
