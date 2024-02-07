from os import sync

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch, Tags
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from langchain.agents import (AgentExecutor, AgentType, initialize_agent,
                              load_tools)
from langchain.agents.agent_toolkits import (JiraToolkit,
                                             PlayWrightBrowserToolkit)
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain_community.llms import ollama
from langchain_community.tools.playwright.utils import aget_current_page
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_openai import ChatOpenAI
from playwright.async_api import async_playwright

from tools.ui.chain import playwright_chain
from ai_assisted_test_repo.cookbooks.langchain_ui_assistant.create_conversational_retrieval_agent import \
    create_conversational_retrieval_agent
from tools.test_management import (save_test_tool,
                                                         test_retriever_tool)
from tools.ui.fill_element import FillTool
from tools.ui.get_elements import GetElementsTool


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="UI Test",
            markdown_description="Get started writing UI Tests",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="GraphQL",
            markdown_description="Test Some GraphQL",
            icon="https://picsum.photos/250",
        ),
        cl.ChatProfile(
            name="API",
            markdown_description="Test Some API Endpoints",
            icon="https://picsum.photos/250",
        ),
    ]

@cl.action_callback("Open debug window")
async def on_action(action):
    # Open a local chrome browser window
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    page = await browser.new_page()
    await page.goto("http://localhost:3000")

@cl.action_callback("Jira")
async def jira_action(action):
    actions = [
        cl.Action(name="Log Bug", value="example_value", description="Click me!"), 
        cl.Action(name="Create Epic", value="example_value", description="Click me!"),
        cl.Action(name="Create Story", value="example_value", description="Click me!"),
        cl.Action(name="Create Task", value="example_value", description="Click me!")
    ]
    await cl.Message(content=f"Lets save this stuff to Jira!", actions=actions).send()


@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
                initial_index=0,
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            )
        ]
    ).send()
    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"Starting chat using the {chat_profile} chat profile"
    ).send()

    # region Playwright Setup
    playwright = await async_playwright().start()
    browser = await playwright.chromium.connect_over_cdp("http://localhost:3000")
    playwright_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    # endregion

    # region Jira Setup
    jira = JiraAPIWrapper()
    jira_toolkit = JiraToolkit.from_jira_api_wrapper(jira)
    jira_tools = []
    for tool in jira_toolkit.get_tools():
        # replace spaces with underscores
        tool.name = tool.name.replace(" ", "_")
        # for now we just want the JQL Query tool, and the project tool
        # the other tools are potentially dangerous
        if tool.name == "JQL_Query":
            jira_tools.append(tool)
        if tool.name == "Get_Projects":
            jira_tools.append(tool)
    # endregion
    
    if chat_profile == "UI Test":
        tools = playwright_toolkit.get_tools()
        # add the jira toolkit
        tools.extend(jira_toolkit.get_tools())
        # we want to override the get_elements tool to use the playwright tool
        for i, tool in enumerate(tools):
            if tool.name == "get_elements":
                tools[i] = GetElementsTool(sync_browser=tool.sync_browser, async_browser=tool.async_browser)
                break
        tools.append(FillTool(sync_browser=tools[0].sync_browser, async_browser=tools[0].async_browser))
        tools.append(save_test_tool)
        tools.append(test_retriever_tool)
        # test_retriever_tool,)
    if chat_profile == "GraphQL":
        tools = load_tools(
        ["graphql"],
        graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index",
        )


    llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-3.5-turbo")
    # llm = ollama.Ollama(model="llama2")

    agent = create_conversational_retrieval_agent(
        llm,
        tools,
        verbose=True,
        max_token_limit=5000,
        handle_parsing_errors=True,
    )
    cl.user_session.set("browser", browser)
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    actions = [
        cl.Action(name="Open debug window", value="example_value", description="Click me!"),
        
        cl.Action(name="Jira", value="TSK-123", description="Interact with Jira"),
    ]
    agent = cl.user_session.get("agent") # type: AgentExecutor
    browser = cl.user_session.get("browser")
    page = await aget_current_page(browser)

    await cl.Step(
        name="Previous view",
        elements=[cl.Image(
                    name="Previous view",
                    content=await page.screenshot(),
                    display="inline",
                    size="medium",
                ),],
        parent_id=message.id,
    ).send()

    message_template = """
    User Message: 
    {user_message}

    Here is some usefull HTML if the user asks you to perform an action, if the user message doesn't appear to be a request, please ignore this message.
    {docs}
    """

    docs = await playwright_chain(await page.content()).ainvoke(message.content)
    await cl.Step(
        name="HTML",
        elements=[cl.Text(
                    name="HTML",
                    content=docs,
                    display="inline",
                    size="medium",
                ),],
        parent_id=message.id,
    ).send()

    message.content = message_template.format(user_message=message.content, docs=docs)

    res = await agent.arun(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res, actions=actions).send()
    await cl.Step(
            name="Current view",
            elements=[cl.Image(
                        name="Current view",
                        content=await page.screenshot(),
                        display="inline",
                        size="medium",
                    ),],
            parent_id=message.id,
    ).send()
