from langchain.chat_models import ChatOpenAI
from langchain_community.tools.playwright.utils import aget_current_page
from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents import AgentType, load_tools
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from playwright.async_api import async_playwright


import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider


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

    playwright = await async_playwright().start()
    browser = await playwright.chromium.connect_over_cdp("http://localhost:3000")
    playwright_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    
    if chat_profile == "UI Test":
        tools = playwright_toolkit.get_tools()
    if chat_profile == "GraphQL":
        tools = load_tools(
        ["graphql"],
        graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index",
        )


    llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-3.5-turbo")

    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    agent = initialize_agent(
        tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, agent_kwargs=agent_kwargs, memory=memory,
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
    await cl.Message(
        author="Screenshot",
        content="Previous view...",
        elements=[cl.Image(
                    name="Previous view",
                    content=await page.screenshot(),
                    display="side",
                    size="medium",
                ),],
        parent_id=message.id,
    ).send()
    res = await agent.arun(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res, actions=actions).send()
    await cl.Message(
            author="Screenshot",
            content="New view...",
            elements=[cl.Image(
                        name="Current view",
                        content=await page.screenshot(),
                        display="inline",
                        size="medium",
                    ),],
            parent_id=message.id,
    ).send()
