from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from playwright.async_api import async_playwright
import chainlit as cl

@cl.action_callback("Open debug window")
async def on_action(action):
    actions = [
        cl.Action(name="Open debug window", value="example_value", description="Click me!")
    ]
    await cl.Message(content=f"Executed {action.name}", actions=actions).send()
    # Optionally remove the action button from the chatbot user interface
    await action.remove()


@cl.on_chat_start
async def start():

    playwright = await async_playwright().start()
    browser = await playwright.chromium.connect_over_cdp("http://localhost:3000")
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    tools = toolkit.get_tools()
    llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-3.5-turbo")

    agent = initialize_agent(
        tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
            # Sending an action button within a chatbot message
    actions = [
        cl.Action(name="Open debug window", value="example_value", description="Click me!")
    ]
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    res = await agent.arun(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )

    await cl.Message(content=res, actions=actions).send()
