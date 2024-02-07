import importlib
from json import tool

from langchain.agents.openai_assistant import OpenAIAssistantRunnable
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from playwright.sync_api import sync_playwright

from tools.toolkits import default_toolkit
from tools.ui import FillTool, GetElementsTool


def playwright_setup():
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    return browser


def _create_assistant(tools):
    return OpenAIAssistantRunnable.create_assistant(
        name="Qubert UI Assistant",
        instructions="You are a UI test assistant, you aid in creating, executing, and debugging UI tests.",
        tools=tools,
        model="gpt-3.5-turbo",
    )

def get_open_ai_ui_assistant(): 
    return OpenAIAssistantRunnable(assistant_id="asst_VZene982S5dWmaLzkuJ0V4n8", as_agent=True)
     

if __name__ == "__main__":
    browser = playwright_setup()

    tools = (
        PlayWrightBrowserToolkit(sync_browser=browser).get_tools()
        + default_toolkit.get_tools()
        + [FillTool(sync_browser=browser),  GetElementsTool(sync_browser=browser)]
    )
    _create_assistant()