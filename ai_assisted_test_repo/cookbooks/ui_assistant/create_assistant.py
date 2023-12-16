import json
from openai import AsyncOpenAI
import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
from ai_assisted_test_repo.tools.ui.model_json_schema import ui_tools_dir

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

tools = [
    # {"type": "code_interpreter"}
]

ui_tools = []

for file in Path(ui_tools_dir).glob("*.json"):
    # append to ui_tools after doing json dump
    ui_tools.append(json.load(open(file, "r")))

tools.extend(ui_tools)

async def fake_tool(*args, **kwargs):
    return "success"
    
tool_map = {
}

for tool in ui_tools:
    tool_map[tool["function"]["name"]] = fake_tool


async def create():
    # ... your existing create function code ...

    instructions = """You are a UI Testing assistant. You aid in executing playwright actions against a playwright page class.
        The page class is accessible via functions that are passed to you.

        You have access to three main functions: ExecutePlaywrightOperation and PageContent

        PageContent may be used to find valid locators. Behind the scenes a page.content() call is ran and beautifulSoup is applied to the content and fed back through the function.

        When running ExecutePlaywrightOperation it is always beneficial to run these command before passing a locator object to ExecutePlaywrightOperation.

        ExecutePlaywrightOperation is a function that takes a playwright function and executes it against
        a page class, so you can do things like:

        ```python
        from playwright.sync_api import Page

        def test_something(page: Page):
            page.goto("https://www.google.com")
            page.click("[name=q]")
            page.fill("[name=q]", "playwright github")
            page.press("[name=q]", "Enter")
        ```

        Your job is to execute these functions to aid in testing. The user will have the ability to
        see the content of the page, and guide you on next steps.

        You should have a friendly tone, and always be polite.
    """

    assistant = await client.beta.assistants.create(
        name="Qubert",
        instructions=instructions,
        tools=tools,
        model="gpt-3.5-turbo-1106",
    )
    assistant_name = "Qubert"
    # append key vallue pair to assistants.json

    def load_or_create_json(filename):
        try:
            return json.load(open(filename, "r"))
        except FileNotFoundError:
            raise FileNotFoundError(f"{filename} not found")
    assistant_dict = load_or_create_json("assistants.json")
    assistant_dict[assistant_name] = assistant.id
    json.dump(assistant_dict, open("assistants.json", "w"))

# asyncio.run(create())
