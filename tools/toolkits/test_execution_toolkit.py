from typing import TYPE_CHECKING, Optional

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
import playwright
from pyparsing import C
from tools.ui.playwright import PlaywrightTool
from tools.ui.click_element import ClickTool
from tools.ui.fill_element import FillTool
from tools.ui.get_elements import GetElementsTool

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.sync_api import Browser as SyncBrowser
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        from playwright.async_api import Browser as AsyncBrowser
        from playwright.sync_api import Browser as SyncBrowser
    except ImportError:
        pass


def get_tools(
    sync_browser: Optional[SyncBrowser] = None,
    async_browser: Optional[AsyncBrowser] = None,
):
    playwright_tools = PlayWrightBrowserToolkit.from_browser(
        async_browser=async_browser, sync_browser=sync_browser
    ).get_tools()

    # Remove get_elements and click_element from the list of tools
    for tool in playwright_tools:
        if tool.name == "get_elements":
            playwright_tools.remove(tool)
        if tool.name == "click_element":
            playwright_tools.remove(tool)

    tools= [
        FillTool(sync_browser=sync_browser, async_browser=async_browser),
        # GetElementsTool(sync_browser=sync_browser, async_browser=async_browser),
        ClickTool(sync_browser=sync_browser, async_browser=async_browser),
    ] + playwright_tools
    
    for tool in tools:
        tool.tags = ["playwright"]
    
    return tools
