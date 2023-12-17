from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from ai_assisted_test_repo.tools.ui.registry import function_registry

# region Playwright
class PlaywrightOperationType(str, Enum):
    goto = "goto"
    get_by_text = "get_by_text"
    locator = "locator"
    click = "click"
    fill = "fill"
    screenshot = "screenshot"
    scroll_into_view_if_needed = "scroll_into_view_if_needed"
    go_back = "go_back"
    go_forward = "go_forward"
    hover = "hover"



class ExecutePlaywrightOperation(BaseModel):
    """
    Model representing an operation to be executed using Playwright.
    """
    operation: PlaywrightOperationType = Field(description="playwright operation to execute")
    args: list = Field(default_factory=list, description="positional arguments for a page operation")
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="keyword arguments for a page operation")


    class Config:
        use_enum_values = True
# endregion

async def execute_playwright_operation(**kwargs):

    page = kwargs.pop('page', None)
    operation = kwargs.pop('operation', None)

    if operation == "content":
        return "Please use the PageContent function to evaluate page content"

    method = getattr(page, operation)
    try:
        args = kwargs.get("args", {})
        await method(*args)
    except Exception as e:
        return f"An error occured, if it was a TimeoutError, please fetch page content exception str{e}" 
    return "Success"

function_registry[ExecutePlaywrightOperation.__name__] = execute_playwright_operation
