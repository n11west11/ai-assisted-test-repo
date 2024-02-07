"""
Playwright tool for executing playwright operations.
"""

from enum import Enum
from typing import Optional, Dict, Any, Type
from langchain_community.tools.playwright.utils import aget_current_page, get_current_page
from pydantic import BaseModel, Field
from langchain_community.tools.playwright.base import BaseBrowserTool


class PlaywrightOperationType(str, Enum):
    """
    Enum representing different types of operations in Playwright.

    Attributes:
    GOTO: Represents a 'goto' operation.
    CLICK: Represents a 'click' operation.
    FILL: Represents a 'fill' operation.
    SCREENSHOT: Represents a 'screenshot' operation.
    SCROLL_INTO_VIEW_IF_NEEDED: Represents a 'scroll_into_view_if_needed' operation.
    GO_BACK: Represents a 'go_back' operation.
    GO_FORWARD: Represents a 'go_forward' operation.
    HOVER: Represents a 'hover' operation.
    """

    GOTO = "goto"
    CLICK = "click"
    FILL = "fill"
    SCREENSHOT = "screenshot"
    SCROLL_INTO_VIEW_IF_NEEDED = "scroll_into_view_if_needed"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    HOVER = "hover"


class PlaywrightToolInput(BaseModel):
    """
    Input for PlaywrightTool.
    """

    operation: PlaywrightOperationType = Field(
        description="playwright operation to execute"
    )
    args: list = Field(
        default_factory=list, description="""positional arguments for a page operation, think of this as the arguments you would pass to a function"""
    )
    # kwargs: Dict[str, Any] = Field(
    #     default_factory=dict, description="keyword arguments for a page operation"
    # )


class PlaywrightTool(BaseBrowserTool):
    """
    Tool for executing playwright operations.
    """

    name: str = "playwright"
    description: str = (
        "A tool for executing playwright operations, such as getting elements."
    )
    args_schema: Type[BaseModel] = PlaywrightToolInput

    class Config:
        """
        Configuration class for Playwright.
        """

        use_enum_values = True

    def _run(self, operation: PlaywrightOperationType, **kwargs):
        """
        Executes a playwright operation.
        """
        page = get_current_page(self.sync_browser)
        operation_args = kwargs.get('args', [])
        # operation_kwargs = kwargs.get('kwargs', {})

        # Execute the operation with the provided arguments and keyword arguments
        getattr(page, operation)(*operation_args)
        return f"Executed {operation} operation on page {page}"

    async def _arun(self, operation: PlaywrightOperationType, **kwargs):
        """
        Executes a playwright operation.
        """
        page = await aget_current_page(self.async_browser)
        page.set_default_timeout(10000)
        operation_args = kwargs.get('args', [])
        # operation_kwargs = kwargs.get('kwargs', {})

        # Execute the operation with the provided arguments and keyword arguments
        try: 
            await getattr(page, operation)(*operation_args)
        except Exception as e:
            return f"Failed to execute {operation} operation on page {page} due to {e}"
        return f"Executed {operation} operation on page {page}"
