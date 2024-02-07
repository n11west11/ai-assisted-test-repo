from __future__ import annotations

from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
    aget_current_page,
    get_current_page,
)


class FillToolInput(BaseModel):
    """Input for FillTool."""

    selector: str = Field(..., description="CSS selector for the element to fill")
    value: str = Field(..., description="Value to fill the element with")


class FillTool(BaseBrowserTool):
    """Tool for filling an element with the given CSS selector."""

    name: str = "fill_element"
    description: str = "Fill an element with the given CSS selector"
    args_schema: Type[BaseModel] = FillToolInput

    visible_only: bool = True
    """Whether to consider only visible elements."""
    playwright_strict: bool = False
    """Whether to employ Playwright's strict mode when filling elements."""
    playwright_timeout: float = 1_000
    """Timeout (in ms) for Playwright to wait for element to be ready."""

    def _selector_effective(self, selector: str) -> str:
        if not self.visible_only:
            return selector
        return f"{selector} >> visible=1"

    def _run(
        self,
        selector: str,
        value: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)
        # Navigate to the desired webpage before using this tool
        selector_effective = self._selector_effective(selector=selector)
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        try:
            page.fill(
                selector_effective,
                value,
                strict=self.playwright_strict,
                timeout=self.playwright_timeout,
            )
        except PlaywrightTimeoutError:
            return f"Unable to fill element '{selector}'"
        return f"Filled element '{selector}'"

    async def _arun(
        self,
        selector: str,
        value: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)
        # Navigate to the desired webpage before using this tool
        selector_effective = self._selector_effective(selector=selector)
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        try:
            await page.fill(
                selector_effective,
                value,
                strict=self.playwright_strict,
                timeout=self.playwright_timeout,
            )
        except PlaywrightTimeoutError:
            return f"Unable to fill element '{selector}'"
        return f"Filled element '{selector}'"
