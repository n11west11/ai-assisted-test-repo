import json
from enum import Enum
import os
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, HttpUrl

from ai_assisted_test_repo.tools.export_model import export_model

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(MODULE_DIR, "functions")

# region BeautifulSoup
class BeautifulSoupElementTypes(str, Enum):
    div = "div"
    button = "button"
    input = "input"
    select = "select"
    textarea = "textarea"
    audio = "audio"
    video = "video"
    embed = "embed"
    iframe = "iframe"
    img = "img"
    object = "object"
    canvas = "canvas"
    map = "map"
    meter = "meter"
    progress = "progress"
    svg = "svg"
    template = "template"
    details = "details"
    dialog = "dialog"
    menu = "menu"
    summary = "summary"
    span = "span"
    a = "a"
    p = "p"
    h1 = "h1"
    h2 = "h2"
    h3 = "h3"
    h4 = "h4"
    h5 = "h5"
    h6 = "h6"
# endregion

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


class PageContent(BaseModel):
    """
     Model representing the content of a webpage.
    """
    items_to_find: BeautifulSoupElementTypes = Field(description="type of element to search for")
    text_to_find: Optional[str] = Field(default="", description="text to find in the page content")

    class Config:
        use_enum_values = True


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
    


# endregion

if __name__ == "__main__":
    # export models to json

    for model in [
        ExecutePlaywrightOperation,
        PageContent,
    ]:
        export_model(model)
