from enum import Enum
from typing import List, Optional
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError, model_validator
from ai_assisted_test_repo.tools.ui import registry


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


class PageContentArgs(BaseModel):
    """
    Parameters for scraping page content.

    Note:
    This function can return a lot of text, be carefull requesting both items to find
    and scrape_text in one function call

    Attributes:
    - items_to_find (Optional[List[BeautifulSoupElementTypes]]): List of elements to find on the page.
    - scrape_text (bool): Flag to scrape text from the page.
    """
    items_to_find: List[BeautifulSoupElementTypes] = Field(
        description="List of elements to find on the page",
        max_items=5)
    scrape_text: bool = Field( description="Flag to scrape text from the page"
    )


async def page_content(**kwargs):
    try:
        # Parse and validate kwargs using Pydantic model
        page = kwargs.pop('page', None)
        args = PageContentArgs(**kwargs)

        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")

        results = []

        if args.scrape_text:
            if args.text_to_find:
                elements = soup.find_all(text=args.text_to_find)
            else:
                elements = soup.find_all(text=True)
            results.extend([element.strip() for element in elements if element.strip()])

        if args.scrape_interactables:
            interactables = ["a", "button", "input", "select", "textarea"]
            for tag in interactables:
                elements = soup.find_all(tag)
                results.extend([str(element) for element in elements])
            return str(results)

        if args.items_to_find:
            for item in args.items_to_find:
                elements = soup.find_all(item)
                results.extend([str(element) for element in elements])

        return str(results)

    except ValidationError as e:
        return f"Validation error: {e}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
registry.function_registry[PageContentArgs.__name__] = page_content
