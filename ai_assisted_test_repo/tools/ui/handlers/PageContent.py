from bs4 import BeautifulSoup
from ai_assisted_test_repo.tools.ui import model
from playwright import Page


def PageContent(args: model.PageContent):
    if "page" in args:
        page: Page = args["page"]
    else:
        raise ValueError(f"args needs to have a page object provided")

    html = await page.content()
    soup = BeautifulSoup(html, 'html.parser')
    elements = soup.body.find_all(args["items_to_find"])
    list_to_return = []
    for element in elements:
        list_to_return.append(element)
    return str(list_to_return)