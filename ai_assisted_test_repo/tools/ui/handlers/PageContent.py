from bs4 import BeautifulSoup
from ai_assisted_test_repo.tools.ui import model


async def PageContent(browser_context, page, **kwargs):
    html = await page.content()
    soup = BeautifulSoup(html, 'html.parser')
    items = kwargs.get("items_to_find", [model.BeautifulSoupElementTypes])
    elements = soup.body.find_all(items)
    list_to_return = []
    for element in elements:
        list_to_return.append(element)
    if list_to_return:
        return str(list_to_return)
    else:
        return "Could not find any elements of that type"