from ai_assisted_test_repo.tools.ui import model


async def ExecutePlaywrightOperation(browser_context, page, **kwargs):

    if not hasattr(page, kwargs["operation"]):
            raise ValueError(f"Invalid operation: {kwargs['operation']}")
    
    operation = kwargs["operation"]

    if "content" in kwargs["operation"]:
        return "Please use the PageContent function to evaluate page content"

    method = getattr(page, operation)
    try:
        args = kwargs.get("args", {})
        await method(*args)
    except Exception as e:
        return f"An error occured, if it was a TimeoutError, please fetch page content exception str{e}"
    
    return "Success"