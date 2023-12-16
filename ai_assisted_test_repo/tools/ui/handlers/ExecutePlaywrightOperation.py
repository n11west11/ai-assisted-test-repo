from ai_assisted_test_repo.tools.ui import model
from playwright import Page


async def ExecutePlaywrightOperation(args: model.ExecutePlaywrightOperation):
  # code goes
    if "page" in args:
        page: Page = args["page"]
    else:
        raise ValueError(f"args needs to have a page object provided")

    if not hasattr(page, args["operation"]):
            raise ValueError(f"Invalid operation: {args['operation']}")
    
    operation_args = args["operation"]

    if "content" in args["operation"]:
        return {"Please use the PageContent function to evaluate page content"}

    method = getattr(page, args["operation"])
    try:
        await method(*operation_args)
    except Exception as e:
        return {"error": "An error occured, if it was a TimeoutError, please fetch page content",
                "exception": str(e)}