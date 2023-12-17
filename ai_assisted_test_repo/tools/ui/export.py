
import os


from ai_assisted_test_repo.tools.export_model import export_model
from ai_assisted_test_repo.tools.ui.handlers.execute_playwright_operation import ExecutePlaywrightOperation
from ai_assisted_test_repo.tools.ui.handlers.page_content import PageContentArgs


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(MODULE_DIR, "model_json_schema")

if __name__ == "__main__":
    # export models to json

    for model in [
        ExecutePlaywrightOperation,
        PageContentArgs,
    ]:
        export_model(model)
