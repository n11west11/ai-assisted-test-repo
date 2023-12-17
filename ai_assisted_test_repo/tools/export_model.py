import json
import inspect
import os

from pydantic import BaseModel
from ai_assisted_test_repo.tools.ui.registry import function_registry


def export_model(model: BaseModel):

    caller_frame = inspect.stack()[1]
    caller_path = caller_frame.filename
    caller_dir = os.path.dirname(os.path.abspath(caller_path))

    output_dir = os.path.join(caller_dir, "model_json_schema") 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    function_name = function_registry[model.__name__].__name__
    file_path = os.path.join(output_dir, f"{function_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        operation_execute = {
            "type": "function",
                "function": {
                    "name": function_name,
                    "description": model.__doc__,
                    "parameters": model.model_json_schema()
                }
        }
        json.dump(operation_execute, f, indent=4)
