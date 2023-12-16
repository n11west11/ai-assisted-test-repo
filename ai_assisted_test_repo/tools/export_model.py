import json
import inspect
import os

from pydantic import BaseModel


def export_model(model: BaseModel):

    caller_frame = inspect.stack()[1]
    caller_path = caller_frame.filename
    caller_dir = os.path.dirname(os.path.abspath(caller_path))

    output_dir = os.path.join(caller_dir, "model_json_schema") 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{model.__name__}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        operation_execute = {
            "type": "function",
                "function": {
                    "name": f"{model.__name__}",
                    "description": model.__doc__,
                    "parameters": model.model_json_schema()
                }
        }
        json.dump(operation_execute, f, indent=4)
