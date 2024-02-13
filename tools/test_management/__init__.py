from cgi import test
from .fetch_test import create_test_tool
from .save_test import save_test_tool
from .update_test import update_test_tool

test_management_tools = [create_test_tool(), save_test_tool, update_test_tool]