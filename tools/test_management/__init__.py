from cgi import test
from .fetch_test import test_retriever_tool
from .save_test import save_test_tool
from .update_test import update_test_tool

test_management_tools = [test_retriever_tool, save_test_tool, update_test_tool]