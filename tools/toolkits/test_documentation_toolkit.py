from tools.test_management import save_test, fetch_test, update_test

def get_tools():
    return [save_test.save_test_tool, fetch_test.create_test_tool(), update_test.update_test_tool]