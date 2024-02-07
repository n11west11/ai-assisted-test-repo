from langchain_community.agent_toolkits import JiraToolkit

from langchain_community.utilities.jira import JiraAPIWrapper


jira = JiraAPIWrapper()
jira_toolkit = JiraToolkit.from_jira_api_wrapper(jira)
jira_tools = []
for tool in jira_toolkit.get_tools():
    # replace spaces with underscores
    tool.name = tool.name.replace(" ", "_")
    # for now we just want the JQL Query tool, and the project tool
    # the other tools are potentially dangerous
    if tool.name == "JQL_Query":
        jira_tools.append(tool)
    if tool.name == "Get_Projects":
        jira_tools.append(tool)
    if tool.name == "Create_Issue":
        jira_tools.append(tool)
