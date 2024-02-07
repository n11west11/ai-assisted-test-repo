import chainlit as cl
from chainlit.input_widget import TextInput
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.globals import set_debug
from langchain.tools import format_tool_to_openai_function
from langchain_community.llms import ollama
# from langchain_experimental.llms import ollama_functions
from langchain_openai import ChatOpenAI

from ai_assisted_test_repo.cookbooks.aws_ssm_assistant.aws_parameter_manager import \
    AWSParameterManager
from ai_assisted_test_repo.cookbooks.aws_ssm_assistant.create_conversational_retrieval_agent import \
    create_conversational_retrieval_agent

load_dotenv()

set_debug(True)


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    


@cl.on_chat_start
async def start():
    # region Settings
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="profile",
                label="AWS Profile",
                initial="default",
            )
        ]
    ).send()
    # endregion
    # region Bot Setup
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613", streaming=True)
    # llm = ollama_functions.OllamaFunctions(model="llama2")
    # endregion

    aws_parameter_manager = AWSParameterManager()

    # region Tools
    tools = [
        aws_parameter_manager.get_parameters_by_path_tool(),
    ]
    for tool in tools:
        tool_openai = format_tool_to_openai_function(tool)
        llm.bind(functions=[tool_openai], function_call={"name": tool_openai["name"]})
    # endregion

    # region Agent Setup
    agent = create_conversational_retrieval_agent(
        llm,
        tools,
        verbose=True,
        max_token_limit=5000,
        handle_parsing_errors=True,
        remember_intermediate_steps=False
    )
    cl.user_session.set("agent", agent)
    # endregion


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    res = await agent.arun(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res).send()
