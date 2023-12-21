import json
import os
import graphql_execute_chain

import chainlit as cl
from chainlit.input_widget import TextInput
from dotenv import load_dotenv
from langchain.agents import ZeroShotAgent, AgentType
from langchain.agents import initialize_agent, AgentExecutor
from langchain.chains import RetrievalQA, LLMChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.tools import Tool
from langchain.vectorstores.faiss import FAISS
from langchain_community.utilities.graphql import GraphQLAPIWrapper
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder

from database_vector import cached_embedder
from introspection import get_introspection_texts, aintrospect

load_dotenv()
GRAPHQL_DEFAULT_ENDPOINT = os.getenv("GRAPHQL_DEFAULT_ENDPOINT")


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    agent = cl.user_session.get("agent")
    agent.tools[2].graphql_wrapper = GraphQLAPIWrapper(
        custom_headers=json.loads(settings["headers"]),
        graphql_endpoint=settings["endpoint"])
    cl.user_session.set("agent", agent)


@cl.on_chat_start
async def start():
    # region Settings
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="endpoint",
                label="GraphQL endpoint",
                initial=GRAPHQL_DEFAULT_ENDPOINT,
                initial_index=0,
            ),
            TextInput(
                id="headers",
                label="GraphQL headers",
                initial="{}",
                initial_index=0,
            )
        ]
    ).send()
    # endregion
    # region Main Bot Setup
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
    }
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    read_only_memory = ReadOnlySharedMemory(memory=memory)
    # endregion
    # region Summary Setup
    summary_template = """This is a conversation between a human and a bot:

    {chat_history}

    Write a summary of the conversation for {input}:
    """
    summary_prompt = PromptTemplate(input_variables=["input", "chat_history"], template=summary_template)
    summary_chain = LLMChain(
        llm=llm,
        prompt=summary_prompt,
        verbose=True,
        memory=read_only_memory,  # use the read-only memory to prevent the tool from modifying the memory
    )
    # endregion
    # region GraphQL Setup
    introspection_result = await aintrospect(GRAPHQL_DEFAULT_ENDPOINT)
    introspection_texts = get_introspection_texts(introspection_result)
    introspection_db = await FAISS.afrom_documents(introspection_texts, cached_embedder)
    summary_template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {chat_history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=summary_template,
    )
    # retrieval_qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=introspection_db.as_retriever(search_kwargs={"k": 4}),
    #     verbose=True,
    #     memory=read_only_memory,
    #     chain_type_kwargs={
    #         "verbose": True,
    #         "prompt": prompt,
    #         "memory": read_only_memory,
    #     }
    # )
    # retrieval_qa_memory = ConversationSummaryMemory(
    #     llm=llm, memory_key="chat_history"
    # )
    retrieval_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=introspection_db.as_retriever()
    )
    # endregion

    # region Tools
    tools = [
        Tool(
            name="GraphQLIntrospect",
            func=retrieval_qa.run,
            coroutine=retrieval_qa.arun,
            description="Searches and returns relevant information from the GraphQL schema."
                        "Useful for finding the correct query to use, and searching the GraphQL"
                        "Data Collection tool. Always provide an input"
        ),
        Tool(
            name="GraphQLExecute",
            func=graphql_execute_chain.chain.invoke,
            description="Useful tool for executing graphql queries, input should be the users original request."
        ),
        Tool(
            name="Summary",
            func=summary_chain.run,
            description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
        ),

    ]
    # endregion

    # region Agent Setup
    # prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
    # suffix = """Begin!"
    #
    # {chat_history}
    # Question: {input}
    # {agent_scratchpad}"""
    # prompt = ZeroShotAgent.create_prompt(
    #     tools,
    #     prefix=prefix,
    #     suffix=suffix,
    #     input_variables=["input", "chat_history", "agent_scratchpad"]
    # )
    # llm_chain = LLMChain(llm=llm, prompt=prompt)
    # agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    # agent = AgentExecutor.from_agent_and_tools(
    #     agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
    # )
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_MULTI_FUNCTIONS,
        verbose=True,
        memory=memory,
        agent_kwargs=agent_kwargs,
        handle_parsing_errors=True,
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
