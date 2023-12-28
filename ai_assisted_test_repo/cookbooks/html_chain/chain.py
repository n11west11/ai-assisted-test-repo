""" 
This module contains the chain that will be used to produce the locators
for the HTML elements. The chain is composed of the following steps:

1. Load the HTML documents
2. Format the documents into a prompt
3. Run the prompt through the LLM
4. Parse the output of the LLM
"""
from re import split
from tabnanny import verbose

from langchain.document_loaders import AsyncHtmlLoader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from ai_assisted_test_repo.cookbooks.html_chain.cached_embedder import cached_embedder

TEMPLATE = """
    Using the HTML determine a locator that can be used to perform the request or set of requests. 
    The locator will be used in playwright to perform the action or set of actions.

    list your top 4 choices for the locator and the reason why you chose them. Use different ID's for each choice

    Request: 
    {request}

    HTML:
    {html}
"""


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")


def split_html(html: str) -> list[Document]:
    """
    This function is used to split the html into chunks
    :param html: The html string that will be split into chunks
    :return: A list of documents that are the chunks of the html
    """
    html_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML, chunk_size=1000, chunk_overlap=50
    )
    html_docs = html_splitter.create_documents([html])
    return html_docs


def playwright_chain(html: str) -> Runnable:
    """
    This function is used to produce a playwright chain that
    will be used to produce the locators for the HTML elements
    :param html: The html string that will be used to produce the locators
    :return: A playwright chain that will be used to produce the locators
    """
    html_docs = split_html(html)

    us_vector_store = FAISS.from_documents(html_docs, cached_embedder)
    playwright_chain = (
        {"request": RunnablePassthrough(), "html": us_vector_store.as_retriever(search_kwargs={"k":7, "fetch_k": 400})}
        | PromptTemplate(input_variables=["request", "html"], template=TEMPLATE)
        | llm
        | StrOutputParser()
    )
    return playwright_chain


# Just an example of how to use the vector store
if __name__ == "__main__":
    # This is an example of how to fetch the html from a url.
    # In most of our use cases, we will be supplying the html as a string
    # from playwrights page.content() method
    url = "https://academybugs.com/find-bugs/"
    loader = AsyncHtmlLoader([url])
    html = loader.load()
    html_docs = split_html(html[0].page_content)

    us_vector_store = FAISS.from_documents(html_docs, cached_embedder)

    us_result = us_vector_store.similarity_search(
        "What is the locator for the dark grey jeans add to cart button", fetch_k=40
    )
    print(us_result)

    print(
        playwright_chain(html[0].page_content)
        .with_config(callbacks=[StdOutCallbackHandler()])
        .invoke("What is the locator for the dark grey jeans add to cart button")
    )
