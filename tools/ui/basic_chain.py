""" 
This module contains the chain that will be used to produce the locators
for the HTML elements. The chain is composed of the following steps:

1. Load the HTML documents
2. Format the documents into a prompt
3. Run the prompt through the LLM
4. Parse the output of the LLM
"""

from bs4 import BeautifulSoup
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from tools.embeddings import cached_embedder
from tools.ui import manage_html

TEMPLATE = """
    Given the following HTML, and plan, please provide a page object model that a test engineer can use to interact with the page.
    The page object model should include the html, and be a json object that is a dictionary of the locators for the input fields and buttons. 
    Assume that the test engineer is using Python, and playwright.

    If the fields are not present, please provide a message that the fields are not present.

    Input Fields:
    {inputs}

    Buttons:
    {buttons}

    Links:
    {links}

    inputs:
    {inputs}

    select:
    {select}

    textarea:
    {textarea}

    form:
    {form}

    iframe:
    {iframe}

    video:
    {video}



    

"""


llm = ChatOpenAI(temperature=0, model="gpt-4")


def split_html(html: str) -> list[Document]:
    """
    This function is used to split the html into chunks
    :param html: The html string that will be split into chunks
    :return: A list of documents that are the chunks of the html
    """

    condensed_html = manage_html.condense_html(html)

    html_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML, chunk_size=1000, chunk_overlap=50
    )
    html_docs = html_splitter.create_documents([str(condensed_html)])
    return html_docs


def html_vector_store(html: str) -> FAISS:
    """
    This function is used to produce a FAISS vector store from the html
    :param html: The html string that will be used to produce the vector store
    :return: A FAISS vector store that will be used to produce the locators
    """
    html_docs = split_html(html)
    vector_store = FAISS.from_documents(html_docs, cached_embedder)
    return vector_store


def get_html(tag: str, html: str) -> str:
    """
    This function is used to get the inputs from the html
    :param html: The html string that will be used to get the inputs
    :return: A list of inputs that are in the html
    """
    soup = BeautifulSoup(html, "html.parser")
    inputs = soup.find_all(tag)
    return "\n".join([str(input) for input in inputs])


def playwright_chain(html: str, url: str) -> Runnable:
    """
    This function is used to produce a playwright chain that
    will be used to produce the locators for the HTML elements
    :param html: The html string that will be used to produce the locators
    :return: A playwright chain that will be used to produce the locators
    """
    condensed_html = manage_html.condense_html(html)

    buttons = get_html("button", condensed_html)
    inputs = get_html("input", condensed_html)
    links = get_html("a", condensed_html)
    select = get_html("select", condensed_html)
    textarea = get_html("textarea", condensed_html)
    form = get_html("form", condensed_html)
    iframe = get_html("iframe", condensed_html)
    video = get_html("video", condensed_html)

    prompt = PromptTemplate(input_variables=["request", "html"], template=TEMPLATE)
    prompt.partial(url=url)
    prompt = prompt.partial(
        buttons=buttons,
        inputs=inputs,
        links=links,
        select=select,
        textarea=textarea,
        form=form,
        iframe=iframe,
        video=video,
    )

    playwright_chain = (
        {
            "request": RunnablePassthrough(),
        }
        | prompt
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
