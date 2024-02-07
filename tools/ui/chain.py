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
from tools.ui import condense_html

TEMPLATE = TEMPLATE = """
    Please provide a page object model for the following HTML:

    The page should include a summary of the HTML, the input fields, the buttons, and any other elements that are relevant to understanding the HTML.
    Any element that can be interacted with should include a locator that can be used by playwright. 

    HTML
    {html}

    Input Fields:
    {inputs}

    Buttons:
    {buttons}

"""


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")


def split_html(html: str) -> list[Document]:
    """
    This function is used to split the html into chunks
    :param html: The html string that will be split into chunks
    :return: A list of documents that are the chunks of the html
    """

    condensed_html = condense_html.condense_html(html)

    html_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML, chunk_size=200, chunk_overlap=50
    )
    html_docs = html_splitter.create_documents([str(condensed_html)])
    return html_docs


def get_buttons(html: str) -> str:
    """
    This function is used to get the buttons from the html
    :param html: The html string that will be used to get the buttons
    :return: A list of buttons that are in the html
    """
    soup = BeautifulSoup(html, "html.parser")
    buttons = soup.find_all("button")
    # return a string of the buttons
    return "\n".join([str(button) for button in buttons])
    


def get_inputs(html: str) -> str:
    """
    This function is used to get the inputs from the html
    :param html: The html string that will be used to get the inputs
    :return: A list of inputs that are in the html
    """
    soup = BeautifulSoup(html, "html.parser")
    inputs = soup.find_all("input")
    return "\n".join([str(input) for input in inputs])


def playwright_chain(html: str) -> Runnable:
    """
    This function is used to produce a playwright chain that
    will be used to produce the locators for the HTML elements
    :param html: The html string that will be used to produce the locators
    :return: A playwright chain that will be used to produce the locators
    """
    condensed_html = condense_html.condense_html(html)
    html_docs = split_html(condensed_html)

    buttons = get_buttons(html)
    inputs = get_inputs(html)

    us_vector_store = FAISS.from_documents(html_docs, cached_embedder)
    playwright_chain = (
        {
            "request": RunnablePassthrough(),
            "html": us_vector_store.as_retriever(
                search_kwargs={"k": 15, "fetch_k": 60}
            )
        }
        | PromptTemplate(input_variables=["request", "html"], template=TEMPLATE).partial(buttons=buttons, inputs=inputs)
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
