""" 
This module contains the chain that will be used to produce the locators
for the HTML elements. The chain is composed of the following steps:

1. Load the HTML documents
2. Format the documents into a prompt
3. Run the prompt through the LLM
4. Parse the output of the LLM
"""

from code import interact
from curses import meta
import json
import re
from ai_assisted_test_repo.openai.num_tokens_from_messages import num_tokens_from_string
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from langchain.chains import create_retrieval_chain
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser, JsonOutputFunctionsParser
from langchain.prompts import Prompt
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List

from tools.embeddings import cached_embedder


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def condense_html(html: str) -> str:
    """
    Returns a list of beautiful soup objects that are the condensed html
    We do this because in some cases the html is too large to process
    in a single go. The maximum number of tokens that can be returned
    is 4096, so we want to make sure that the html is condensed to
    a similar size.
    :param html: The html that will be condensed
    :return: A list of beautiful soup objects that are the condensed html
    """

    soup = BeautifulSoup(html, 'html.parser')
    # just get the body
    soup = soup.body

    # Remove script, style, head, and meta tags
    for tag in soup(['script', 'style', 'head', 'meta', 'link']):
        tag.decompose()

    # Remove comments
    comments = soup.find_all(text=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    # Remove unnecessary attributes
    for tag in soup.find_all(True):
        allowed_attrs = ['class', 'id', 'href', 'data-testid', 'aria-label']
        # if id is in list remove class
        for attr in list(tag.attrs):
            if attr not in allowed_attrs:
                del tag[attr]

    # Collapse whitespace
    for tag in soup.find_all(text=True):
        tag.replace_with(' '.join(tag.split()))

    # Search for shadow dom elements
    shadow_dom = soup.find_all(lambda tag: tag.has_attr('shadowroot'))

    # return html back
    return str(soup)



# Just an example of how to use the vector store
if __name__ == "__main__":
    # This is an example of how to fetch the html from a url.
    # In most of our use cases, we will be supplying the html as a string
    # from playwrights page.content() method
    url = "https://academybugs.com/find-bugs/"
    loader = AsyncHtmlLoader([url])
    html = loader.load()
    # condense the html
    html = condense_html(html[0].page_content)