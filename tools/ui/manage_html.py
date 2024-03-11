""" 
This module contains the chain that will be used to produce the locators
for the HTML elements. The chain is composed of the following steps:

1. Load the HTML documents
2. Format the documents into a prompt
3. Run the prompt through the LLM
4. Parse the output of the LLM
"""

import json
import re
import os
from code import interact
from curses import meta
from typing import List
from langchain_community.document_loaders import BSHTMLLoader



from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from langchain.chains import create_retrieval_chain
from langchain.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser, JsonOutputFunctionsParser)
from langchain.prompts import Prompt
from langchain.text_splitter import (HTMLHeaderTextSplitter, Language,
                                     RecursiveCharacterTextSplitter)
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import BaseModel, Field

from ai_assisted_test_repo.openai.num_tokens_from_messages import \
    num_tokens_from_string
from tools.embeddings import cached_embedder
import tempfile


def condense_html(html: str) -> str:
    """
    Condenses the HTML content by keeping only essential elements and attributes.
    This helps in reducing the size of the HTML to be processed.

    :param html: The HTML content to be condensed.
    :return: A string representing the condensed HTML content.
    """
    soup = BeautifulSoup(html, 'html.parser')
    soup = soup.body if soup.body else soup

    for tag in soup(['script', 'style', 'head', 'meta', 'link']):
        tag.decompose()

    # Remove comments
    comments = soup.find_all(text=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    # Define allowed tags and their attributes
    default_tags =  ['class', 'id', 'href', 'data-testid', 'aria-label']
    allowed_attr_dict = {
        'a': ['href', 'title', 'name'] + default_tags,
        # 'img': ['src', 'alt'] + default_tags,
        'input': ['type', 'name', 'value'] + default_tags,
        'select': ['type', 'name', 'value'] + default_tags,
        'button': ['type', 'name', 'value'] + default_tags,
        'option': ['type', 'name', 'value'] + default_tags,
        'label': ['for'] + default_tags,
        'textarea': ['type', 'name', 'value'] + default_tags,
        'form': ['action'] + default_tags,
        'iframe': ['src', 'title', 'name'] + default_tags,
        # 'video': ['src', 'title', 'name'] + default_tags,
    }

    # Remove all tags not in allowed_tags, keep only allowed attributes for allowed tags
    for tag in soup.find_all(True):
        # Keep only allowed attributes for this tag, remove others
        allowed_attrs = allowed_attr_dict[tag.name] if tag.name in allowed_attr_dict else []
        for attr in list(tag.attrs):
            if attr not in allowed_attrs:
                del tag[attr]
        if tag.name not in allowed_attr_dict:
           # delete every attribute but keep the txt
              tag.attrs = {}

    # Collapse whitespace in text
    for tag in soup.find_all(text=True):
        new_text = ' '.join(tag.split())
        tag.replace_with(new_text)

    # unwrap tags that have no attributes
    for tag in soup.find_all(True):
        if not tag.attrs:
            tag.unwrap()

    return str(soup)

def get_page_content(html: str) -> str:
    """
    This function is used to get the page content from the html
    :param html: The html string that will be used to get the page content
    :return: The page content that is in the html
    """
    # Save the HTML content to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(html)
        temp_file_path = temp_file.name


    loader = BSHTMLLoader(temp_file_path)
    data = loader.load()
    # Remove the temporary file
    os.remove(temp_file_path)
    return data


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