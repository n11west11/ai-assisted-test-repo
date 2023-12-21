from typing import Dict, Any

from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from graphql import get_introspection_query, print_schema, build_client_schema
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class GraphQLBaseParameters(BaseModel):
    headers: Dict[str, Any] = Field({}, description="The headers to use in the GraphQL server")
    url: str = Field(..., description="The GraphQL server url")


def introspect(url: str, headers=None):
    """Introspect a GraphQL server to fetch high level information about the schema"""
    if headers is None:
        headers = {}
    try:
        transport = AIOHTTPTransport(
            url,
            headers=headers
        )
        graphql_client = Client(transport=transport, fetch_schema_from_transport=True)
        introspection = get_introspection_query()
        result = graphql_client.execute(gql(introspection))
        schema_str = print_schema(build_client_schema(result))
        return schema_str
    except Exception as e:
        return "Error: " + str(e)


async def aintrospect(url: str, headers=None):
    """Introspect a GraphQL server to fetch high level information about the schema"""
    if headers is None:
        headers = {}
    try:
        transport = AIOHTTPTransport(
            url,
            headers=headers
        )
        graphql_client = Client(transport=transport, fetch_schema_from_transport=True)
        introspection = get_introspection_query()
        result = await graphql_client.execute_async(gql(introspection))
        schema_str = print_schema(build_client_schema(result))
        return schema_str
    except Exception as e:
        return "Error: " + str(e)


def get_introspection_texts(query: str) -> list[Document]:
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.create_documents([query])
    return documents


async def aget_introspection_texts(query: str) -> list[Document]:
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.create_documents([query])
    return documents
