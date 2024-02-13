import json
from typing import Any, Dict
from dns import query

from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from graphql import GraphQLList, GraphQLNonNull, GraphQLSchema, build_client_schema, get_introspection_query, print_schema
from graphql.type import introspection
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field
from tools.embeddings import cached_embedder

class GraphQLBaseParameters(BaseModel):
    headers: Dict[str, Any] = Field(
        {}, description="The headers to use in the GraphQL server"
    )
    url: str = Field(..., description="The GraphQL server url")


def type_to_sdl(field_type):
    """Convert a GraphQL field type to its SDL representation, simplified."""
    if hasattr(field_type, 'of_type'):
        # Handle wrapper types like NonNull or List
        inner_type = type_to_sdl(field_type.of_type)
        if isinstance(field_type, GraphQLNonNull):
            return f"{inner_type}!"
        elif isinstance(field_type, GraphQLList):
            return f"[{inner_type}]"
    else:
        return field_type.name

def field_to_sdl(field_name, field):
    """Convert a GraphQL field to its SDL representation."""
    field_type_sdl = type_to_sdl(field.type)
    args_sdl = ""
    if field.args:
        # Format the arguments in SDL
        args_sdl = "(" + ", ".join([f"{arg_name}: {type_to_sdl(arg.type)}" for arg_name, arg in field.args.items()]) + ")"
    return f"{field_name}{args_sdl}: {field_type_sdl}"

def schema_queries(schema: GraphQLSchema):
    """Extract and format the fields of the query type from a GraphQLSchema into a single SDL string."""
    query_fields_sdl = []

    for field_name, field in schema.query_type.fields.items():
        query_fields_sdl.append(field_to_sdl(field_name, field))

    # Join all field definitions into a single string, wrapped in 'type Query { ... }'
    sdl = "type Query {\n" + "\n".join(query_fields_sdl) + "\n}"

    return sdl

def schema_mutations(schema: GraphQLSchema):
    """Extract and format the fields of the mutation type from a GraphQLSchema into a single SDL string."""
    mutation_fields_sdl = []

    for field_name, field in schema.mutation_type.fields.items():
        mutation_fields_sdl.append(field_to_sdl(field_name, field))

    # Join all field definitions into a single string, wrapped in 'type Mutation { ... }'
    sdl = "type Mutation {\n" + "\n".join(mutation_fields_sdl) + "\n}"

    return sdl


def introspect(url: str, headers=None):
    """Introspect a GraphQL server to fetch high level information about the schema"""
    if headers is None:
        headers = {}
    try:
        transport = AIOHTTPTransport(url, headers=headers)
        graphql_client = Client(transport=transport, fetch_schema_from_transport=True)
        introspection = get_introspection_query()
        result = graphql_client.execute(gql(introspection))
        return build_client_schema(result)
    except Exception as e:
        return "Error: " + str(e)


async def aintrospect(url: str, headers=None):
    """Introspect a GraphQL server to fetch high level information about the schema"""
    if headers is None:
        headers = {}
    try:
        transport = AIOHTTPTransport(url, headers=headers)
        graphql_client = Client(transport=transport, fetch_schema_from_transport=True)
        introspection = get_introspection_query()
        result = await graphql_client.execute_async(gql(introspection))
        return build_client_schema(result)
    except Exception as e:
        return "Error: " + str(e)
    
def queries(schema: GraphQLSchema) -> list[str]:
    """
    This function is used to produce a list of queries from a graphql schema
    :param schema: The schema that will be used to produce the queries
    :return: A list of queries from a graphql schema
    """
    return json.dumps(schema_queries_to_dict(schema))

async def aqueries(schema: GraphQLSchema) -> list[str]:
    """
    This function is used to produce a list of queries from a graphql schema
    :param schema: The schema that will be used to produce the queries
    :return: A list of queries from a graphql schema
    """
    return json.dumps(schema_queries_to_dict(schema))

def mutations(schema: GraphQLSchema) -> list[str]:
    """
    This function is used to produce a list of mutations from a graphql schema
    :param schema: The schema that will be used to produce the mutations
    :return: A list of mutations from a graphql schema
    """
    return json.dumps(schema_mutations_to_dict(schema))

async def amutations(schema: GraphQLSchema) -> list[str]:
    """
    This function is used to produce a list of mutations from a graphql schema
    :param schema: The schema that will be used to produce the mutations
    :return: A list of mutations from a graphql schema
    """
    return json.dumps(schema_mutations_to_dict(schema))


def get_introspection_texts(schema: GraphQLSchema) -> list[Document]:
    query = print_schema(schema)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.create_documents([query])
    return documents


async def aget_introspection_texts(schema: GraphQLSchema) -> list[Document]:
    query = print_schema(schema)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.create_documents([query])
    return documents


def introspection_db(schema: GraphQLSchema) -> VectorStore:
    """
    This function is used to produce a vector store that
    will be used as context for the graphql query
    :param endpoint: The endpoint that will be used to produce the vector store
    :return: A vector store that will be used as context for the graphql query
    """
    introspection_texts = get_introspection_texts(schema)
    db = FAISS.from_documents(introspection_texts, cached_embedder)
    return db


async def aintrospection_db(schema: GraphQLSchema) -> VectorStore:
    """
    This function is used to produce a vector store that
    will be used as context for the graphql query
    :param endpoint: The endpoint that will be used to produce the vector store
    :return: A vector store that will be used as context for the graphql query
    """
    introspection_texts = get_introspection_texts(schema)
    db = await FAISS.afrom_documents(introspection_texts, cached_embedder)
    return db
