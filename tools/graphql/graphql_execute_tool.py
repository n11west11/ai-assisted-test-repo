from __future__ import annotations

import json
import os
from typing import Dict, Optional, Type

from dotenv import load_dotenv
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, validator

from ai_assisted_test_repo.cookbooks.graphql_assistant.introspection import *

load_dotenv()


class ToolInputSchema(BaseModel):
    """Input schema for the GraphQLExecute tool"""

    query: str = Field(description="The users request")
    

class GraphQLExecuteTool(BaseTool):
    GRAPHQL_DEFAULT_ENDPOINT = os.getenv("GRAPHQL_DEFAULT_ENDPOINT")
    endpoint: str = Field(
        default=GRAPHQL_DEFAULT_ENDPOINT,
        description="The endpoint to use for the GraphQL API",
    )
    custom_headers: Dict[str, str] | None = Field(
        default={},
        description="Custom headers to use for the GraphQL API",
    )
    name = "GraphQLExecute"
    description: str = """A tool to execute a GraphQL query on a server. Be careful asking for too much data! Try to keep your queries small and focused."""
    args_schema: Type[BaseModel] = ToolInputSchema

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        try:
            client = Client(
                transport=AIOHTTPTransport(
                    url=self.endpoint, headers=self.custom_headers
                ),
                fetch_schema_from_transport=True,
            )
            result = client.execute(gql(query))
            return result
        except Exception as e:
            raise ToolException(f"Error running GraphQLExecute tool: {e}")

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        try:
            client = Client(
                transport=AIOHTTPTransport(
                    url=self.endpoint, headers=self.custom_headers
                ),
                fetch_schema_from_transport=True,
            )
            result = await client.execute_async(gql(query))
            return result
        except Exception as e:
            raise ToolException(f"Error running GraphQLExecute tool: {e}")
