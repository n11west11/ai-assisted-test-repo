from dataclasses import Field
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, constr
# from data import path as data_path

class GetParametersByPathInput(BaseModel):
    path: constr(regex=r'^/(default|development|staging|production)$') = Field(..., description="The path to search for parameters.")
    recursive: bool = Field(True, description="Whether to recursively search for parameters under the given path.")
    


class AWSParameterManager:
        

    @staticmethod
    def _get_parameters_by_path(path: str, recursive: bool = True):
        """Get all parameters from the AWS Parameter Store under a given path."""
        try:
            session = boto3.Session()
            ssm = session.client('ssm')
            paginator = ssm.get_paginator('get_parameters_by_path')
            page_iterator = paginator.paginate(
                Path=path,
                WithDecryption=True,
                Recursive=recursive
            )

            parameters = []
            for page in page_iterator:
                parameters.extend(page['Parameters'])

            parameters = {param['Name']: param['Value'] for param in parameters}
            # pretty print the parameters
            parameters = '\n'.join([f"{key}: {value}" for key, value in parameters.items()])
            return str(parameters)
        except ClientError as e:
             raise ToolException(f"An error occurred: {e}")
        
    
    def get_parameters_by_path_tool(self):
        return StructuredTool.from_function(
            self._get_parameters_by_path,
            args_schema=GetParametersByPathInput, 
            handle_tool_error=True,
            return_direct=True
        )
