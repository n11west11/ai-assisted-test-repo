from abc import abstractmethod

import chainlit as cl


class BaseProfile:
    """Expands the Chat profile to include a initialize and process_message method"""

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def process_message(self, message: cl.Message):
        pass