from abc import ABC, abstractmethod

class BaseComponent(ABC):
    def run(self, *args, **kwargs):
        ...

    async def arun(self, *args, **kwargs):
        ...
